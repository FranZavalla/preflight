use miette::{Context, IntoDiagnostic, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::runtime::Handle;

use crate::model::{
    FunctionContextEntry, MiniPrompt, PermissionPromptSpec, SkillIterationResult, TokenUsage,
    ValidatorContextMap, VulnerabilityFinding, VulnerabilitySkill,
};

/// Default per-skill agent step cap, shared by the CLI default and the
/// interactive "how many more steps" default.
pub(crate) const DEFAULT_MAX_AGENT_STEPS: usize = 30;
/// Maximum time one skill may spend before it is abandoned.
pub(crate) const DEFAULT_SKILL_TIMEOUT: Duration = Duration::from_secs(10 * 60);
/// Maximum time to wait for a read from the model endpoint.
pub(crate) const DEFAULT_REQUEST_READ_TIMEOUT: Duration = Duration::from_secs(120);
/// Maximum time to establish a model connection.
pub(crate) const DEFAULT_REQUEST_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Default cap on the bytes of local tool output handed to the model.
///
/// A single `read_file` on a large validator, or a loose `grep`, can otherwise
/// put hundreds of kilobytes into the conversation and every later step resends
/// them. Larger files are still readable through a narrower request.
pub(crate) const DEFAULT_MAX_TOOL_OUTPUT_BYTES: usize = 32 * 1024;
/// Tool results kept verbatim in the conversation.
const MAX_RETAINED_TOOL_RESULTS: usize = 6;

/// Tool results tolerated before a prune runs.
///
/// Pruning rewrites older messages, which invalidates provider prompt caching
/// at or before the rewrite. Draining down to `MAX_RETAINED_TOOL_RESULTS` only
/// after twice that many have accumulated pays that invalidation once per six
/// steps instead of once per step.
const TOOL_RESULT_PRUNE_TRIGGER: usize = MAX_RETAINED_TOOL_RESULTS * 2;
/// Consecutive unparseable replies tolerated before a skill is abandoned.
const MAX_MALFORMED_REPLIES: usize = 3;
const MALFORMED_REPLY_NUDGE: &str = "Your last message was not valid JSON. \
Reply with exactly one JSON object and nothing else - no prose, no code fences, \
no commentary. Use one of the documented actions (read_file, grep, list_dir, \
find_files, final).";

/// Directory names that are always outside the audit scope. This mirrors the
/// skip list in `discover_source_files` (`crate::lib`) so that AI tool reads
/// cannot reach what source discovery excluded.
pub(crate) const EXCLUDED_DIR_NAMES: &[&str] = &[".git", "target", ".tx3", "build"];

/// Returns `true` when `canonical_path` lies inside a directory that source
/// discovery excludes: a `.git`/`target`/`.tx3`/`build` directory anywhere
/// along the path, or a nested project (a non-root directory containing its own
/// `aiken.toml`).
///
/// `canonical_path` must already be canonicalized and under `project_root`.
pub(crate) fn is_ignored_path(project_root: &Path, canonical_path: &Path) -> bool {
    let relative = match canonical_path.strip_prefix(project_root) {
        Ok(relative) => relative,
        Err(_) => return false,
    };

    let mut ancestor = project_root.to_path_buf();
    for component in relative.components() {
        let name = component.as_os_str();
        if EXCLUDED_DIR_NAMES
            .iter()
            .any(|excluded| name == std::ffi::OsStr::new(excluded))
        {
            return true;
        }
        ancestor.push(component);
        if ancestor != project_root && ancestor.join("aiken.toml").is_file() {
            return true;
        }
    }
    false
}

const AGENT_SYSTEM_PROMPT: &str =
    include_str!("../../templates/aiken/audit_agent_system_prompt.md");
const INITIAL_USER_PROMPT_TEMPLATE: &str =
    include_str!("../../templates/aiken/audit_agent_initial_user_prompt.md");
const PERMISSION_PROMPT_TEMPLATE: &str = include_str!("../../templates/aiken/permission_prompt.md");
const TOOL_RESULT_PROMPT_TEMPLATE: &str =
    include_str!("../../templates/aiken/audit_agent_tool_result_prompt.md");

#[derive(Debug)]
pub(super) enum AgentAction {
    Final(Value),
    ReadRequest(ReadRequest),
}

#[derive(Debug)]
pub(super) enum ReadRequest {
    ReadFile {
        path: String,
    },
    Grep {
        pattern: String,
        path: String,
        context_lines: usize,
    },
    ListDir {
        path: String,
    },
    FindFiles {
        path: String,
        glob: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
struct RawReadRequest {
    action: Option<String>,
    path: Option<String>,
    pattern: Option<String>,
    context_lines: Option<usize>,
    glob: Option<String>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ReasoningStreamState {
    pub(super) started: bool,
    pub(super) line_break_emitted: bool,
    pub(super) last_summary_index: Option<i64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ContentStreamState {
    pub(super) started: bool,
    pub(super) ends_with_newline: bool,
}

pub(super) fn stream_reasoning_delta_to_stdout(
    enabled: crate::logging::LogLevel,
    state: &mut ReasoningStreamState,
    delta: &str,
) {
    if !enabled.is_full() || delta.is_empty() {
        return;
    }

    let mut stdout = io::stdout().lock();

    if !state.started {
        let _ = writeln!(stdout, "🤖 🧠 Reasoning summary:");
        state.started = true;
    }

    let _ = write!(stdout, "{}", delta);
    let _ = stdout.flush();
    state.line_break_emitted = false;
}

pub(super) fn emit_reasoning_line_break(
    enabled: crate::logging::LogLevel,
    state: &mut ReasoningStreamState,
) {
    if !enabled.is_full() || !state.started || state.line_break_emitted {
        return;
    }

    let mut stdout = io::stdout().lock();
    let _ = writeln!(stdout);
    let _ = stdout.flush();
    state.line_break_emitted = true;
}

pub(super) fn emit_reasoning_double_line_break(
    enabled: crate::logging::LogLevel,
    state: &mut ReasoningStreamState,
) {
    if !enabled.is_full() || !state.started || state.line_break_emitted {
        return;
    }

    let mut stdout = io::stdout().lock();
    let _ = write!(stdout, "\n\n");
    let _ = stdout.flush();
    state.line_break_emitted = true;
}

pub(super) fn finalize_reasoning_stdout(
    enabled: crate::logging::LogLevel,
    state: &mut ReasoningStreamState,
) {
    emit_reasoning_line_break(enabled, state);
}

pub(super) fn stream_content_delta_to_stdout(
    enabled: crate::logging::LogLevel,
    state: &mut ContentStreamState,
    delta: &str,
) {
    if !enabled.is_full() || delta.is_empty() {
        return;
    }

    let mut stdout = io::stdout().lock();

    if !state.started {
        let _ = write!(stdout, "🤖 ↳ Output: ");
        state.started = true;
        state.ends_with_newline = false;
    }

    let _ = write!(stdout, "{}", delta);
    let _ = stdout.flush();

    state.ends_with_newline = delta.ends_with('\n');
}

pub(super) fn finalize_content_stdout(
    enabled: crate::logging::LogLevel,
    state: &mut ContentStreamState,
) {
    if !enabled.is_full() || !state.started || state.ends_with_newline {
        return;
    }

    let mut stdout = io::stdout().lock();
    let _ = writeln!(stdout);
    let _ = stdout.flush();
    state.ends_with_newline = true;
}

pub(super) fn build_agent_system_prompt() -> &'static str {
    AGENT_SYSTEM_PROMPT
}

fn parse_line_number(value: Option<&Value>) -> Option<usize> {
    value.and_then(|entry| {
        if let Some(number) = entry.as_u64() {
            return usize::try_from(number).ok();
        }

        entry
            .as_str()
            .and_then(|text| text.trim().parse::<usize>().ok())
    })
}

pub(super) fn build_initial_user_prompt(
    prompt: &MiniPrompt,
    source_references: &[String],
    validator_context: &ValidatorContextMap,
    permission_prompt: &PermissionPromptSpec,
) -> String {
    INITIAL_USER_PROMPT_TEMPLATE
        .replace("{{SKILL}}", &prompt.text)
        .replace(
            "{{SOURCE_REFERENCES}}",
            &render_source_references(source_references),
        )
        .replace(
            "{{VALIDATOR_CONTEXT_MAP}}",
            &render_validator_context_map(validator_context),
        )
        .replace(
            "{{PERMISSION_PROMPT}}",
            &render_permission_prompt(permission_prompt),
        )
}

pub(super) fn build_tool_result_user_prompt(
    request: &ReadRequest,
    output: &str,
    max_output_bytes: usize,
) -> String {
    TOOL_RESULT_PROMPT_TEMPLATE
        .replace("{{REQUEST}}", &summarize_read_request(request))
        .replace(
            "{{OUTPUT}}",
            &truncate_tool_output(output, max_output_bytes),
        )
}

/// Caps one tool result so a single oversized read or grep cannot dominate the
/// conversation. `0` disables the cap.
pub(super) fn truncate_tool_output(output: &str, max_output_bytes: usize) -> String {
    if max_output_bytes == 0 || output.len() <= max_output_bytes {
        return output.to_string();
    }

    let mut cut = max_output_bytes;
    while cut > 0 && !output.is_char_boundary(cut) {
        cut -= 1;
    }

    // Prefer a line boundary so the model never reads half a line of Aiken as
    // if it were complete. The byte cut is a fallback when no line break is
    // reasonably close to the budget.
    let cut = output[..cut]
        .rfind('\n')
        .filter(|newline| newline.saturating_mul(2) >= cut)
        .map(|newline| newline + 1)
        .unwrap_or(cut);

    format!(
        "{}\n… (truncated: {} of {} bytes shown. Re-request with a narrower \
scope -- a tighter grep pattern, a more specific path -- if you need the rest. \
Do not treat what you have as the complete output.)",
        &output[..cut],
        cut,
        output.len()
    )
}

pub(super) struct AgentLoopContext<'a> {
    pub(super) endpoint: &'a str,
    pub(super) ai_logs: crate::logging::LogLevel,
    pub(super) project_root: &'a Path,
    pub(super) permission_prompt: &'a PermissionPromptSpec,
    pub(super) provider_label: &'a str,
    pub(super) skill_timeout: Duration,
    pub(super) max_agent_steps: usize,
    pub(super) max_tool_output_bytes: usize,
}

/// The user declined to keep iterating; the whole audit must stop rather than
/// moving on to the next skill.
#[derive(Debug)]
pub(crate) struct UserStopped;

impl std::fmt::Display for UserStopped {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("audit stopped by user at the agent step limit")
    }
}

impl std::error::Error for UserStopped {}

impl miette::Diagnostic for UserStopped {}

pub(super) fn run_agent_loop<F>(
    skill: &VulnerabilitySkill,
    context: AgentLoopContext<'_>,
    messages: &mut Vec<Value>,
    mut request_model: F,
) -> Result<SkillIterationResult>
where
    F: FnMut(&[Value]) -> Result<(String, TokenUsage)>,
{
    let AgentLoopContext {
        endpoint,
        ai_logs,
        project_root,
        permission_prompt,
        provider_label,
        skill_timeout,
        max_agent_steps,
        max_tool_output_bytes,
    } = context;

    let deadline = Instant::now() + skill_timeout;
    let interactive = io::stdin().is_terminal();
    let mut max_steps = max_agent_steps;
    let mut step_idx = 0usize;
    let mut total_usage = TokenUsage::default();
    let mut malformed_replies = 0usize;
    let mut tool_cache = ToolResultCache::default();

    loop {
        if Instant::now() >= deadline {
            return Err(miette::miette!(
                "{} exceeded the {}s skill budget for '{}' after {} step(s)",
                provider_label,
                skill_timeout.as_secs(),
                skill.id,
                step_idx
            ));
        }

        if step_idx >= max_steps {
            if let Some(additional) = prompt_for_additional_agent_steps(
                provider_label,
                &skill.id,
                max_steps,
                interactive,
            )? {
                max_steps = max_steps.saturating_add(additional);
                log_agent_progress(
                    ai_logs,
                    format!(
                        "Continuing skill '{}' with {} additional steps (new max={})",
                        skill.id, additional, max_steps
                    ),
                );
                continue;
            }

            if interactive {
                return Err(miette::Report::new(UserStopped));
            }

            return Err(miette::miette!(
                "{} exceeded max agent steps ({}) for skill '{}'; raise --max-agent-steps to continue",
                provider_label,
                max_steps,
                skill.id
            ));
        }

        log_full(
            ai_logs,
            format!(
                "step {}/{}, requesting '{}' from {}",
                step_idx + 1,
                max_steps,
                skill.id,
                endpoint
            ),
        );

        // Timed before the request so the brief line covers both the model
        // latency and the local tool call, which is what a slow step costs.
        let step_started_at = Instant::now();
        let response_result = request_model(messages.as_slice());
        let elapsed = step_started_at.elapsed();

        if let Err(error) = &response_result {
            log_agent_progress(
                ai_logs,
                format!(
                    "step {}/{} failed after {}: {}",
                    step_idx + 1,
                    max_steps,
                    crate::logging::format_duration(elapsed),
                    error
                ),
            );
        }

        let (content, step_usage) = response_result?;
        total_usage.add(&step_usage);

        // The brief line is completed after the action is known, so one line per
        // step carries the timing, the action and the bytes it read.
        messages.push(json!({
            "role": "assistant",
            "content": content,
        }));

        log_full(ai_logs, format!("model output:\n{}", &content));

        let action = match parse_agent_action(&content) {
            Ok(action) => {
                malformed_replies = 0;
                action
            }
            // The model has no native tool-calling here, so it can simply emit
            // prose or garbage. Correct it and retry the step; abandoning 19 other
            // skills over one bad reply is far worse than spending another step.
            Err(error) => {
                malformed_replies = malformed_replies.saturating_add(1);

                if malformed_replies > MAX_MALFORMED_REPLIES {
                    return Err(error).wrap_err(format!(
                        "{} returned {} unparseable replies in a row for skill '{}'",
                        provider_label, malformed_replies, skill.id
                    ));
                }

                log_agent_progress(
                    ai_logs,
                    format!(
                        "step {}/{} unparseable reply ({}/{}), asking the model to retry: {}",
                        step_idx + 1,
                        max_steps,
                        malformed_replies,
                        MAX_MALFORMED_REPLIES,
                        error
                    ),
                );

                messages.push(json!({
                    "role": "user",
                    "content": MALFORMED_REPLY_NUDGE,
                }));

                step_idx = step_idx.saturating_add(1);
                continue;
            }
        };

        match action {
            AgentAction::Final(parsed) => {
                let findings = parsed
                    .get("findings")
                    .and_then(Value::as_array)
                    .map(|items| items.len())
                    .unwrap_or(0);
                let status = parsed
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("completed");
                let analysis_summary = parsed
                    .get("analysis_summary")
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty());

                if let Some(summary) = analysis_summary {
                    log_full(ai_logs, format!("analysis summary:\n{}", summary));
                }

                log_step(
                    ai_logs,
                    format!(
                        "step {}/{}  final              status={} findings={}  {}{}",
                        step_idx + 1,
                        max_steps,
                        status,
                        findings,
                        crate::logging::format_duration(step_started_at.elapsed()),
                        format_usage_suffix(&step_usage)
                    ),
                );

                log_agent_progress(
                    ai_logs,
                    format!(
                        "{} completed at step {}/{}, findings={}, tokens=({})",
                        skill.id,
                        step_idx + 1,
                        max_steps,
                        findings,
                        format_token_usage_inline(&total_usage)
                    ),
                );

                return Ok(iteration_from_parsed(skill, parsed, total_usage));
            }
            AgentAction::ReadRequest(request) => {
                log_full(
                    ai_logs,
                    format!("running {}", describe_read_request_friendly(&request)),
                );

                let output = execute_read_request_cached(
                    &mut tool_cache,
                    &request,
                    project_root,
                    permission_prompt,
                    step_idx + 1,
                )
                .unwrap_or_else(|error| format!("Request failed: {}", error));

                log_full(
                    ai_logs,
                    format!(
                        "tool output:\n{}",
                        render_tool_output_for_log(&request, &output)
                    ),
                );

                let was_truncated =
                    max_tool_output_bytes > 0 && output.len() > max_tool_output_bytes;
                let sent_bytes = if was_truncated {
                    max_tool_output_bytes
                } else {
                    output.len()
                };

                log_step(
                    ai_logs,
                    format!(
                        "step {}/{}  {:<32} {:>8}{}  {}{}",
                        step_idx + 1,
                        max_steps,
                        truncate_action(&summarize_read_request(&request)),
                        crate::logging::format_bytes(sent_bytes),
                        if was_truncated { " (capped)" } else { "" },
                        crate::logging::format_duration(step_started_at.elapsed()),
                        format_usage_suffix(&step_usage)
                    ),
                );

                messages.push(json!({
                    "role": "user",
                    "content": build_tool_result_user_prompt(
                        &request,
                        &output,
                        max_tool_output_bytes,
                    ),
                }));

                prune_stale_tool_results(messages, ai_logs);
            }
        }

        step_idx = step_idx.saturating_add(1);
    }
}

fn prompt_for_additional_agent_steps(
    provider_label: &str,
    skill_id: &str,
    max_steps: usize,
    interactive: bool,
) -> Result<Option<usize>> {
    // Non-interactive runs must never block on stdin. Raise --max-agent-steps
    // instead of prompting when there is no terminal to answer.
    if !interactive {
        return Ok(None);
    }

    eprintln!(
        "[audit][agent] {} reached the current max steps ({}) for skill '{}'",
        provider_label, max_steps, skill_id
    );
    eprint!("Continue iterating? [y/N] (N stops the whole audit): ");
    io::stderr().flush().into_diagnostic()?;

    let mut answer = String::new();
    if io::stdin().read_line(&mut answer).into_diagnostic()? == 0 {
        return Ok(None);
    }

    let accepted = matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes");
    if !accepted {
        return Ok(None);
    }

    eprint!(
        "How many additional steps? [default {}; 0 stops the whole audit]: ",
        max_steps
    );
    io::stderr().flush().into_diagnostic()?;

    let mut additional = String::new();
    if io::stdin().read_line(&mut additional).into_diagnostic()? == 0 {
        return Ok(None);
    }

    Ok(additional_steps_from_answer(&additional, max_steps))
}

/// Empty input means "use the effective cap again". Zero or unparseable input
/// means the user declined, which stops the whole audit.
fn additional_steps_from_answer(answer: &str, max_steps: usize) -> Option<usize> {
    let answer = answer.trim();

    if answer.is_empty() {
        return (max_steps > 0).then_some(max_steps);
    }

    match answer.parse::<usize>() {
        Ok(0) | Err(_) => None,
        Ok(value) => Some(value),
    }
}

fn render_permission_prompt(permission_prompt: &PermissionPromptSpec) -> String {
    PERMISSION_PROMPT_TEMPLATE
        .replace("{{ workspace_root }}", &permission_prompt.workspace_root)
        .replace(
            "{{ allowed_commands }}",
            &permission_prompt.allowed_commands.join(", "),
        )
        .replace(
            "{{ scope_rules }}",
            &permission_prompt.scope_rules.join("\n- "),
        )
}

fn render_source_references(source_references: &[String]) -> String {
    if source_references.is_empty() {
        return "- (none)".to_string();
    }

    source_references
        .iter()
        .map(|path| format!("- {}", path))
        .collect::<Vec<String>>()
        .join("\n")
}

fn render_validator_context_map(validator_context: &ValidatorContextMap) -> String {
    let validators_rendered = validator_context
        .validators
        .iter()
        .map(|validator| {
            let handlers = validator
                .handlers
                .iter()
                .map(|handler| {
                    let signature = handler
                        .parameters
                        .iter()
                        .map(|parameter| format!("{}: {}", parameter.name, parameter.r#type))
                        .collect::<Vec<String>>()
                        .join(", ");

                    format!("  - `{}({})`", handler.name, signature)
                })
                .collect::<Vec<String>>()
                .join("\n");

            format!(
                "- `{}`\n  - source: `{}`\n{}",
                validator.id, validator.source_file, handlers
            )
        })
        .collect::<Vec<String>>()
        .join("\n");

    let helpers_rendered = render_helper_context(&validator_context.functions);

    match (validators_rendered.is_empty(), helpers_rendered.is_empty()) {
        (true, true) => "- (none)".to_string(),
        (false, true) => validators_rendered,
        (true, false) => helpers_rendered,
        (false, false) => format!("{}\n\n{}", validators_rendered, helpers_rendered),
    }
}

/// Signatures only, grouped by source file. Bodies are read on demand by the
/// model; this block exists to make helpers discoverable without guessing which
/// file to open.
///
/// This caps the helper block only. The skill, source references, validator
/// map, and permission prompt are separate sections and are not covered by a
/// global prompt budget.
const MAX_HELPER_CONTEXT_CHARS: usize = 4 * 1024;

fn render_helper_context(functions: &[FunctionContextEntry]) -> String {
    if functions.is_empty() {
        return String::new();
    }

    let mut groups: BTreeMap<&str, Vec<&FunctionContextEntry>> = BTreeMap::new();
    for function in functions {
        groups
            .entry(function.source_file.as_str())
            .or_default()
            .push(function);
    }

    let mut lines = vec!["Top-level functions (helpers):".to_string()];
    for (source_file, entries) in groups {
        lines.push(format!("- `{}`", source_file));
        for function in entries {
            let name = function
                .id
                .rsplit_once('.')
                .map(|(_, name)| name)
                .unwrap_or(&function.id);
            let signature = function
                .parameters
                .iter()
                .map(|parameter| format!("{}: {}", parameter.name, parameter.r#type))
                .collect::<Vec<String>>()
                .join(", ");
            lines.push(format!("  - `{}({})`", name, signature));
        }
    }

    let rendered = lines.join("\n");
    if rendered.len() <= MAX_HELPER_CONTEXT_CHARS {
        return rendered;
    }

    let mut cut = MAX_HELPER_CONTEXT_CHARS;
    while cut > 0 && !rendered.is_char_boundary(cut) {
        cut -= 1;
    }

    // Prefer a line boundary so the model never sees a half signature. Only
    // trim to it when it does not throw away most of the budget.
    let cut = rendered[..cut]
        .rfind('\n')
        .map(|newline| newline + 1)
        .filter(|line_cut| line_cut.saturating_mul(2) >= cut)
        .unwrap_or(cut);

    format!(
        "{}\n… (helper context truncated: {} of {} chars shown; use grep or read_file for the rest)",
        &rendered[..cut],
        cut,
        rendered.len()
    )
}

pub(super) fn parse_agent_action(content: &str) -> Result<AgentAction> {
    let parsed = parse_structured_content(content)?;

    let has_final_shape = parsed.get("findings").is_some() || parsed.get("status").is_some();
    let action_value = parsed
        .get("action")
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase());

    if action_value.is_none() && has_final_shape {
        return Ok(AgentAction::Final(parsed));
    }

    let raw: RawReadRequest = serde_json::from_value(parsed.clone())
        .into_diagnostic()
        .context("Invalid agent action payload")?;

    match raw.action.unwrap_or_else(|| "final".to_string()).as_str() {
        "final" => Ok(AgentAction::Final(parsed)),
        "read_file" => Ok(AgentAction::ReadRequest(ReadRequest::ReadFile {
            path: raw.path.unwrap_or_else(|| ".".to_string()),
        })),
        "grep" => Ok(AgentAction::ReadRequest(ReadRequest::Grep {
            pattern: raw.pattern.unwrap_or_default(),
            path: raw.path.unwrap_or_else(|| ".".to_string()),
            context_lines: raw.context_lines.unwrap_or(2).min(20),
        })),
        "list_dir" => Ok(AgentAction::ReadRequest(ReadRequest::ListDir {
            path: raw.path.unwrap_or_else(|| ".".to_string()),
        })),
        "find_files" => Ok(AgentAction::ReadRequest(ReadRequest::FindFiles {
            path: raw.path.unwrap_or_else(|| ".".to_string()),
            glob: raw.glob,
        })),
        other => Err(miette::miette!("Unsupported agent action '{}'", other)),
    }
}

pub(super) fn execute_read_request(
    request: &ReadRequest,
    project_root: &Path,
    permission_prompt: &PermissionPromptSpec,
) -> Result<String> {
    match request {
        ReadRequest::ReadFile { path } => {
            ensure_allowed(permission_prompt, "cat")?;
            let scoped_path = resolve_scoped_path(project_root, path)?;
            enforce_read_scope(request, &scoped_path, project_root, permission_prompt)?;
            confirm_request_if_interactive(request, &scoped_path, project_root, permission_prompt)?;
            // Line numbers are part of the evidence contract: findings carry a
            // `line`, and the initial prompt tells the model to reuse the ones it
            // was shown. Without `-n` it has to count lines by hand and guesses.
            let args = vec!["-n".to_string(), scoped_path.to_string_lossy().to_string()];
            run_command_capture("cat", &args, project_root)
        }
        ReadRequest::Grep {
            pattern,
            path,
            context_lines,
        } => {
            ensure_allowed(permission_prompt, "grep")?;
            let scoped_path = resolve_scoped_path(project_root, path)?;
            enforce_read_scope(request, &scoped_path, project_root, permission_prompt)?;
            confirm_request_if_interactive(request, &scoped_path, project_root, permission_prompt)?;
            // `-E` because the model writes extended regexes (`a|b|c`), which plain
            // grep reads literally and silently returns no matches for.
            let mut base_args = vec![
                "-E".to_string(),
                "-n".to_string(),
                "-C".to_string(),
                context_lines.to_string(),
            ];

            if scoped_path.is_dir() {
                if gnu_grep_available() {
                    // GNU grep prunes excluded dirs natively.
                    base_args.push("-r".to_string());
                    for excluded in EXCLUDED_DIR_NAMES {
                        base_args.push(format!("--exclude-dir={}", excluded));
                    }
                } else {
                    // BSD grep (macOS) has no `--exclude-dir`; pre-walk in Rust
                    // and grep the allowed files explicitly so excluded dirs
                    // never leak through a recursive search.
                    let allowed = collect_allowed_files(project_root, &scoped_path)?;
                    if allowed.is_empty() {
                        return Ok("(no in-scope files to search)".to_string());
                    }
                    let mut args = base_args;
                    args.push("--".to_string());
                    args.push(pattern.clone());
                    for file in allowed {
                        args.push(file.to_string_lossy().to_string());
                    }
                    return run_command_capture("grep", &args, project_root);
                }
            }

            let mut args = base_args;
            args.push("--".to_string());
            args.push(pattern.clone());
            args.push(scoped_path.to_string_lossy().to_string());
            run_command_capture("grep", &args, project_root)
        }
        ReadRequest::ListDir { path } => {
            ensure_allowed(permission_prompt, "ls")?;
            let scoped_path = resolve_scoped_path(project_root, path)?;
            enforce_read_scope(request, &scoped_path, project_root, permission_prompt)?;
            confirm_request_if_interactive(request, &scoped_path, project_root, permission_prompt)?;
            let args = vec!["-la".to_string(), scoped_path.to_string_lossy().to_string()];
            let raw = run_command_capture("ls", &args, project_root)?;
            // Drop entries that are excluded directories (or nested projects),
            // so a root-level listing never reveals them.
            let mut kept = Vec::new();
            for line in raw.lines() {
                if let Some(name) = line.split_whitespace().next_back() {
                    let name = name.trim();
                    let ignored = EXCLUDED_DIR_NAMES.contains(&name)
                        || is_ignored_path(project_root, &scoped_path.join(name));
                    if ignored {
                        continue;
                    }
                }
                kept.push(line);
            }
            Ok(kept.join("\n"))
        }
        ReadRequest::FindFiles { path, glob } => {
            ensure_allowed(permission_prompt, "find")?;
            let scoped_path = resolve_scoped_path(project_root, path)?;
            enforce_read_scope(request, &scoped_path, project_root, permission_prompt)?;
            confirm_request_if_interactive(request, &scoped_path, project_root, permission_prompt)?;
            let allowed = collect_allowed_files(project_root, &scoped_path)?;
            let filtered: Vec<String> = allowed
                .iter()
                .filter(|file| match glob {
                    Some(glob) => glob_match(glob, &file_name_of(file)),
                    None => true,
                })
                .map(|file| file.to_string_lossy().to_string())
                .collect();
            Ok(if filtered.is_empty() {
                "(no matching files found)".to_string()
            } else {
                filtered.join("\n")
            })
        }
    }
}

fn enforce_read_scope(
    request: &ReadRequest,
    scoped_path: &Path,
    project_root: &Path,
    permission_prompt: &PermissionPromptSpec,
) -> Result<()> {
    // Excluded directories are outside the audit scope under every read scope,
    // so this gate applies before the strict/workspace branch.
    if is_ignored_path(project_root, scoped_path) {
        return Err(miette::miette!(
            "Request denied: '{}' is inside an excluded directory ({}); \
             these directories are outside the audit scope",
            display_relative_path(project_root, scoped_path),
            EXCLUDED_DIR_NAMES.join(", ")
        ));
    }

    if !permission_prompt.read_scope.eq_ignore_ascii_case("strict") {
        return Ok(());
    }

    if matches!(
        request,
        ReadRequest::ListDir { .. } | ReadRequest::FindFiles { .. }
    ) {
        return Err(miette::miette!(
            "Request denied by strict read scope: directory listing and file discovery are not allowed"
        ));
    }

    if !scoped_path.is_file() {
        return Err(miette::miette!(
            "Request denied by strict read scope: only known source files can be accessed"
        ));
    }

    let allowed_paths = resolve_allowed_paths(project_root, permission_prompt)?;

    if allowed_paths.iter().any(|allowed| allowed == scoped_path) {
        return Ok(());
    }

    Err(miette::miette!(
        "Request denied by strict read scope: '{}' is not an allowed source file",
        display_relative_path(project_root, scoped_path)
    ))
}

fn resolve_allowed_paths(
    project_root: &Path,
    permission_prompt: &PermissionPromptSpec,
) -> Result<Vec<PathBuf>> {
    permission_prompt
        .allowed_paths
        .iter()
        .map(|path| resolve_scoped_path(project_root, path))
        .collect::<Result<Vec<PathBuf>>>()
}

fn confirm_request_if_interactive(
    request: &ReadRequest,
    scoped_path: &Path,
    project_root: &Path,
    permission_prompt: &PermissionPromptSpec,
) -> Result<()> {
    if !permission_prompt.interactive_permissions {
        return Ok(());
    }

    eprintln!(
        "[audit][permission] {} -> {}",
        summarize_read_request(request),
        display_relative_path(project_root, scoped_path)
    );
    eprint!("Allow this request? [y/N]: ");
    io::stderr().flush().into_diagnostic()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer).into_diagnostic()?;
    let accepted = matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes");

    if accepted {
        return Ok(());
    }

    Err(miette::miette!(
        "Request denied by user confirmation: {}",
        summarize_read_request(request)
    ))
}

fn display_relative_path(project_root: &Path, scoped_path: &Path) -> String {
    scoped_path
        .strip_prefix(project_root)
        .map(|relative| relative.display().to_string())
        .unwrap_or_else(|_| scoped_path.display().to_string())
}

fn ensure_allowed(permission_prompt: &PermissionPromptSpec, command: &str) -> Result<()> {
    if permission_prompt
        .allowed_commands
        .iter()
        .any(|allowed| allowed.eq_ignore_ascii_case(command))
    {
        return Ok(());
    }

    Err(miette::miette!(
        "Command '{}' is not permitted by permission prompt",
        command
    ))
}

fn resolve_scoped_path(project_root: &Path, requested_path: &str) -> Result<PathBuf> {
    let requested_path = requested_path.trim();
    let requested_path = if requested_path.is_empty() {
        "."
    } else {
        requested_path
    };

    let joined = if Path::new(requested_path).is_absolute() {
        PathBuf::from(requested_path)
    } else {
        project_root.join(requested_path)
    };

    let canonical = joined
        .canonicalize()
        .into_diagnostic()
        .with_context(|| format!("Path does not exist or is inaccessible: {}", requested_path))?;

    if !canonical.starts_with(project_root) {
        return Err(miette::miette!(
            "Path escapes project root and is not allowed: {}",
            requested_path
        ));
    }

    Ok(canonical)
}

/// Whether the `grep` on PATH is GNU (which supports `--exclude-dir`). Cached,
/// since the subprocess probe only needs to run once per process.
fn gnu_grep_available() -> bool {
    static GNU_GREP: OnceLock<bool> = OnceLock::new();
    *GNU_GREP.get_or_init(|| {
        Command::new("grep")
            .arg("--version")
            .output()
            .ok()
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .map(|text| text.contains("GNU grep"))
            .unwrap_or(false)
    })
}

/// Recursively collects files under `target` (or `target` itself when it is a
/// file), pruning excluded directories and nested projects so they never reach
/// a recursive grep or file listing.
fn collect_allowed_files(project_root: &Path, target: &Path) -> Result<Vec<PathBuf>> {
    if target.is_file() {
        return Ok(vec![target.to_path_buf()]);
    }

    let mut files = Vec::new();
    let mut to_visit = vec![target.to_path_buf()];

    while let Some(dir) = to_visit.pop() {
        let entries = std::fs::read_dir(&dir)
            .into_diagnostic()
            .with_context(|| format!("Failed to read directory {}", dir.display()))?;

        for entry in entries {
            let entry = entry.into_diagnostic()?;
            let path = entry.path();
            if path.is_dir() {
                if !is_ignored_path(project_root, &path) {
                    to_visit.push(path);
                }
            } else {
                files.push(path);
            }
        }
    }

    files.sort();
    Ok(files)
}

fn file_name_of(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// Minimal shell-style glob supporting `*` and `?`, matching against the whole
/// name. Enough for the model's `find_files` globs (`*.ak`, `*test*.compact`).
pub(crate) fn glob_match(pattern: &str, name: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let name: Vec<char> = name.chars().collect();
    let (mut p, mut n) = (0usize, 0usize);
    let mut star: Option<(usize, usize)> = None;

    while n < name.len() {
        match pattern.get(p) {
            Some('*') => {
                star = Some((p, n));
                p += 1;
            }
            Some('?') => {
                p += 1;
                n += 1;
            }
            Some(c) if *c == name[n] => {
                p += 1;
                n += 1;
            }
            _ => match star {
                Some((sp, sn)) => {
                    p = sp + 1;
                    n = sn + 1;
                    star = Some((sp, n));
                }
                None => return false,
            },
        }
    }

    while pattern.get(p) == Some(&'*') {
        p += 1;
    }
    p == pattern.len()
}

fn run_command_capture(command: &str, args: &[String], cwd: &Path) -> Result<String> {
    let output = Command::new(command)
        .args(args)
        .current_dir(cwd)
        .output()
        .into_diagnostic()
        .with_context(|| format!("Failed to run command '{}'", command))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let mut combined = String::new();

    if !stdout.trim().is_empty() {
        combined.push_str(&stdout);
    }

    if !stderr.trim().is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str(&stderr);
    }

    if combined.trim().is_empty() {
        combined = format!(
            "(no output; command exited with status {})",
            output.status.code().unwrap_or_default()
        );
    }

    if !output.status.success() {
        combined.push_str(&format!(
            "\n(command exited with status {})",
            output.status.code().unwrap_or_default()
        ));
    }

    Ok(combined)
}

/// Returns the first balanced JSON object embedded in `content`, ignoring braces
/// inside string literals. Models without native tool-calling routinely wrap the
/// object in stray characters or prose, so we recover the payload instead of
/// failing the whole run.
fn find_embedded_json_object(content: &str) -> Option<&str> {
    let bytes = content.as_bytes();
    let start = content.find('{')?;

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, &byte) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }

        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&content[start..=offset]);
                }
            }
            _ => {}
        }
    }

    None
}

fn parse_structured_content(content: &str) -> Result<Value> {
    if let Ok(parsed) = serde_json::from_str::<Value>(content) {
        return Ok(parsed);
    }

    let trimmed = content.trim();
    let fenced = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .map(str::trim);

    if let Some(fenced_content) = fenced {
        let fenced_content = fenced_content.strip_suffix("```").unwrap_or(fenced_content);
        if let Ok(parsed) = serde_json::from_str::<Value>(fenced_content.trim()) {
            return Ok(parsed);
        }
    }

    if let Some(embedded) = find_embedded_json_object(trimmed)
        && let Ok(parsed) = serde_json::from_str::<Value>(embedded)
    {
        return Ok(parsed);
    }

    Err(miette::miette!(
        "AI provider response is not valid JSON for structured findings. Raw response was: {}",
        preview_for_error(trimmed)
    ))
}

/// Keeps the failing payload in the error without flooding the terminal with a
/// multi-kilobyte response.
fn preview_for_error(content: &str) -> String {
    const MAX_PREVIEW_CHARS: usize = 600;

    if content.is_empty() {
        return "(empty response)".to_string();
    }

    let truncated: String = content.chars().take(MAX_PREVIEW_CHARS).collect();
    if truncated.chars().count() < content.chars().count() {
        format!("{}… (truncated)", truncated)
    } else {
        truncated
    }
}

pub(super) fn block_on_runtime_aware<F, T>(future: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    match Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(future)),
        Err(_) => {
            let runtime = tokio::runtime::Runtime::new().into_diagnostic()?;
            runtime.block_on(future)
        }
    }
}

pub(super) fn summarize_read_request(request: &ReadRequest) -> String {
    match request {
        ReadRequest::ReadFile { path } => format!("read_file {}", path),
        ReadRequest::Grep {
            pattern,
            path,
            context_lines,
        } => format!(
            "grep pattern='{}' path={} context_lines={}",
            pattern, path, context_lines
        ),
        ReadRequest::ListDir { path } => format!("list_dir {}", path),
        ReadRequest::FindFiles { path, glob } => {
            format!(
                "find_files path={} glob={}",
                path,
                glob.as_deref().unwrap_or("*")
            )
        }
    }
}

pub(super) fn describe_read_request_friendly(request: &ReadRequest) -> String {
    match request {
        ReadRequest::ReadFile { path } => {
            format!("read file '{}'", path)
        }
        ReadRequest::Grep {
            pattern,
            path,
            context_lines,
        } => format!(
            "search '{}' in '{}' ({} context lines)",
            pattern, path, context_lines
        ),
        ReadRequest::ListDir { path } => {
            format!("list directory '{}'", path)
        }
        ReadRequest::FindFiles { path, glob } => format!(
            "find files in '{}' with glob '{}'",
            path,
            glob.as_deref().unwrap_or("*")
        ),
    }
}

pub(super) fn render_tool_output_for_log(request: &ReadRequest, output: &str) -> String {
    match request {
        ReadRequest::ReadFile { path } => {
            format!(
                "📄 File '{}' read (content hidden in logs, {} chars)",
                path,
                output.chars().count()
            )
        }
        _ => output.to_string(),
    }
}

pub(super) fn log_agent_progress(level: crate::logging::LogLevel, message: impl AsRef<str>) {
    if level.is_enabled() {
        crate::logging::write_line("🤖 ", message.as_ref());
    }
}

/// Progress a short audit can show for every step, without the raw payload.
pub(super) fn log_step(level: crate::logging::LogLevel, message: impl AsRef<str>) {
    if level.is_enabled() {
        crate::logging::write_line("  ", message.as_ref());
    }
}

/// Per-skill cache of tool results the model has already been shown.
///
/// The agent loop re-reads the same file many times: without this, every repeat
/// resends the whole (possibly capped) payload and burns context for nothing.
#[derive(Default)]
pub(super) struct ToolResultCache {
    entries: std::collections::HashMap<String, CachedToolResult>,
}

struct CachedToolResult {
    digest: String,
    step: usize,
}

impl ToolResultCache {
    fn key(request: &ReadRequest) -> Option<String> {
        match request {
            ReadRequest::ReadFile { path } => Some(format!("read\0{path}")),
            ReadRequest::Grep {
                pattern,
                path,
                context_lines,
            } => Some(format!("grep\0{path}\0{context_lines}\0{pattern}")),
            // Listings and file discovery are cheap and change more readily than
            // a source file; caching them would return stale views of the tree.
            ReadRequest::ListDir { .. } | ReadRequest::FindFiles { .. } => None,
        }
    }
}

/// Runs a request, replacing an unchanged repeat with a short note.
pub(super) fn execute_read_request_cached(
    cache: &mut ToolResultCache,
    request: &ReadRequest,
    project_root: &Path,
    permission_prompt: &PermissionPromptSpec,
    step: usize,
) -> Result<String> {
    let output = execute_read_request(request, project_root, permission_prompt)?;

    let Some(key) = ToolResultCache::key(request) else {
        return Ok(output);
    };

    let digest = format!("sha256:{}", crate::ast::sha256_hex(output.as_bytes()));
    let note = cache.entries.get(&key).and_then(|cached| {
        (cached.digest == digest).then(|| {
            format!(
                "Unchanged since step {} ({digest}). Content identical to what you \
already have; not resent to save context. Do not re-read unless the file may \
have been modified externally.",
                cached.step
            )
        })
    });

    if let Some(note) = note {
        return Ok(note);
    }

    cache.entries.insert(key, CachedToolResult { digest, step });
    Ok(output)
}

/// Marks the first line of every tool-result message. Pruning recognises
/// results by this prefix instead of remembering positions, so inserting or
/// removing a message can never make it elide the wrong one.
const TOOL_RESULT_MARKER: &str = "Tool result for ";

/// Replaces the oldest tool results with a stub once too many have accumulated.
///
/// Per-result truncation bounds one read; this bounds the whole skill. Without
/// it a 30-step run carries all 30 results forever, and the ceiling is the step
/// count times the cap regardless of how well behaved any single read was. The
/// model has already reasoned over these -- its conclusions live in the
/// assistant turns, which are kept -- so what is dropped is re-readable detail,
/// and the stub names the request so it can ask again.
///
/// Results are found by their marker rather than by stored index: the message
/// list is append-only today, but an index would silently point at the wrong
/// message if that ever changed.
fn prune_stale_tool_results(messages: &mut [Value], ai_logs: crate::logging::LogLevel) {
    let tool_results = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| is_tool_result_message(message))
        .map(|(index, _)| index)
        .collect::<Vec<usize>>();

    if tool_results.len() < TOOL_RESULT_PRUNE_TRIGGER {
        return;
    }

    let elide_count = tool_results.len() - MAX_RETAINED_TOOL_RESULTS;

    for index in tool_results.into_iter().take(elide_count) {
        let Some(message) = messages.get_mut(index) else {
            continue;
        };

        let summary = message["content"]
            .as_str()
            .and_then(strip_tool_result_marker)
            .unwrap_or("an earlier request")
            .to_string();

        message["content"] = Value::String(format!(
            "{TOOL_RESULT_MARKER}{summary} was elided to keep this conversation \
inside the model's context window. Request it again if you still need it."
        ));
    }

    log_full(
        ai_logs,
        format!(
            "elided {elide_count} older tool result(s), keeping the most recent \
{MAX_RETAINED_TOOL_RESULTS}"
        ),
    );
}

fn is_tool_result_message(message: &Value) -> bool {
    message["content"]
        .as_str()
        .is_some_and(|content| content.starts_with(TOOL_RESULT_MARKER))
}

/// Returns `summary` from `Tool result for {summary}:`, so a pruned message can
/// name what it used to hold.
fn strip_tool_result_marker(content: &str) -> Option<&str> {
    content
        .strip_prefix(TOOL_RESULT_MARKER)?
        .split_once(':')
        .map(|(summary, _)| summary.split('\n').next().unwrap_or(summary).trim())
}

/// Diagnostic detail: request lines, analysis summaries, elided results and raw
/// payloads. Only the `full` level prints these.
pub(super) fn log_full(level: crate::logging::LogLevel, message: impl AsRef<str>) {
    if level.is_full() {
        crate::logging::write_line("🤖 ", message.as_ref());
    }
}

fn format_usage_suffix(usage: &TokenUsage) -> String {
    if usage.is_empty() {
        String::new()
    } else {
        format!("  ({})", format_token_usage_inline(usage))
    }
}

/// Keeps one action within a fixed column so brief logs stay scannable.
fn truncate_action(action: &str) -> String {
    const WIDTH: usize = 32;

    if action.chars().count() <= WIDTH {
        return action.to_string();
    }

    let mut shortened = action.chars().take(WIDTH - 1).collect::<String>();
    shortened.push('…');
    shortened
}

pub(super) fn iteration_from_parsed(
    skill: &VulnerabilitySkill,
    parsed: Value,
    token_usage: TokenUsage,
) -> SkillIterationResult {
    let findings = parsed
        .get("findings")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|item| {
                    let file = item
                        .get("file")
                        .and_then(Value::as_str)
                        .filter(|value| !value.trim().is_empty())
                        .map(ToString::to_string)
                        .or_else(|| {
                            item.get("location")
                                .and_then(|value| value.get("file"))
                                .and_then(Value::as_str)
                                .filter(|value| !value.trim().is_empty())
                                .map(ToString::to_string)
                        });

                    let line = parse_line_number(item.get("line")).or_else(|| {
                        parse_line_number(item.get("location").and_then(|value| value.get("line")))
                    });

                    VulnerabilityFinding {
                        title: item
                            .get("title")
                            .and_then(Value::as_str)
                            .unwrap_or("Untitled finding")
                            .to_string(),
                        severity: item
                            .get("severity")
                            .and_then(Value::as_str)
                            .unwrap_or(&skill.severity)
                            .to_string(),
                        summary: item
                            .get("summary")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        evidence: item
                            .get("evidence")
                            .and_then(Value::as_array)
                            .map(|e| {
                                e.iter()
                                    .filter_map(Value::as_str)
                                    .map(ToString::to_string)
                                    .collect::<Vec<String>>()
                            })
                            .unwrap_or_default(),
                        recommendation: item
                            .get("recommendation")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        file,
                        line,
                    }
                })
                .collect::<Vec<VulnerabilityFinding>>()
        })
        .unwrap_or_default();

    let status = parsed
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("completed")
        .to_string();

    let next_prompt = parsed
        .get("next_prompt")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(|text| MiniPrompt {
            skill_id: skill.id.clone(),
            text: text.to_string(),
        });

    let analysis_summary = parsed
        .get("analysis_summary")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(ToString::to_string);

    let error = parsed
        .get("error")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(ToString::to_string)
        .or_else(|| {
            (status == "failed").then(|| "provider reported failure without a reason".to_string())
        });

    SkillIterationResult {
        skill_id: skill.id.clone(),
        status,
        findings,
        next_prompt,
        skill_fingerprint: None,
        provider: None,
        duration_ms: None,
        analysis_summary,
        error,
        token_usage,
    }
}

pub(super) fn format_token_usage_inline(usage: &TokenUsage) -> String {
    if usage.is_empty() {
        return "no token usage reported".to_string();
    }

    let mut parts = vec![
        format!("in: {}", usage.input_tokens),
        format!("out: {}", usage.output_tokens),
    ];

    if let Some(cached) = usage.cache_read_input_tokens
        && cached > 0
    {
        parts.push(format!("cached: {}", cached));
    }
    if let Some(reasoning) = usage.reasoning_tokens
        && reasoning > 0
    {
        parts.push(format!("reasoning: {}", reasoning));
    }

    parts.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        MiniPrompt, PermissionPromptSpec, ValidatorContextEntry, ValidatorContextMap,
        ValidatorHandlerContext, ValidatorParameterContext,
    };

    #[test]
    fn execute_read_request_numbers_lines() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        let file = root.join("validators/spend.ak");

        std::fs::create_dir_all(file.parent().expect("parent")).expect("create dir");
        std::fs::write(&file, "validator spend {}\nbad\n").expect("write file");

        let prompt = PermissionPromptSpec {
            shell: "bash".to_string(),
            allowed_commands: vec!["cat".to_string()],
            scope_rules: vec![],
            workspace_root: root.display().to_string(),
            read_scope: "workspace".to_string(),
            interactive_permissions: false,
            allowed_paths: vec![],
        };

        let output = execute_read_request(
            &ReadRequest::ReadFile {
                path: "validators/spend.ak".to_string(),
            },
            &root.canonicalize().expect("canonical root"),
            &prompt,
        )
        .expect("request should be allowed");

        let first_line = output.lines().next().expect("one line");
        assert!(
            first_line.trim_start().starts_with("1\tvalidator spend"),
            "read output must carry line numbers, got: {}",
            first_line
        );
        assert!(output.contains("2\tbad"));
    }

    #[test]
    fn repeated_read_of_an_unchanged_file_returns_a_note_instead_of_content() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        let file = root.join("validators/spend.ak");

        std::fs::create_dir_all(file.parent().expect("parent")).expect("create dir");
        std::fs::write(&file, "validator spend {}\nbad\n").expect("write file");

        let prompt = workspace_prompt(root, vec!["cat"]);
        let canonical_root = root.canonicalize().expect("canonical root");
        let request = ReadRequest::ReadFile {
            path: "validators/spend.ak".to_string(),
        };
        let mut cache = ToolResultCache::default();

        let first = execute_read_request_cached(&mut cache, &request, &canonical_root, &prompt, 3)
            .expect("first read should be allowed");
        assert!(
            first.contains("1\tvalidator spend"),
            "the first read must carry the content: {first}"
        );

        let second = execute_read_request_cached(&mut cache, &request, &canonical_root, &prompt, 5)
            .expect("repeat read should be allowed");

        assert!(
            !second.contains("bad"),
            "a repeat read must not resend the file: {second}"
        );
        assert!(
            second.contains("Unchanged since step 3"),
            "the note should name the step that produced it: {second}"
        );
        assert!(
            second.contains("sha256:"),
            "the note should carry a digest: {second}"
        );
    }

    #[test]
    fn the_cache_distinguishes_unchanged_from_changed_content() {
        // Same length, different bytes: a length-based or path-based cache would
        // wrongly report "unchanged" here, so this pins the digest comparison.
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        let file = root.join("validators/spend.ak");

        std::fs::create_dir_all(file.parent().expect("parent")).expect("create dir");
        std::fs::write(&file, "validator spend {}\n").expect("write file");

        let prompt = workspace_prompt(root, vec!["cat"]);
        let canonical_root = root.canonicalize().expect("canonical root");
        let request = ReadRequest::ReadFile {
            path: "validators/spend.ak".to_string(),
        };
        let mut cache = ToolResultCache::default();

        let first = execute_read_request_cached(&mut cache, &request, &canonical_root, &prompt, 1)
            .expect("first read");
        assert!(first.contains("validator spend"));

        std::fs::write(&file, "validator spend {}\n").expect("rewrite identical file");
        let same = execute_read_request_cached(&mut cache, &request, &canonical_root, &prompt, 2)
            .expect("second read");
        assert!(
            same.contains("Unchanged since step 1"),
            "identical bytes must hit the cache: {same}"
        );

        // Same byte length as the original, different content.
        std::fs::write(&file, "validator spend X\n").expect("rewrite changed file");
        let changed =
            execute_read_request_cached(&mut cache, &request, &canonical_root, &prompt, 3)
                .expect("third read");
        assert!(
            changed.contains("validator spend X"),
            "different bytes of the same length must be resent: {changed}"
        );
        assert!(!changed.contains("Unchanged since"));
    }

    #[test]
    fn grep_cache_keys_on_pattern_path_and_context_lines() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        let file = root.join("validators/spend.ak");

        std::fs::create_dir_all(file.parent().expect("parent")).expect("create dir");
        std::fs::write(&file, "validator spend {}\nspend again\n").expect("write file");

        let prompt = workspace_prompt(root, vec!["grep"]);
        let canonical_root = root.canonicalize().expect("canonical root");
        let mut cache = ToolResultCache::default();

        let grep_request = |pattern: &str, context_lines: usize| ReadRequest::Grep {
            pattern: pattern.to_string(),
            path: "validators/spend.ak".to_string(),
            context_lines,
        };

        let first = execute_read_request_cached(
            &mut cache,
            &grep_request("spend", 2),
            &canonical_root,
            &prompt,
            1,
        )
        .expect("read");
        assert!(first.contains("spend"), "first grep returns matches");

        let same = execute_read_request_cached(
            &mut cache,
            &grep_request("spend", 2),
            &canonical_root,
            &prompt,
            2,
        )
        .expect("repeat");
        assert!(
            same.contains("Unchanged since step 1"),
            "an identical grep must hit the cache: {same}"
        );

        let other_context = execute_read_request_cached(
            &mut cache,
            &grep_request("spend", 0),
            &canonical_root,
            &prompt,
            3,
        )
        .expect("narrower grep");
        assert!(
            !other_context.contains("Unchanged since"),
            "a different context_lines is a different request: {other_context}"
        );

        let other_pattern = execute_read_request_cached(
            &mut cache,
            &grep_request("validator", 2),
            &canonical_root,
            &prompt,
            4,
        )
        .expect("different pattern");
        assert!(
            !other_pattern.contains("Unchanged since"),
            "a different pattern is a different request: {other_pattern}"
        );
    }

    #[test]
    fn listings_and_find_results_are_never_cached() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        let file = root.join("validators/spend.ak");

        std::fs::create_dir_all(file.parent().expect("parent")).expect("create dir");
        std::fs::write(&file, "validator spend {}\n").expect("write file");

        let prompt = workspace_prompt(root, vec!["ls", "find"]);
        let canonical_root = root.canonicalize().expect("canonical root");
        let mut cache = ToolResultCache::default();

        for step in 1..=2 {
            let listing = execute_read_request_cached(
                &mut cache,
                &ReadRequest::ListDir {
                    path: "validators".to_string(),
                },
                &canonical_root,
                &prompt,
                step,
            )
            .expect("list_dir should be allowed");
            assert!(
                !listing.contains("Unchanged since"),
                "directory listings must always be fresh (step {step}): {listing}"
            );

            let found = execute_read_request_cached(
                &mut cache,
                &ReadRequest::FindFiles {
                    path: ".".to_string(),
                    glob: Some("*.ak".to_string()),
                },
                &canonical_root,
                &prompt,
                step,
            )
            .expect("find_files should be allowed");
            assert!(
                !found.contains("Unchanged since"),
                "file discovery must always be fresh (step {step}): {found}"
            );
        }

        assert!(
            cache.entries.is_empty(),
            "neither listings nor discovery should populate the cache"
        );
    }

    #[test]
    fn execute_read_request_strict_allows_known_file() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        let file = root.join("validators/spend.ak");

        std::fs::create_dir_all(file.parent().expect("parent")).expect("create dir");
        std::fs::write(&file, "validator spend {}\n").expect("write file");

        let prompt = PermissionPromptSpec {
            shell: "bash".to_string(),
            allowed_commands: vec!["cat".to_string()],
            scope_rules: vec![],
            workspace_root: root.display().to_string(),
            read_scope: "strict".to_string(),
            interactive_permissions: false,
            allowed_paths: vec!["validators/spend.ak".to_string()],
        };

        let output = execute_read_request(
            &ReadRequest::ReadFile {
                path: "validators/spend.ak".to_string(),
            },
            &root.canonicalize().expect("canonical root"),
            &prompt,
        )
        .expect("request should be allowed");

        assert!(output.contains("validator spend"));
    }

    #[test]
    fn execute_read_request_strict_rejects_list_dir() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();

        let prompt = PermissionPromptSpec {
            shell: "bash".to_string(),
            allowed_commands: vec!["ls".to_string()],
            scope_rules: vec![],
            workspace_root: root.display().to_string(),
            read_scope: "strict".to_string(),
            interactive_permissions: false,
            allowed_paths: vec!["validators/spend.ak".to_string()],
        };

        let err = execute_read_request(
            &ReadRequest::ListDir {
                path: ".".to_string(),
            },
            &root.canonicalize().expect("canonical root"),
            &prompt,
        )
        .expect_err("strict scope should reject list_dir");

        assert!(err.to_string().contains("strict read scope"));
    }

    #[test]
    fn initial_prompt_renders_validator_context_map() {
        let permission_prompt = PermissionPromptSpec {
            shell: "bash".to_string(),
            allowed_commands: vec!["cat".to_string()],
            scope_rules: vec!["rule".to_string()],
            workspace_root: ".".to_string(),
            read_scope: "workspace".to_string(),
            interactive_permissions: false,
            allowed_paths: vec![],
        };

        let validator_context = ValidatorContextMap {
            validators: vec![ValidatorContextEntry {
                id: "validators.vesting.hello_world".to_string(),
                module: "validators/vesting.ak".to_string(),
                source_file: "validators/vesting.ak".to_string(),
                source_span: None,
                handlers: vec![ValidatorHandlerContext {
                    name: "spend".to_string(),
                    parameters: vec![ValidatorParameterContext {
                        name: "datum".to_string(),
                        r#type: "Option<Datum>".to_string(),
                    }],
                }],
            }],
            functions: vec![],
        };

        let prompt = build_initial_user_prompt(
            &MiniPrompt {
                skill_id: "s1".to_string(),
                text: "skill".to_string(),
            },
            &["validators/vesting.ak".to_string()],
            &validator_context,
            &permission_prompt,
        );

        assert!(prompt.contains("Validator context map:"));
        assert!(prompt.contains("validators.vesting.hello_world"));
        assert!(prompt.contains("spend(datum: Option<Datum>)"));
    }

    #[test]
    fn initial_prompt_renders_top_level_functions() {
        let permission_prompt = PermissionPromptSpec {
            shell: "bash".to_string(),
            allowed_commands: vec!["cat".to_string()],
            scope_rules: vec![],
            workspace_root: ".".to_string(),
            read_scope: "workspace".to_string(),
            interactive_permissions: false,
            allowed_paths: vec![],
        };

        let validator_context = ValidatorContextMap {
            validators: vec![],
            functions: vec![FunctionContextEntry {
                id: "validators.order.mint_unique".to_string(),
                module: "validators.order".to_string(),
                source_file: "validators/order.ak".to_string(),
                source_span: None,
                parameters: vec![ValidatorParameterContext {
                    name: "labels".to_string(),
                    r#type: "List<AssetName>".to_string(),
                }],
            }],
        };

        let prompt = build_initial_user_prompt(
            &MiniPrompt {
                skill_id: "s1".to_string(),
                text: "skill".to_string(),
            },
            &["validators/order.ak".to_string()],
            &validator_context,
            &permission_prompt,
        );

        assert!(prompt.contains("Top-level functions (helpers):"));
        assert!(prompt.contains("validators/order.ak"));
        assert!(prompt.contains("mint_unique(labels: List<AssetName>)"));
    }

    #[test]
    fn helper_context_is_capped() {
        let functions = (0..200)
            .map(|index| FunctionContextEntry {
                id: format!("validators.big.helper_{}", index),
                module: "validators.big".to_string(),
                source_file: "validators/big.ak".to_string(),
                source_span: None,
                parameters: vec![ValidatorParameterContext {
                    name: "value".to_string(),
                    r#type: "ByteArray".to_string(),
                }],
            })
            .collect::<Vec<_>>();

        let rendered = render_helper_context(&functions);
        let body = rendered
            .split("\n… (helper context truncated")
            .next()
            .expect("truncated body");

        assert!(rendered.contains("helper context truncated"));
        assert!(rendered.contains("use grep or read_file"));
        assert!(rendered.contains("- `validators/big.ak`"));
        assert!(
            body.len() <= MAX_HELPER_CONTEXT_CHARS,
            "helper body exceeded its cap: {}",
            body.len()
        );
        // The cut must land on a whole line, never mid-signature.
        assert!(body.ends_with('\n'));
        assert!(body.lines().last().unwrap().ends_with('`'));
    }

    #[test]
    fn initial_prompt_renders_empty_validator_context_map() {
        let permission_prompt = PermissionPromptSpec {
            shell: "bash".to_string(),
            allowed_commands: vec!["cat".to_string()],
            scope_rules: vec!["rule".to_string()],
            workspace_root: ".".to_string(),
            read_scope: "workspace".to_string(),
            interactive_permissions: false,
            allowed_paths: vec![],
        };

        let prompt = build_initial_user_prompt(
            &MiniPrompt {
                skill_id: "s1".to_string(),
                text: "skill".to_string(),
            },
            &[],
            &ValidatorContextMap::default(),
            &permission_prompt,
        );

        assert!(prompt.contains("Validator context map:"));
        assert!(prompt.contains("- (none)"));
    }

    fn workspace_prompt(root: &Path, commands: Vec<&str>) -> PermissionPromptSpec {
        PermissionPromptSpec {
            shell: "bash".to_string(),
            allowed_commands: commands.into_iter().map(ToString::to_string).collect(),
            scope_rules: vec![],
            workspace_root: root.display().to_string(),
            read_scope: "workspace".to_string(),
            interactive_permissions: false,
            allowed_paths: vec![],
        }
    }

    fn seed_project(root: &Path) {
        std::fs::create_dir_all(root.join("validators")).expect("create validators");
        std::fs::write(root.join("validators/spend.ak"), "validator spend {}\n")
            .expect("write source");
        for dir in ["build", "target", ".git", ".tx3"] {
            std::fs::create_dir_all(root.join(dir)).expect("create excluded dir");
        }
        std::fs::write(root.join("build/secret.ak"), "validator secret {}\n")
            .expect("write secret in build");
        std::fs::write(root.join(".git/config"), "[core]\n").expect("write git config");
    }

    #[test]
    fn is_ignored_path_detects_excluded_and_nested_project_paths() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        std::fs::create_dir_all(root.join("build")).expect("create build");
        std::fs::create_dir_all(root.join("lib/vendored")).expect("create vendored");
        std::fs::create_dir_all(root.join("validators")).expect("create validators");
        std::fs::write(root.join("lib/vendored/aiken.toml"), "").expect("write aiken.toml");
        std::fs::write(root.join("validators/x.ak"), "").expect("write normal file");
        std::fs::write(root.join("build/secret.ak"), "").expect("write secret");

        let canonical = |p: &Path| p.canonicalize().expect("canonicalize");

        assert!(is_ignored_path(
            root,
            &canonical(&root.join("build/secret.ak"))
        ));
        assert!(is_ignored_path(root, &canonical(&root.join("build"))));
        assert!(is_ignored_path(
            root,
            &canonical(&root.join("lib/vendored/aiken.toml"))
        ));
        assert!(is_ignored_path(
            root,
            &canonical(&root.join("lib/vendored"))
        ));
        assert!(!is_ignored_path(
            root,
            &canonical(&root.join("validators/x.ak"))
        ));
        assert!(!is_ignored_path(root, &canonical(&root.join("validators"))));
        assert!(!is_ignored_path(root, root));
    }

    #[test]
    fn read_file_inside_excluded_dir_is_denied_even_in_workspace_scope() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        seed_project(root);
        let prompt = workspace_prompt(root, vec!["cat"]);

        let err = execute_read_request(
            &ReadRequest::ReadFile {
                path: "build/secret.ak".to_string(),
            },
            &root.canonicalize().expect("canonical root"),
            &prompt,
        )
        .expect_err("read inside excluded dir must be denied");

        assert!(err.to_string().contains("excluded directory"));
    }

    #[test]
    fn read_file_in_normal_dir_is_allowed_in_workspace_scope() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        seed_project(root);
        let prompt = workspace_prompt(root, vec!["cat"]);

        let output = execute_read_request(
            &ReadRequest::ReadFile {
                path: "validators/spend.ak".to_string(),
            },
            &root.canonicalize().expect("canonical root"),
            &prompt,
        )
        .expect("normal read should be allowed");

        assert!(output.contains("validator spend"));
    }

    #[test]
    fn find_files_on_root_excludes_build_git_target_tx3() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        seed_project(root);
        std::fs::write(root.join("validators/extra.ak"), "").expect("write extra");
        let prompt = workspace_prompt(root, vec!["find"]);

        let output = execute_read_request(
            &ReadRequest::FindFiles {
                path: ".".to_string(),
                glob: Some("*.ak".to_string()),
            },
            &root.canonicalize().expect("canonical root"),
            &prompt,
        )
        .expect("find should succeed");

        assert!(output.contains("spend.ak"), "got: {output}");
        assert!(output.contains("extra.ak"), "got: {output}");
        assert!(
            !output.contains("secret.ak"),
            "excluded dir file must not leak: {output}"
        );
    }

    #[test]
    fn list_dir_on_root_omits_excluded_dirs() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        seed_project(root);
        let prompt = workspace_prompt(root, vec!["ls"]);

        let output = execute_read_request(
            &ReadRequest::ListDir {
                path: ".".to_string(),
            },
            &root.canonicalize().expect("canonical root"),
            &prompt,
        )
        .expect("list_dir should succeed");

        for excluded in [".git", "target", ".tx3", "build"] {
            assert!(
                !output.lines().any(|line| line.ends_with(excluded)),
                "{output}"
            );
        }
        assert!(
            output.lines().any(|line| line.ends_with("validators")),
            "{output}"
        );
    }

    #[test]
    fn grep_on_root_prunes_excluded_dirs() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        seed_project(root);
        std::fs::write(root.join("validators/extra.ak"), "validator extra {}\n").expect("write");
        let prompt = workspace_prompt(root, vec!["grep"]);

        let output = execute_read_request(
            &ReadRequest::Grep {
                pattern: "validator".to_string(),
                path: ".".to_string(),
                context_lines: 0,
            },
            &root.canonicalize().expect("canonical root"),
            &prompt,
        )
        .expect("grep should succeed");

        assert!(output.contains("spend.ak"), "got: {output}");
        assert!(
            !output.contains("secret.ak"),
            "grep must not descend into excluded dirs: {output}"
        );
    }

    #[test]
    fn parse_structured_content_recovers_object_behind_stray_prefix() {
        // Observed in the wild: Opus emitted `={"action":...}` and the stray `=`
        // aborted the whole audit run.
        let parsed = parse_structured_content(
            "={\"action\":\"read_file\",\"path\":\"lib/butane/subvalidators/gov_issue.ak\"}",
        )
        .expect("should recover embedded object");

        assert_eq!(parsed["action"], "read_file");
        assert_eq!(parsed["path"], "lib/butane/subvalidators/gov_issue.ak");
    }

    #[test]
    fn parse_agent_action_recovers_read_request_behind_stray_prefix() {
        let action = parse_agent_action("={\"action\":\"read_file\",\"path\":\"lib/a.ak\"}")
            .expect("should classify as a read request");

        match action {
            AgentAction::ReadRequest(ReadRequest::ReadFile { path }) => {
                assert_eq!(path, "lib/a.ak");
            }
            other => panic!("expected a read_file request, got {:?}", other),
        }
    }

    #[test]
    fn parse_structured_content_ignores_prose_around_the_object() {
        let parsed = parse_structured_content(
            "Here is my answer:\n{\"status\":\"completed\",\"findings\":[]}\nHope that helps.",
        )
        .expect("should recover embedded object");

        assert_eq!(parsed["status"], "completed");
        assert_eq!(parsed["findings"].as_array().map(Vec::len), Some(0));
    }

    #[test]
    fn parse_structured_content_does_not_stop_at_braces_inside_strings() {
        let parsed = parse_structured_content(
            "={\"action\":\"grep\",\"pattern\":\"fn foo() { bar }\",\"path\":\"lib\"}",
        )
        .expect("should recover embedded object");

        assert_eq!(parsed["pattern"], "fn foo() { bar }");
        assert_eq!(parsed["path"], "lib");
    }

    #[test]
    fn parse_structured_content_handles_escaped_quotes_inside_strings() {
        let parsed = parse_structured_content(
            "={\"action\":\"grep\",\"pattern\":\"say \\\"hi\\\" }\",\"path\":\"lib\"}",
        )
        .expect("should recover embedded object");

        assert_eq!(parsed["pattern"], "say \"hi\" }");
    }

    #[test]
    fn parse_structured_content_still_accepts_plain_and_fenced_json() {
        let plain = parse_structured_content("{\"action\":\"list_dir\",\"path\":\".\"}")
            .expect("plain json should parse");
        assert_eq!(plain["action"], "list_dir");

        let fenced =
            parse_structured_content("```json\n{\"action\":\"list_dir\",\"path\":\".\"}\n```")
                .expect("fenced json should parse");
        assert_eq!(fenced["action"], "list_dir");
    }

    #[test]
    fn parse_structured_content_error_includes_the_raw_response() {
        let error = parse_structured_content("I could not complete this analysis.")
            .expect_err("should fail without an object");

        assert!(
            error
                .to_string()
                .contains("I could not complete this analysis.")
        );
    }

    #[test]
    fn parse_structured_content_rejects_unbalanced_object() {
        let error = parse_structured_content("={\"action\":\"read_file\",\"path\":\"a.ak\"")
            .expect_err("truncated object should not parse");

        assert!(error.to_string().contains("not valid JSON"));
    }

    #[test]
    fn tool_result_prompt_uses_readable_request_summary() {
        let prompt = build_tool_result_user_prompt(
            &ReadRequest::ReadFile {
                path: "lib/a.ak".to_string(),
            },
            "file contents",
            DEFAULT_MAX_TOOL_OUTPUT_BYTES,
        );

        assert!(prompt.contains("read_file lib/a.ak"));
        // Rust `Debug` syntax must not leak into the model prompt.
        assert!(!prompt.contains("ReadFile {"));
    }

    fn dummy_skill() -> VulnerabilitySkill {
        VulnerabilitySkill {
            id: "test-skill".to_string(),
            name: "test".to_string(),
            severity: "low".to_string(),
            description: String::new(),
            prompt_fragment: String::new(),
            examples: Vec::new(),
            false_positives: Vec::new(),
            references: Vec::new(),
            tags: Vec::new(),
            confidence_hint: None,
            guidance_markdown: String::new(),
        }
    }

    fn loop_context<'a>(root: &'a Path, prompt: &'a PermissionPromptSpec) -> AgentLoopContext<'a> {
        AgentLoopContext {
            endpoint: "test://endpoint",
            ai_logs: crate::logging::LogLevel::Off,
            project_root: root,
            permission_prompt: prompt,
            provider_label: "Test provider",
            skill_timeout: DEFAULT_SKILL_TIMEOUT,
            max_agent_steps: DEFAULT_MAX_AGENT_STEPS,
            max_tool_output_bytes: DEFAULT_MAX_TOOL_OUTPUT_BYTES,
        }
    }

    #[test]
    fn tool_output_within_the_cap_is_untouched() {
        let output = "validator a { spend(_: Int) { True } }";
        assert_eq!(truncate_tool_output(output, 4096), output);
        assert_eq!(
            truncate_tool_output(output, 0),
            output,
            "0 disables the cap"
        );
    }

    #[test]
    fn oversized_tool_output_is_truncated_with_a_warning() {
        let output = (0..200)
            .map(|line| format!("line {line}: datum == expected"))
            .collect::<Vec<_>>()
            .join("\n");

        let truncated = truncate_tool_output(&output, 512);

        assert!(truncated.len() < output.len());
        assert!(truncated.contains("truncated:"));
        assert!(truncated.contains(&output.len().to_string()));
        assert!(
            truncated.ends_with("complete output.)"),
            "the model must be told the remainder is missing"
        );
        let body = truncated
            .split("\n… (truncated:")
            .next()
            .expect("body before the warning");
        assert!(
            output.starts_with(body) && body.ends_with('\n'),
            "truncation should stop on a line boundary"
        );
    }

    #[test]
    fn truncation_respects_utf8_boundaries() {
        let output = "ñ".repeat(400);
        let truncated = truncate_tool_output(&output, 101);
        assert!(truncated.contains("truncated:"));
    }

    fn tool_result_message(request: &str, output: &str) -> Value {
        json!({
            "role": "user",
            "content": build_tool_result_user_prompt(
                &ReadRequest::ReadFile {
                    path: request.to_string(),
                },
                output,
                DEFAULT_MAX_TOOL_OUTPUT_BYTES,
            ),
        })
    }

    fn conversation_with_tool_results(count: usize) -> Vec<Value> {
        let mut messages = vec![json!({"role": "system", "content": "keep me"})];

        for index in 0..count {
            messages.push(json!({"role": "assistant", "content": format!("step {index}")}));
            messages.push(tool_result_message(
                &format!("lib/{index}.ak"),
                &format!("body {index}"),
            ));
        }

        messages
    }

    fn elided_count(messages: &[Value]) -> usize {
        messages
            .iter()
            .filter(|message| {
                message["content"]
                    .as_str()
                    .is_some_and(|text| text.contains("was elided"))
            })
            .count()
    }

    #[test]
    fn tool_result_template_starts_with_the_prune_marker() {
        // Pruning recognises results by this prefix, so a template change that
        // drops it must fail here rather than silently disabling elision.
        let prompt = build_tool_result_user_prompt(
            &ReadRequest::ReadFile {
                path: "lib/a.ak".to_string(),
            },
            "body",
            DEFAULT_MAX_TOOL_OUTPUT_BYTES,
        );

        assert!(
            prompt.starts_with(TOOL_RESULT_MARKER),
            "tool-result template no longer starts with {TOOL_RESULT_MARKER:?}"
        );
    }

    #[test]
    fn pruning_elides_only_older_results() {
        let mut messages = conversation_with_tool_results(TOOL_RESULT_PRUNE_TRIGGER);

        prune_stale_tool_results(&mut messages, crate::logging::LogLevel::Off);

        assert_eq!(
            elided_count(&messages),
            TOOL_RESULT_PRUNE_TRIGGER - MAX_RETAINED_TOOL_RESULTS
        );
        assert_eq!(messages[0]["content"], "keep me", "the prompt survives");

        let elided = messages[2]["content"].as_str().expect("content");
        assert!(elided.contains("was elided"));
        assert!(
            elided.contains("lib/0.ak"),
            "the stub should name the request it replaced: {elided}"
        );

        let newest = messages
            .iter()
            .rev()
            .find(|message| is_tool_result_message(message))
            .expect("a tool result");
        assert!(
            newest["content"].as_str().is_some_and(
                |text| text.contains(&format!("body {}", TOOL_RESULT_PRUNE_TRIGGER - 1))
            ),
            "the newest result stays verbatim"
        );
    }

    #[test]
    fn pruning_waits_for_the_trigger() {
        let mut messages = conversation_with_tool_results(TOOL_RESULT_PRUNE_TRIGGER - 1);

        prune_stale_tool_results(&mut messages, crate::logging::LogLevel::Off);

        assert_eq!(elided_count(&messages), 0);
    }

    #[test]
    fn pruning_survives_an_inserted_message() {
        // An index-based implementation would elide the wrong entries here.
        let mut messages = conversation_with_tool_results(TOOL_RESULT_PRUNE_TRIGGER);
        messages.insert(0, json!({"role": "system", "content": "inserted later"}));

        prune_stale_tool_results(&mut messages, crate::logging::LogLevel::Off);

        assert_eq!(messages[0]["content"], "inserted later");
        assert_eq!(messages[1]["content"], "keep me");
        assert_eq!(
            elided_count(&messages),
            TOOL_RESULT_PRUNE_TRIGGER - MAX_RETAINED_TOOL_RESULTS
        );
    }

    #[test]
    fn pruning_survives_a_removed_message() {
        let mut messages = conversation_with_tool_results(TOOL_RESULT_PRUNE_TRIGGER);
        messages.remove(0);

        prune_stale_tool_results(&mut messages, crate::logging::LogLevel::Off);

        assert_eq!(
            elided_count(&messages),
            TOOL_RESULT_PRUNE_TRIGGER - MAX_RETAINED_TOOL_RESULTS
        );
        assert!(
            messages
                .iter()
                .filter(|message| is_tool_result_message(message))
                .all(|message| !message["content"]
                    .as_str()
                    .is_some_and(|text| text.contains("was elided was elided"))),
            "a pruned stub must not be pruned again"
        );
    }

    #[test]
    fn pruning_is_idempotent() {
        let mut messages = conversation_with_tool_results(TOOL_RESULT_PRUNE_TRIGGER);
        prune_stale_tool_results(&mut messages, crate::logging::LogLevel::Off);
        let after_first = elided_count(&messages);

        prune_stale_tool_results(&mut messages, crate::logging::LogLevel::Off);

        assert_eq!(elided_count(&messages), after_first);
    }

    #[test]
    fn brief_step_lines_stay_scannable() {
        let line = format!(
            "step {}/{}  {:<32} {:>8}  {}{}",
            3,
            30,
            truncate_action("read_file lib/very/long/path/to/some/validator.ak"),
            crate::logging::format_bytes(20480),
            crate::logging::format_duration(Duration::from_millis(9100)),
            format_usage_suffix(&TokenUsage::default())
        );

        assert!(line.starts_with("step 3/30  read_file lib/very/long/path/to…"));
        assert!(line.contains("20.0 KB"));
        assert!(line.contains("9.1s"));
        assert_eq!(truncate_action("grep"), "grep");
    }

    #[test]
    fn additional_steps_default_to_effective_cap() {
        assert_eq!(additional_steps_from_answer("", 7), Some(7));
        assert_eq!(additional_steps_from_answer("", 0), None);
        assert_eq!(additional_steps_from_answer(" \n", 60), Some(60));
        assert_eq!(additional_steps_from_answer("5\n", 7), Some(5));
        assert_eq!(additional_steps_from_answer("0", 7), None);
        assert_eq!(additional_steps_from_answer("not-a-number", 7), None);
    }

    #[test]
    fn non_interactive_runs_do_not_prompt_for_more_steps() {
        let result = prompt_for_additional_agent_steps("Test provider", "skill", 30, false)
            .expect("non-interactive prompt should not fail");

        assert!(result.is_none());
    }

    #[test]
    fn agent_loop_stops_when_skill_budget_is_exhausted() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path().canonicalize().expect("canonical root");
        let prompt = PermissionPromptSpec {
            shell: "bash".to_string(),
            allowed_commands: vec![],
            scope_rules: vec![],
            workspace_root: root.display().to_string(),
            read_scope: "workspace".to_string(),
            interactive_permissions: false,
            allowed_paths: vec![],
        };
        let mut context = loop_context(&root, &prompt);
        context.skill_timeout = Duration::ZERO;
        let mut messages = Vec::new();

        let error = run_agent_loop(&dummy_skill(), context, &mut messages, |_| {
            panic!("model must not be called after the skill budget is exhausted");
        })
        .expect_err("zero budget should stop immediately");

        assert!(error.to_string().contains("skill budget"));
    }

    #[test]
    fn the_loop_tells_the_model_when_a_file_is_unchanged() {
        // The model asked for the same file twice; the second answer must be the
        // cache note, not a second copy of the file.
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        let file = root.join("validators/spend.ak");
        std::fs::create_dir_all(file.parent().expect("parent")).expect("create dir");
        std::fs::write(&file, "validator spend {}\n").expect("write file");

        let canonical_root = root.canonicalize().expect("canonical root");
        let prompt = workspace_prompt(&canonical_root, vec!["cat"]);
        let read = json!({"action": "read_file", "path": "validators/spend.ak"}).to_string();
        let replies = [
            read.clone(),
            read,
            json!({"status": "completed", "findings": []}).to_string(),
        ];
        let mut sent = 0usize;
        let skill = dummy_skill();
        let mut messages = Vec::new();

        run_agent_loop(
            &skill,
            loop_context(&canonical_root, &prompt),
            &mut messages,
            |_| {
                let reply = replies[sent].clone();
                sent += 1;
                Ok((reply, TokenUsage::default()))
            },
        )
        .expect("loop should finish");

        let tool_results = messages
            .iter()
            .filter(|message| {
                message["content"]
                    .as_str()
                    .is_some_and(|text| text.starts_with("Tool result for "))
            })
            .collect::<Vec<_>>();

        assert_eq!(tool_results.len(), 2, "both reads produced a tool result");
        assert!(
            tool_results[0]["content"]
                .as_str()
                .is_some_and(|text| text.contains("validator spend")),
            "the first read carries the file"
        );
        assert!(
            tool_results[1]["content"]
                .as_str()
                .is_some_and(|text| text.contains("Unchanged since step 1")),
            "the second read is a cache note: {}",
            tool_results[1]["content"]
        );
    }

    #[test]
    fn agent_loop_recovers_from_one_malformed_reply() {
        // Reproduces the Sonnet 5 run that emitted "Tool disabled\nconsole.log('a')"
        // and took all 20 skills down with it.
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path().canonicalize().expect("canonical root");
        let prompt = PermissionPromptSpec {
            shell: "bash".to_string(),
            allowed_commands: vec![],
            scope_rules: vec![],
            workspace_root: root.display().to_string(),
            read_scope: "workspace".to_string(),
            interactive_permissions: false,
            allowed_paths: vec![],
        };

        let replies = [
            "\nTool disabled\nconsole.log('a')".to_string(),
            r#"{"status":"completed","findings":[]}"#.to_string(),
        ];
        let mut sent = 0usize;
        let skill = dummy_skill();
        let mut messages = Vec::new();

        let result = run_agent_loop(&skill, loop_context(&root, &prompt), &mut messages, |_| {
            let reply = replies[sent].clone();
            sent += 1;
            Ok((reply, TokenUsage::default()))
        })
        .expect("loop should recover and finish");

        assert_eq!(sent, 2, "the bad reply should have been retried");
        assert_eq!(result.findings.len(), 0);
        assert!(
            messages.iter().any(|m| m["content"]
                .as_str()
                .is_some_and(|c| c.contains("not valid JSON"))),
            "a corrective nudge should have been sent"
        );
    }

    #[test]
    fn agent_loop_gives_up_after_repeated_malformed_replies() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path().canonicalize().expect("canonical root");
        let prompt = PermissionPromptSpec {
            shell: "bash".to_string(),
            allowed_commands: vec![],
            scope_rules: vec![],
            workspace_root: root.display().to_string(),
            read_scope: "workspace".to_string(),
            interactive_permissions: false,
            allowed_paths: vec![],
        };

        let mut sent = 0usize;
        let skill = dummy_skill();
        let mut messages = Vec::new();

        let error = run_agent_loop(&skill, loop_context(&root, &prompt), &mut messages, |_| {
            sent += 1;
            Ok(("still not json".to_string(), TokenUsage::default()))
        })
        .expect_err("persistent garbage should fail the skill");

        assert_eq!(sent, MAX_MALFORMED_REPLIES + 1, "should stop after the cap");
        assert!(error.to_string().contains("unparseable replies in a row"));
    }

    #[test]
    fn iteration_from_parsed_reads_analysis_summary() {
        let parsed = json!({
            "status": "completed",
            "findings": [],
            "analysis_summary": "checked the spend path"
        });

        let iteration = iteration_from_parsed(&dummy_skill(), parsed, TokenUsage::default());

        assert_eq!(
            iteration.analysis_summary.as_deref(),
            Some("checked the spend path")
        );
        assert!(iteration.error.is_none());
    }

    #[test]
    fn iteration_from_parsed_defaults_error_for_failed_status() {
        let parsed = json!({
            "status": "failed",
            "findings": []
        });

        let iteration = iteration_from_parsed(&dummy_skill(), parsed, TokenUsage::default());

        assert_eq!(
            iteration.error.as_deref(),
            Some("provider reported failure without a reason")
        );
    }

    #[test]
    fn iteration_from_parsed_reads_error_when_present() {
        let parsed = json!({
            "status": "failed",
            "error": "429 Too Many Requests",
            "findings": []
        });

        let iteration = iteration_from_parsed(&dummy_skill(), parsed, TokenUsage::default());

        assert_eq!(iteration.error.as_deref(), Some("429 Too Many Requests"));
    }
}
