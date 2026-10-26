use clap::{Args as ClapArgs, ValueEnum};
use miette::{Context, IntoDiagnostic, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub mod ast;
pub mod logging;
pub mod model;
pub mod providers;

use self::ast::{generate_ast_and_validator_context, sha256_hex};
use self::logging::{LogLevel, format_duration_ms};
use self::model::{
    AnalysisStateJson, AstMetadata, MiniPrompt, PermissionPromptSpec, ProviderSpec,
    SkillIterationResult, ValidatorContextMap, VulnerabilityFinding, VulnerabilityReportSpec,
    VulnerabilitySkill,
};
use self::providers::shared::{UserStopped, glob_match, is_ignored_path};
use self::providers::{AnalysisProvider, build_provider};

const DEFAULT_SKILLS_DIR: &str = "skills/vulnerabilities";
/// Default destination for the timestamped progress log.
const DEFAULT_LOG_FILE: &str = ".tx3/audit/preflight.log";

/// Bumped when audit behavior changes without a binary version bump.
const AUDIT_CONTEXT_VERSION: &str = "1";
/// Bumped when heuristic detector behavior changes.
const HEURISTIC_DETECTOR_VERSION: &str = "1";
const AGENT_SYSTEM_PROMPT_TEMPLATE: &str =
    include_str!("../templates/aiken/audit_agent_system_prompt.md");
const INITIAL_USER_PROMPT_TEMPLATE: &str =
    include_str!("../templates/aiken/audit_agent_initial_user_prompt.md");
const TOOL_RESULT_PROMPT_TEMPLATE: &str =
    include_str!("../templates/aiken/audit_agent_tool_result_prompt.md");

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ReadScopeArg {
    Workspace,
    Strict,
}

impl ReadScopeArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Workspace => "workspace",
            Self::Strict => "strict",
        }
    }
}

#[derive(ClapArgs)]
pub struct Args {
    /// Path where the incremental analysis state JSON will be written.
    #[arg(long, default_value = ".tx3/audit/state.json")]
    pub state_out: String,

    /// Resume from the state file: skip completed skills and retry failed or
    /// missing ones. Source changes are rejected; skills whose content changed
    /// are re-run.
    #[arg(long)]
    pub resume: bool,

    /// With --resume, re-run completed skills when the provider or model changed.
    #[arg(long)]
    pub resume_strict: bool,

    /// Path where the final vulnerability report markdown will be written.
    #[arg(long, default_value = ".tx3/audit/vulnerabilities.md")]
    pub report_out: String,

    /// Path to vulnerability skill definitions.
    #[arg(long, default_value = "skills/vulnerabilities")]
    pub skills_dir: String,

    /// Analysis provider: scaffold | heuristic | openai | anthropic | ollama
    #[arg(long, default_value = "scaffold")]
    pub provider: String,

    /// API endpoint override. Default depends on --provider.
    #[arg(long)]
    pub endpoint: Option<String>,

    /// Model override. Default depends on --provider.
    #[arg(long)]
    pub model: Option<String>,

    /// API key environment variable override. Default depends on --provider.
    #[arg(long)]
    pub api_key_env: Option<String>,

    /// Print chat-style progress while auditing.
    ///
    /// Pass the flag alone for one compact block per skill and one line per
    /// step, or `--ai-logs=full` to see raw model output and reasoning.
    #[arg(
        long,
        value_enum,
        num_args = 0..=1,
        default_missing_value = "brief",
        default_value_t = LogLevel::Off
    )]
    pub ai_logs: LogLevel,

    /// Also write the progress log to this file, with timestamps.
    ///
    /// Only written when `--ai-logs` is set. Pass an empty value to disable it.
    #[arg(long, default_value = DEFAULT_LOG_FILE)]
    pub log_file: PathBuf,

    /// Maximum agent steps per skill before stopping.
    #[arg(long, default_value_t = crate::providers::shared::DEFAULT_MAX_AGENT_STEPS)]
    pub max_agent_steps: usize,

    /// Optional reasoning effort hint for OpenAI-compatible providers (e.g. low|medium|high).
    #[arg(long)]
    pub reasoning_effort: Option<String>,

    /// Path where the Aiken AST snapshot JSON will be written.
    #[arg(long, default_value = ".tx3/audit/aiken-ast.json")]
    pub ast_out: String,

    /// Regenerate AST even if an up-to-date snapshot is already available.
    #[arg(long, default_value_t = false)]
    pub no_ast_cache: bool,

    /// File read scope for AI-assisted local tool requests: workspace | strict.
    #[arg(long, value_enum, default_value_t = ReadScopeArg::Workspace)]
    pub read_scope: ReadScopeArg,

    /// Ask confirmation before executing each AI-requested local read action.
    #[arg(long, default_value_t = false)]
    pub interactive_permissions: bool,

    /// Fallback main source file path (used when no .ak files are discovered).
    #[arg(long)]
    pub main_source: Option<String>,

    /// Exclude `*.tests.ak` files from source discovery and the audit context.
    #[arg(long, default_value_t = false)]
    pub exclude_tests: bool,
}

/// Decides whether this run opens a log file.
///
/// The default path is written only when `--ai-logs` is on, so a silent run does
/// not create an empty file. A path the user named explicitly, or an empty path
/// used to opt out, is always honoured.
fn resolve_log_file(log_file: &Path, ai_logs: LogLevel) -> Option<PathBuf> {
    if log_file.as_os_str().is_empty() {
        return None;
    }

    if ai_logs.is_enabled() || log_file != Path::new(DEFAULT_LOG_FILE) {
        return Some(log_file.to_path_buf());
    }

    None
}

pub fn run(args: Args) -> Result<()> {
    let provider = build_provider(&args)?;
    run_analysis(args, provider.as_ref())
}

fn run_analysis(args: Args, provider: &dyn AnalysisProvider) -> Result<()> {
    // A file log keeps a durable record without scrolling the terminal, so it
    // runs alongside brief terminal output rather than replacing it.
    // The default path is only opened when logging is on, so a silent run
    // leaves no empty log behind. An explicitly named file is always written.
    let log_file = resolve_log_file(&args.log_file, args.ai_logs);
    let tee_terminal = args.ai_logs.is_enabled();
    logging::init(log_file.as_deref(), tee_terminal)?;

    let skills_dir = PathBuf::from(&args.skills_dir);
    let state_out = PathBuf::from(&args.state_out);
    let report_out = PathBuf::from(&args.report_out);
    let ast_out = PathBuf::from(&args.ast_out);
    let project_root = std::env::current_dir().into_diagnostic()?;
    let exclude_patterns = build_exclude_patterns(args.exclude_tests);
    let aiken_source_files = discover_source_files(&project_root, &exclude_patterns)?;
    let source_files = if aiken_source_files.is_empty() {
        // `--main-source` is an explicit opt-in to a single file, so it still
        // wins when excluding tests removed everything; warn so the override is
        // not silent.
        if let Some(main_source) = &args.main_source {
            if args.exclude_tests {
                eprintln!(
                    "⚠️  Excluding test files left no discovered sources; auditing --main-source {} instead.",
                    main_source
                );
            }
            vec![PathBuf::from(main_source)]
        } else if args.exclude_tests {
            return Err(miette::miette!(
                "No .ak source files left after excluding *.tests.ak; remove --exclude-tests or pass --main-source"
            ));
        } else {
            return Err(miette::miette!(
                "No .ak source files found and no --main-source specified"
            ));
        }
    } else {
        aiken_source_files.clone()
    };

    log_audit_progress(
        args.ai_logs,
        format!(
            "Starting audit • provider={} • source files={}",
            provider.provider_spec().name,
            source_files.len()
        ),
    );

    let permission_prompt = build_permission_prompt_spec(
        args.read_scope,
        args.interactive_permissions,
        &project_root,
        &source_files,
    );
    let skills = load_skills(&skills_dir, &args.skills_dir)?;
    let ast_context = generate_ast_and_validator_context(
        &project_root,
        &aiken_source_files,
        &ast_out,
        args.no_ast_cache,
    )?;

    let mut state = AnalysisStateJson {
        version: "2".to_string(),
        source_files: source_files
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        excluded_patterns: exclude_patterns.clone(),
        provider: provider.provider_spec(),
        permission_prompt: permission_prompt.clone(),
        ast: Some(ast_context.metadata.clone()),
        validator_context: ast_context.validator_context.clone(),
        iterations: vec![],
        total_token_usage: Default::default(),
        total_duration_ms: 0,
    };

    if args.resume {
        match load_resume_state(&state_out, &ast_context.metadata, &skills) {
            Ok(Some(previous)) => {
                if previous.provider.name != state.provider.name
                    || previous.provider.model != state.provider.model
                {
                    eprintln!(
                        "⚠️  Resuming with a different provider/model; completed iterations are kept as-is."
                    );
                }

                state.iterations = previous.iterations;
                state.total_token_usage = previous.total_token_usage;
                state.total_duration_ms = previous.total_duration_ms;
                log_audit_progress(
                    args.ai_logs,
                    format!(
                        "Resuming from {} • {} previous skill result(s)",
                        state_out.display(),
                        state.iterations.len()
                    ),
                );
            }
            Ok(None) => {
                log_audit_progress(
                    args.ai_logs,
                    format!(
                        "No previous state at {}; starting a fresh run",
                        state_out.display()
                    ),
                );
            }
            Err(error) => return Err(error).wrap_err("Cannot resume previous state"),
        }
    }

    write_state(&state_out, &state)?;

    let skill_loop_context = SkillLoopContext {
        source_files: &source_files,
        project_root: &project_root,
        permission_prompt: &permission_prompt,
        validator_context: &ast_context.validator_context,
        provider,
        ai_logs: args.ai_logs,
        state_out: &state_out,
        resume_strict: args.resume_strict,
    };

    // Write the report even when the skill loop aborts: a partial run still has
    // useful findings and the state file records what was covered.
    let loop_result = run_skill_loop(&skills, &mut state, skill_loop_context);

    let report = build_report(&state, &skills, loop_result.is_err());
    let report_markdown = render_report_markdown(&report);
    write_text_file(&report_out, &report_markdown)?;

    loop_result?;

    println!(
        "⚠️  EXPERIMENTAL: Audit complete. Iterations processed: {}",
        state.iterations.len()
    );
    println!("Source files analyzed: {}", state.source_files.len());
    if !state.total_token_usage.is_empty() {
        println!(
            "Token usage (total): {}",
            format_total_token_usage(&state.total_token_usage)
        );
    }
    println!("State written to: {}", state_out.display());
    println!("Report written to: {}", report_out.display());

    Ok(())
}

struct SkillLoopContext<'a> {
    source_files: &'a [PathBuf],
    project_root: &'a Path,
    permission_prompt: &'a PermissionPromptSpec,
    validator_context: &'a ValidatorContextMap,
    provider: &'a dyn AnalysisProvider,
    ai_logs: LogLevel,
    state_out: &'a Path,
    resume_strict: bool,
}

/// Consecutive skill failures tolerated before the run stops. A blocked account
/// fails every skill, and retrying 20 of them wastes both time and money.
const MAX_CONSECUTIVE_SKILL_FAILURES: usize = 3;

fn run_skill_loop(
    skills: &[VulnerabilitySkill],
    state: &mut AnalysisStateJson,
    context: SkillLoopContext<'_>,
) -> Result<()> {
    let source_references = context
        .source_files
        .iter()
        .map(|path| display_path_for_prompt(context.project_root, path))
        .collect::<Vec<String>>();

    let total_skills = skills.len();
    let provider_spec = context.provider.provider_spec();
    let mut consecutive_failures = 0usize;

    for (skill_idx, skill) in skills.iter().enumerate() {
        let prompt = build_mini_prompt(skill);
        let fingerprint = prompt_fingerprint(&prompt);

        if let Some(existing) = reusable_iteration(
            state,
            &skill.id,
            &fingerprint,
            &provider_spec,
            context.resume_strict,
        ) {
            log_audit_progress(
                context.ai_logs,
                format!(
                    "Skill {}/{} • skipped '{}' (already {} for this skill version)",
                    skill_idx + 1,
                    total_skills,
                    skill.id,
                    existing.status
                ),
            );
            continue;
        }

        log_audit_progress(
            context.ai_logs,
            format!(
                "▶ {}/{} {} ({})",
                skill_idx + 1,
                total_skills,
                skill.id,
                skill.name
            ),
        );

        let started = Instant::now();
        match run_skill_once(skill, &prompt, &fingerprint, &context, &source_references) {
            Ok(mut iteration) => {
                consecutive_failures = 0;
                iteration.duration_ms = Some(started.elapsed().as_millis() as u64);
                record_iteration(state, iteration, &context)?;
            }
            // One skill failing must not discard every skill after it. Record the
            // failure in the state file so the gap is visible in the report, then
            // carry on -- a rate limit that blocks skill 4 has usually cleared by
            // the time skill 9 comes around.
            Err(error) => {
                let user_stopped = error.downcast_ref::<UserStopped>().is_some();

                if !user_stopped {
                    consecutive_failures = consecutive_failures.saturating_add(1);
                }

                let error_text = truncate_error_text(&error.to_string());

                log_audit_progress(
                    context.ai_logs,
                    if user_stopped {
                        format!(
                            "Skill {}/{} • STOPPED '{}' by user: {}",
                            skill_idx + 1,
                            total_skills,
                            skill.id,
                            error
                        )
                    } else {
                        format!(
                            "Skill {}/{} • FAILED '{}' ({} in a row): {}",
                            skill_idx + 1,
                            total_skills,
                            skill.id,
                            consecutive_failures,
                            error
                        )
                    },
                );

                let failed = SkillIterationResult {
                    skill_id: skill.id.clone(),
                    status: "failed".to_string(),
                    findings: Vec::new(),
                    next_prompt: None,
                    skill_fingerprint: Some(fingerprint),
                    provider: Some(provider_spec.clone()),
                    duration_ms: Some(started.elapsed().as_millis() as u64),
                    analysis_summary: None,
                    error: Some(error_text),
                    token_usage: crate::model::TokenUsage::default(),
                };

                // Record the failure before aborting: the third consecutive
                // failure is still an attempted skill and must appear in state.
                upsert_iteration(state, failed);
                update_total_duration(state);
                write_state(context.state_out, state)?;

                if user_stopped {
                    return Err(error).wrap_err(format!(
                        "Stopping: audit stopped by user at the agent step limit; {} remaining skill(s) were not attempted",
                        total_skills.saturating_sub(skill_idx + 1)
                    ));
                }

                if consecutive_failures >= MAX_CONSECUTIVE_SKILL_FAILURES {
                    return Err(error).wrap_err(format!(
                        "Stopping: {} skills failed in a row, the remaining {} were not attempted",
                        consecutive_failures,
                        total_skills.saturating_sub(skill_idx + 1)
                    ));
                }
            }
        }
    }

    sort_iterations_by_skill_order(state, skills);
    update_total_duration(state);
    write_state(context.state_out, state)?;

    Ok(())
}

fn run_skill_once(
    skill: &VulnerabilitySkill,
    prompt: &MiniPrompt,
    fingerprint: &str,
    context: &SkillLoopContext<'_>,
    source_references: &[String],
) -> Result<SkillIterationResult> {
    let mut iteration = context.provider.analyze_skill(
        skill,
        prompt,
        source_references,
        context.validator_context,
        context.project_root,
        context.permission_prompt,
    )?;
    iteration.skill_fingerprint = Some(fingerprint.to_string());
    iteration.provider = Some(context.provider.provider_spec());
    Ok(iteration)
}

fn record_iteration(
    state: &mut AnalysisStateJson,
    iteration: SkillIterationResult,
    context: &SkillLoopContext<'_>,
) -> Result<()> {
    let skill_id = iteration.skill_id.clone();
    let findings_count = iteration.findings.len();
    let status = iteration.status.clone();
    let iteration_usage = iteration.token_usage.clone();

    upsert_iteration(state, iteration);
    state.total_token_usage.add(&iteration_usage);
    update_total_duration(state);
    write_state(context.state_out, state)?;

    log_audit_progress(
        context.ai_logs,
        format!(
            "✔ {} {} findings={} tokens=({}) state saved",
            skill_id,
            status,
            findings_count,
            format_total_token_usage(&iteration_usage)
        ),
    );

    Ok(())
}

fn upsert_iteration(state: &mut AnalysisStateJson, iteration: SkillIterationResult) {
    if let Some(existing) = state
        .iterations
        .iter_mut()
        .find(|existing| existing.skill_id == iteration.skill_id)
    {
        *existing = iteration;
    } else {
        state.iterations.push(iteration);
    }
}

fn reusable_iteration<'a>(
    state: &'a AnalysisStateJson,
    skill_id: &str,
    fingerprint: &str,
    current_provider: &ProviderSpec,
    strict: bool,
) -> Option<&'a SkillIterationResult> {
    state.iterations.iter().rev().find(|iteration| {
        iteration.skill_id == skill_id
            && iteration.skill_fingerprint.as_deref() == Some(fingerprint)
            && iteration_is_reusable(&iteration.status, &current_provider.name)
            && (!strict || provider_matches(iteration.provider.as_ref(), current_provider))
    })
}

fn iteration_is_reusable(status: &str, provider_name: &str) -> bool {
    match status {
        "completed" => true,
        // These statuses are terminal only for the provider that produced them.
        "unsupported-skill" => provider_name == "heuristic",
        "scaffolded" => provider_name == "scaffold",
        _ => false,
    }
}

fn provider_matches(previous: Option<&ProviderSpec>, current: &ProviderSpec) -> bool {
    previous.is_some_and(|previous| {
        previous.name == current.name
            && previous.model == current.model
            && previous.notes == current.notes
    })
}

fn sort_iterations_by_skill_order(state: &mut AnalysisStateJson, skills: &[VulnerabilitySkill]) {
    state.iterations.sort_by_key(|iteration| {
        skills
            .iter()
            .position(|skill| skill.id == iteration.skill_id)
            .unwrap_or(usize::MAX)
    });
}

fn audit_context_fingerprint() -> String {
    format!(
        "{}:{}:{}:{}:{}:{}",
        AUDIT_CONTEXT_VERSION,
        HEURISTIC_DETECTOR_VERSION,
        env!("CARGO_PKG_VERSION"),
        AGENT_SYSTEM_PROMPT_TEMPLATE,
        INITIAL_USER_PROMPT_TEMPLATE,
        TOOL_RESULT_PROMPT_TEMPLATE
    )
}

fn prompt_fingerprint(prompt: &MiniPrompt) -> String {
    prompt_fingerprint_with_context(prompt, &audit_context_fingerprint())
}

fn prompt_fingerprint_with_context(prompt: &MiniPrompt, context: &str) -> String {
    format!(
        "sha256:{}",
        sha256_hex(format!("{}\n{}", context, prompt.text).as_bytes())
    )
}

fn update_total_duration(state: &mut AnalysisStateJson) {
    state.total_duration_ms = state
        .iterations
        .iter()
        .filter_map(|iteration| iteration.duration_ms)
        .sum();
}

fn load_resume_state(
    state_out: &Path,
    current_ast: &AstMetadata,
    skills: &[VulnerabilitySkill],
) -> Result<Option<AnalysisStateJson>> {
    if !state_out.exists() {
        return Ok(None);
    }

    let text = std::fs::read_to_string(state_out)
        .into_diagnostic()
        .with_context(|| format!("Failed to read state file {}", state_out.display()))?;
    let mut previous: AnalysisStateJson = serde_json::from_str(&text)
        .into_diagnostic()
        .with_context(|| format!("Failed to parse state file {}", state_out.display()))?;

    if previous.version != "2" {
        return Err(miette::miette!(
            "State file {} uses unsupported version '{}' (expected '2'); run without --resume for a fresh audit",
            state_out.display(),
            previous.version
        ));
    }

    let previous_source_fingerprint = previous
        .ast
        .as_ref()
        .map(|ast| ast.source_fingerprint.as_str())
        .unwrap_or_default();

    if previous_source_fingerprint.is_empty() || current_ast.source_fingerprint.is_empty() {
        return Err(miette::miette!(
            "State file {} has no source fingerprint; run without --resume for a fresh audit",
            state_out.display()
        ));
    }

    if previous_source_fingerprint != current_ast.source_fingerprint {
        return Err(miette::miette!(
            "Sources changed since {} was written; run without --resume for a fresh audit",
            state_out.display()
        ));
    }

    // Keep one iteration per currently loaded skill, in the current skill order.
    // Iterations for removed skills are dropped so their findings cannot leak
    // into the report.
    let mut kept = Vec::new();
    for skill in skills {
        if let Some(iteration) = previous
            .iterations
            .iter()
            .rev()
            .find(|iteration| iteration.skill_id == skill.id)
        {
            kept.push(iteration.clone());
        }
    }
    previous.iterations = kept;

    Ok(Some(previous))
}

fn format_total_token_usage(usage: &crate::model::TokenUsage) -> String {
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

fn log_audit_progress(level: LogLevel, message: impl AsRef<str>) {
    if level.is_enabled() {
        logging::write_line("🧭 ", message.as_ref());
    }
}

fn display_path_for_prompt(project_root: &Path, path: &Path) -> String {
    path.strip_prefix(project_root)
        .map(|relative| relative.display().to_string())
        .unwrap_or_else(|_| path.display().to_string())
}

const TEST_SOURCE_GLOB: &str = "*.tests.ak";

fn build_exclude_patterns(exclude_tests: bool) -> Vec<String> {
    if exclude_tests {
        vec![TEST_SOURCE_GLOB.to_string()]
    } else {
        Vec::new()
    }
}

fn discover_source_files(project_root: &Path, exclude_patterns: &[String]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut to_visit = vec![project_root.to_path_buf()];

    while let Some(dir) = to_visit.pop() {
        let entries = std::fs::read_dir(&dir)
            .into_diagnostic()
            .with_context(|| format!("Failed to read directory {}", dir.display()))?;

        for entry in entries {
            let entry = entry.into_diagnostic()?;
            let path = entry.path();

            if path.is_dir() {
                // A `.git`/`target`/`.tx3`/`build` directory, or a subdirectory
                // with its own aiken.toml (vendored, legacy, or an example), is a
                // separate concern and is not part of this audit.
                if !is_ignored_path(project_root, &path) {
                    to_visit.push(path);
                }
                continue;
            }

            let is_source_file = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("ak"))
                .unwrap_or(false);

            if is_source_file && !is_excluded_source(project_root, &path, exclude_patterns) {
                files.push(path);
            }
        }
    }

    files.sort();
    Ok(files)
}

fn is_excluded_source(project_root: &Path, path: &Path, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return false;
    }

    let relative = path.strip_prefix(project_root).unwrap_or(path);
    let relative = relative.to_string_lossy();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();

    patterns
        .iter()
        .any(|pattern| glob_match(pattern, &relative) || glob_match(pattern, file_name))
}

fn build_mini_prompt(skill: &VulnerabilitySkill) -> MiniPrompt {
    let text = compose_skill_prompt(skill);

    MiniPrompt {
        skill_id: skill.id.clone(),
        text,
    }
}

fn compose_skill_prompt(skill: &VulnerabilitySkill) -> String {
    let mut sections = vec![
        "Skill Summary (use as source of truth):".to_string(),
        format!("- Skill ID: {}", skill.id),
        format!("- Name: {}", skill.name),
        format!("- Severity: {}", skill.severity),
        format!("- Description: {}", skill.description),
        format!("- Primary detection objective: {}", skill.prompt_fragment),
    ];

    if !skill.tags.is_empty() {
        sections.push(format!("- Tags: {}", skill.tags.join(", ")));
    }

    if let Some(hint) = &skill.confidence_hint {
        sections.push(format!("- Confidence Hint: {}", hint));
    }

    if !skill.examples.is_empty() {
        sections.push(format!("Examples:\n- {}", skill.examples.join("\n- ")));
    }

    if !skill.false_positives.is_empty() {
        sections.push(format!(
            "False Positives To Avoid:\n- {}",
            skill.false_positives.join("\n- ")
        ));
    }

    if !skill.references.is_empty() {
        sections.push(format!("References:\n- {}", skill.references.join("\n- ")));
    }

    if !skill.guidance_markdown.trim().is_empty() {
        sections.push(format!(
            "Detailed Guidance (authoritative):\n{}",
            skill.guidance_markdown.trim()
        ));
    }

    sections.join("\n\n")
}

fn build_permission_prompt_spec(
    read_scope: ReadScopeArg,
    interactive_permissions: bool,
    project_root: &Path,
    source_files: &[PathBuf],
) -> PermissionPromptSpec {
    let allowed_paths = if matches!(read_scope, ReadScopeArg::Strict) {
        source_files
            .iter()
            .map(|path| display_path_for_prompt(project_root, path))
            .collect::<Vec<String>>()
    } else {
        vec![]
    };

    let mut scope_rules = vec![
        "Only execute commands within the current project root.".to_string(),
        "Do not write outside designated output artifacts.".to_string(),
    ];

    if matches!(read_scope, ReadScopeArg::Strict) {
        scope_rules.push(
            "Read scope is strict: only known source files are allowed for reads/searches; directory listing and file discovery requests are denied.".to_string(),
        );
    } else {
        scope_rules.push(
            "Read scope is workspace: any path under project root can be read/searched, except generated/excluded directories (.git, target, .tx3, build) and nested projects, which are never accessible."
                .to_string(),
        );
    }

    if interactive_permissions {
        scope_rules.push(
            "Interactive permissions are enabled: every local read action requires explicit user confirmation."
                .to_string(),
        );
    }

    PermissionPromptSpec {
        shell: "bash".to_string(),
        allowed_commands: vec![
            "grep".to_string(),
            "cat".to_string(),
            "find".to_string(),
            "ls".to_string(),
        ],
        scope_rules,
        workspace_root: project_root.display().to_string(),
        read_scope: read_scope.as_str().to_string(),
        interactive_permissions,
        allowed_paths,
    }
}

fn build_report(
    state: &AnalysisStateJson,
    skills: &[VulnerabilitySkill],
    aborted: bool,
) -> VulnerabilityReportSpec {
    let findings = state
        .iterations
        .iter()
        .flat_map(|iteration| iteration.findings.iter().cloned())
        .collect::<Vec<VulnerabilityFinding>>();

    VulnerabilityReportSpec {
        title: "Vulnerability Report".to_string(),
        generated_at: chrono::Utc::now().to_rfc3339(),
        coverage_markdown: render_coverage_markdown(state, skills, findings.len(), aborted),
        findings,
    }
}

/// Renders what was and was not covered. A report that only lists findings
/// silently reads as "everything was checked and nothing else was found", even
/// when a skill failed or never ran -- so failures and gaps belong in the
/// report next to the findings.
fn render_coverage_markdown(
    state: &AnalysisStateJson,
    skills: &[VulnerabilitySkill],
    findings_count: usize,
    aborted: bool,
) -> String {
    let mut completed = 0usize;
    let mut failed = 0usize;
    let mut unsupported = 0usize;
    let mut scaffolded = 0usize;
    let mut not_run = 0usize;
    let mut skill_lines = Vec::new();

    for skill in skills {
        let iteration = state
            .iterations
            .iter()
            .rev()
            .find(|iteration| iteration.skill_id == skill.id);

        let (status, findings, summary, error) = match iteration {
            Some(iteration) => (
                iteration.status.as_str(),
                iteration.findings.len(),
                iteration.analysis_summary.as_deref(),
                iteration.error.as_deref(),
            ),
            None => ("not run", 0, None, None),
        };

        match status {
            "completed" => completed += 1,
            "failed" => failed += 1,
            "unsupported-skill" => unsupported += 1,
            "scaffolded" => scaffolded += 1,
            "not run" => not_run += 1,
            _ => {}
        }

        let provider = iteration
            .and_then(|iteration| iteration.provider.as_ref())
            .map(format_provider)
            .unwrap_or_else(|| "provider unknown".to_string());
        let duration = iteration
            .and_then(|iteration| iteration.duration_ms)
            .map(format_duration_ms)
            .unwrap_or_else(|| "time unknown".to_string());
        let error = error
            .map(|value| format!(" — error: {}", truncate_error_text(value)))
            .unwrap_or_default();
        let summary = summary
            .map(|value| format!(" — summary: {}", truncate_summary_text(value)))
            .unwrap_or_default();

        skill_lines.push(format!(
            "- `{}` — {} — {} finding(s) — {} — {}{}{}",
            skill.id, status, findings, provider, duration, error, summary
        ));
    }

    let ignored = state
        .iterations
        .iter()
        .filter(|iteration| !skills.iter().any(|skill| skill.id == iteration.skill_id))
        .count();

    let incomplete = aborted || failed > 0 || not_run > 0;
    let mut coverage = String::new();
    if incomplete {
        let reason = if aborted {
            "the skill loop stopped early".to_string()
        } else {
            let mut parts = Vec::new();
            if failed > 0 {
                parts.push(format!("{} skill(s) failed", failed));
            }
            if not_run > 0 {
                parts.push(format!("{} skill(s) not run", not_run));
            }
            parts.join(", ")
        };
        coverage.push_str(&format!("- Status: INCOMPLETE — {}.\n", reason));
    }
    coverage.push_str(&format!(
        "- Skills: {} completed, {} failed, {} unsupported, {} scaffolded, {} not run ({} total)\n\
         - Source files analyzed: {}\n\
         - Findings: {}\n\
         - Total tokens: {}\n\
         - Total time: {}",
        completed,
        failed,
        unsupported,
        scaffolded,
        not_run,
        skills.len(),
        state.source_files.len(),
        findings_count,
        format_total_token_usage(&state.total_token_usage),
        format_duration_ms(state.total_duration_ms),
    ));

    if !state.excluded_patterns.is_empty() {
        coverage.push_str(&format!(
            "\n- Excluded patterns: {}",
            state.excluded_patterns.join(", ")
        ));
    }

    if ignored > 0 {
        coverage.push_str(&format!(
            "\n- Stale iterations ignored (skill no longer loaded): {}",
            ignored
        ));
    }

    if !skill_lines.is_empty() {
        coverage.push_str("\n\n### Skill results\n\n");
        coverage.push_str(&skill_lines.join("\n"));
    }

    coverage
}

fn format_provider(provider: &ProviderSpec) -> String {
    match &provider.model {
        Some(model) => format!("{} ({})", provider.name, model),
        None => provider.name.clone(),
    }
}

fn truncate_summary_text(summary: &str) -> String {
    const MAX_CHARS: usize = 240;
    let summary = summary.trim().replace('\n', " ");
    if summary.chars().count() <= MAX_CHARS {
        return summary;
    }

    let truncated = summary.chars().take(MAX_CHARS).collect::<String>();
    format!("{}…", truncated)
}

fn truncate_error_text(error: &str) -> String {
    const MAX_CHARS: usize = 500;
    const HEAD_CHARS: usize = 300;
    const TAIL_CHARS: usize = 150;

    let error = error.trim().replace('\n', " ");
    let total = error.chars().count();
    if total <= MAX_CHARS {
        return error;
    }

    // Keep both ends: miette context wrappers put the request at the head and
    // the actionable cause (for example "Stopping: ...") at the tail.
    let head = error.chars().take(HEAD_CHARS).collect::<String>();
    let tail = error.chars().skip(total - TAIL_CHARS).collect::<String>();
    let omitted = total.saturating_sub(HEAD_CHARS + TAIL_CHARS);

    format!("{} … <omitted {} chars> … {}", head, omitted, tail)
}

fn load_skills(skills_dir: &Path, skills_dir_arg: &str) -> Result<Vec<VulnerabilitySkill>> {
    if !skills_dir.exists() {
        if skills_dir_arg == DEFAULT_SKILLS_DIR {
            return load_embedded_seed_skills();
        }

        return Err(miette::miette!(
            "Audit skills directory not found: {}",
            skills_dir.display()
        ));
    }

    let mut entries = std::fs::read_dir(skills_dir)
        .into_diagnostic()
        .context("Failed to read skills directory")?
        .filter_map(|entry| entry.ok().map(|value| value.path()))
        .filter(|path| path.is_file())
        .collect::<Vec<PathBuf>>();

    entries.sort();

    let mut skills = entries
        .iter()
        .map(|path| load_skill_from_file(path))
        .collect::<Result<Vec<VulnerabilitySkill>>>()?;

    sort_skills_by_severity(&mut skills);

    if skills.is_empty() {
        return Err(miette::miette!(
            "No vulnerability skills found in {}",
            skills_dir.display()
        ));
    }

    Ok(skills)
}

fn load_embedded_seed_skills() -> Result<Vec<VulnerabilitySkill>> {
    let seed_files = [
        (
            Path::new("skills/vulnerabilities/001-strict-value-equality.md"),
            include_str!("../skills/vulnerabilities/001-strict-value-equality.md"),
        ),
        (
            Path::new("skills/vulnerabilities/002-missing-address-validation.md"),
            include_str!("../skills/vulnerabilities/002-missing-address-validation.md"),
        ),
        (
            Path::new("skills/vulnerabilities/003-unvalidated-datum.md"),
            include_str!("../skills/vulnerabilities/003-unvalidated-datum.md"),
        ),
    ];

    let mut skills = seed_files
        .iter()
        .map(|(path, content)| parse_skill_content(path, content))
        .collect::<Result<Vec<VulnerabilitySkill>>>()?;

    sort_skills_by_severity(&mut skills);

    Ok(skills)
}

fn severity_rank(severity: &str) -> u8 {
    match severity.trim().to_ascii_lowercase().as_str() {
        "critical" => 0,
        "high" => 1,
        "medium" => 2,
        "low" => 3,
        _ => 4,
    }
}

/// Run higher-impact checks before low-severity optimizations, so a broad
/// low-value skill cannot consume the budget ahead of a critical check.
fn sort_skills_by_severity(skills: &mut [VulnerabilitySkill]) {
    skills.sort_by(|a, b| {
        severity_rank(&a.severity)
            .cmp(&severity_rank(&b.severity))
            .then_with(|| a.id.cmp(&b.id))
    });
}

fn load_skill_from_file(path: &Path) -> Result<VulnerabilitySkill> {
    let content = std::fs::read_to_string(path)
        .into_diagnostic()
        .with_context(|| format!("Failed to read vulnerability skill file {}", path.display()))?;

    parse_skill_content(path, &content)
}

fn parse_skill_content(path: &Path, content: &str) -> Result<VulnerabilitySkill> {
    let (frontmatter, body) = split_frontmatter(content).with_context(|| {
        format!(
            "Failed to parse frontmatter from vulnerability skill file {}",
            path.display()
        )
    })?;

    let normalized_frontmatter = normalize_yaml_indentation(&frontmatter);

    let parsed: SkillFrontmatter = serde_yaml_ng::from_str(&normalized_frontmatter)
        .into_diagnostic()
        .with_context(|| {
            format!(
                "Invalid YAML frontmatter in vulnerability skill file {}",
                path.display()
            )
        })?;

    let severity = parsed.severity.trim().to_ascii_lowercase();
    if !matches!(severity.as_str(), "low" | "medium" | "high" | "critical") {
        return Err(miette::miette!(
            "Invalid `severity` value '{}' in vulnerability skill file {}. Expected one of: low, medium, high, critical",
            parsed.severity,
            path.display()
        ));
    }

    Ok(VulnerabilitySkill {
        id: require_non_empty("id", path, parsed.id)?,
        name: require_non_empty("name", path, parsed.name)?,
        severity,
        description: require_non_empty("description", path, parsed.description)?,
        prompt_fragment: require_non_empty("prompt_fragment", path, parsed.prompt_fragment)?,
        examples: parsed.examples,
        false_positives: parsed.false_positives,
        references: parsed.references,
        tags: parsed.tags,
        confidence_hint: parsed
            .confidence_hint
            .filter(|value| !value.trim().is_empty()),
        guidance_markdown: body.trim().to_string(),
    })
}

fn normalize_yaml_indentation(input: &str) -> String {
    input.replace('\t', "  ")
}

fn split_frontmatter(content: &str) -> Result<(String, String)> {
    let content = content.trim_start_matches('\u{feff}');
    let mut lines = content.lines();

    let Some(first_line) = lines.next() else {
        return Err(miette::miette!("Skill file is empty"));
    };

    if first_line.trim() != "---" {
        return Err(miette::miette!("Missing frontmatter start delimiter `---`"));
    }

    let mut frontmatter_lines = Vec::new();
    let mut found_end = false;

    for line in lines.by_ref() {
        if line.trim() == "---" {
            found_end = true;
            break;
        }
        frontmatter_lines.push(line);
    }

    if !found_end {
        return Err(miette::miette!("Missing frontmatter end delimiter `---`"));
    }

    let body_lines = lines.collect::<Vec<_>>();

    Ok((frontmatter_lines.join("\n"), body_lines.join("\n")))
}

fn require_non_empty(field: &str, path: &Path, value: String) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(miette::miette!(
            "Field `{}` must be non-empty in vulnerability skill file {}",
            field,
            path.display()
        ));
    }

    Ok(trimmed.to_string())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SkillFrontmatter {
    id: String,
    name: String,
    severity: String,
    description: String,
    prompt_fragment: String,
    #[serde(default)]
    examples: Vec<String>,
    #[serde(default)]
    false_positives: Vec<String>,
    #[serde(default)]
    references: Vec<String>,
    #[serde(default)]
    tags: Vec<String>,
    confidence_hint: Option<String>,
}

fn write_state(path: &Path, state: &AnalysisStateJson) -> Result<()> {
    let serialized = serde_json::to_string_pretty(state).into_diagnostic()?;
    write_text_file(path, &serialized)
}

fn write_text_file(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .into_diagnostic()
            .with_context(|| format!("Failed to create output directory {}", parent.display()))?;
    }

    std::fs::write(path, content)
        .into_diagnostic()
        .with_context(|| format!("Failed to write file {}", path.display()))
}

fn render_report_markdown(report: &VulnerabilityReportSpec) -> String {
    let template = include_str!("../templates/aiken/report.md");
    let findings_markdown = render_findings_markdown(&report.findings);

    template
        .replace("{{ generated_at }}", &report.generated_at)
        .replace("{{ coverage_markdown }}", &report.coverage_markdown)
        .replace("{{ findings_markdown }}", &findings_markdown)
}

fn render_findings_markdown(findings: &[VulnerabilityFinding]) -> String {
    if findings.is_empty() {
        return "- *(none)*".to_string();
    }

    findings
        .iter()
        .map(|finding| {
            let mut markdown = format!(
                "- **{}** (`{}`)\n  - Summary: {}\n  - Recommendation: {}",
                finding.title, finding.severity, finding.summary, finding.recommendation
            );

            let location = match (&finding.file, finding.line) {
                (Some(file), Some(line)) => Some(format!("{}:{}", file, line)),
                (Some(file), None) => Some(file.clone()),
                (None, Some(line)) => Some(format!("line {}", line)),
                (None, None) => None,
            };

            if let Some(location) = location {
                markdown.push_str(&format!("\n  - Location: {}", location));
            }

            markdown
        })
        .collect::<Vec<String>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn log_file_default_only_applies_when_logging_is_on() {
        let default = Path::new(DEFAULT_LOG_FILE);
        let custom = Path::new("/tmp/custom.log");

        assert_eq!(resolve_log_file(default, LogLevel::Off), None);
        assert_eq!(
            resolve_log_file(default, LogLevel::Brief),
            Some(default.to_path_buf())
        );
        assert_eq!(
            resolve_log_file(custom, LogLevel::Off),
            Some(custom.to_path_buf()),
            "an explicitly named file is always written"
        );
        assert_eq!(
            resolve_log_file(Path::new(""), LogLevel::Full),
            None,
            "an empty path opts out"
        );
    }

    #[test]
    fn parse_skill_content_reads_frontmatter_and_guidance() {
        let content = r#"---
id: strict-value-equality-001
name: Strict value equality
severity: high
description: Detect strict equality checks for ADA.
prompt_fragment: Find strict equality on ADA or full values.
examples:
  - output.value == expected
tags:
  - plutus-v2
confidence_hint: medium
---
# Instructions

Check validator outputs and avoid false positives for without_lovelace().
"#;

        let skill = parse_skill_content(Path::new("skill.md"), content).expect("should parse");

        assert_eq!(skill.id, "strict-value-equality-001");
        assert_eq!(skill.name, "Strict value equality");
        assert_eq!(skill.severity, "high");
        assert_eq!(skill.examples.len(), 1);
        assert!(skill.guidance_markdown.contains("# Instructions"));
    }

    #[test]
    fn parse_skill_content_requires_frontmatter() {
        let content = "id: foo";
        let error = parse_skill_content(Path::new("skill.md"), content).expect_err("should fail");
        assert!(error.to_string().contains("frontmatter"));
    }

    #[test]
    fn parse_skill_content_rejects_invalid_severity() {
        let content = r#"---
id: skill-1
name: Test skill
severity: urgent
description: desc
prompt_fragment: prompt
---
body
"#;

        let error = parse_skill_content(Path::new("skill.md"), content).expect_err("should fail");
        assert!(error.to_string().contains("Invalid `severity` value"));
    }

    #[test]
    fn discover_source_files_finds_ak_files_recursively() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        let validators = root.join("onchain/validators");

        fs::create_dir_all(&validators).expect("create validators dir");
        fs::write(validators.join("spend.ak"), "validator spend {}").expect("write ak file");
        fs::write(validators.join("readme.md"), "# ignore").expect("write non-ak file");

        let files = discover_source_files(root, &[]).expect("should discover files");

        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("onchain/validators/spend.ak"));
    }

    #[test]
    fn discover_source_files_skips_target_tx3_and_build_dirs() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();

        let normal_dir = root.join("contracts");
        let target_dir = root.join("target/generated");
        let tx3_dir = root.join(".tx3/tmp");
        let build_dir = root.join("build/output");

        fs::create_dir_all(&normal_dir).expect("create normal dir");
        fs::create_dir_all(&target_dir).expect("create target dir");
        fs::create_dir_all(&tx3_dir).expect("create tx3 dir");
        fs::create_dir_all(&build_dir).expect("create build dir");

        fs::write(normal_dir.join("ok.ak"), "validator ok {}").expect("write ak");
        fs::write(target_dir.join("skip.ak"), "validator skip {}").expect("write ak in target");
        fs::write(tx3_dir.join("skip2.ak"), "validator skip2 {}").expect("write ak in tx3");
        fs::write(build_dir.join("skip3.ak"), "validator skip3 {}").expect("write ak in build");

        let files = discover_source_files(root, &[]).expect("should discover files");

        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("contracts/ok.ak"));
    }

    #[test]
    fn discover_source_files_includes_tests_by_default() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        let validators = root.join("validators");

        fs::create_dir_all(&validators).expect("create validators");
        fs::write(validators.join("spend.ak"), "validator spend {}").expect("write source");
        fs::write(validators.join("spend.tests.ak"), "test spend {}").expect("write test");

        let files = discover_source_files(root, &[]).expect("should discover files");

        assert_eq!(files.len(), 2);
    }

    #[test]
    fn discover_source_files_excludes_tests_when_pattern_applied() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        let validators = root.join("validators");

        fs::create_dir_all(&validators).expect("create validators");
        fs::write(validators.join("spend.ak"), "validator spend {}").expect("write source");
        fs::write(validators.join("spend.tests.ak"), "test spend {}").expect("write test");

        let files = discover_source_files(root, &build_exclude_patterns(true))
            .expect("should discover files");

        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("validators/spend.ak"));
    }

    #[test]
    fn build_exclude_patterns_maps_the_exclude_tests_flag() {
        assert!(build_exclude_patterns(false).is_empty());
        assert_eq!(build_exclude_patterns(true), vec!["*.tests.ak".to_string()]);
    }

    #[test]
    fn render_findings_markdown_includes_location_when_available() {
        let findings = vec![VulnerabilityFinding {
            title: "Strict equality on value".to_string(),
            severity: "high".to_string(),
            summary: "Potential bypass due to strict value equality".to_string(),
            evidence: vec!["validators/spend.ak:42".to_string()],
            recommendation: "Compare lovelace and assets separately".to_string(),
            file: Some("validators/spend.ak".to_string()),
            line: Some(42),
        }];

        let markdown = render_findings_markdown(&findings);

        assert!(markdown.contains("Location: validators/spend.ak:42"));
    }

    fn skill_named(id: &str) -> VulnerabilitySkill {
        VulnerabilitySkill {
            id: id.to_string(),
            name: id.to_string(),
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

    #[test]
    fn severity_rank_orders_critical_before_optimizations() {
        assert!(severity_rank("critical") < severity_rank("high"));
        assert!(severity_rank("high") < severity_rank("medium"));
        assert!(severity_rank("LOW") < severity_rank("unknown"));
    }

    #[test]
    fn sorts_skills_by_severity_then_id() {
        let mut skills = vec![
            skill_named("004"),
            skill_named("critical"),
            skill_named("003"),
        ];
        skills[0].severity = "low".to_string();
        skills[1].severity = "critical".to_string();
        skills[2].severity = "medium".to_string();

        sort_skills_by_severity(&mut skills);

        let ids = skills
            .iter()
            .map(|skill| skill.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["critical", "003", "004"]);
    }

    fn permission_spec(root: &Path) -> PermissionPromptSpec {
        PermissionPromptSpec {
            shell: "bash".to_string(),
            allowed_commands: vec![],
            scope_rules: vec![],
            workspace_root: root.display().to_string(),
            read_scope: "workspace".to_string(),
            interactive_permissions: false,
            allowed_paths: vec![],
        }
    }

    fn empty_state() -> AnalysisStateJson {
        AnalysisStateJson {
            version: "2".to_string(),
            source_files: Vec::new(),
            excluded_patterns: Vec::new(),
            provider: crate::model::ProviderSpec {
                name: "test".to_string(),
                model: None,
                notes: String::new(),
            },
            permission_prompt: PermissionPromptSpec {
                shell: "bash".to_string(),
                allowed_commands: vec![],
                scope_rules: vec![],
                workspace_root: String::new(),
                read_scope: "workspace".to_string(),
                interactive_permissions: false,
                allowed_paths: vec![],
            },
            ast: None,
            validator_context: ValidatorContextMap::default(),
            iterations: Vec::new(),
            total_token_usage: crate::model::TokenUsage::default(),
            total_duration_ms: 0,
        }
    }

    #[test]
    fn skill_loop_records_a_failure_and_continues_to_the_next_skill() {
        // A rate limit that blocks skill 2 must not discard skills 3 and 4.
        struct FlakyProvider {
            calls: std::cell::RefCell<usize>,
        }

        impl AnalysisProvider for FlakyProvider {
            fn provider_spec(&self) -> crate::model::ProviderSpec {
                crate::model::ProviderSpec {
                    name: "test".to_string(),
                    model: None,
                    notes: String::new(),
                }
            }

            fn analyze_skill(
                &self,
                skill: &VulnerabilitySkill,
                _prompt: &MiniPrompt,
                _source_references: &[String],
                _validator_context: &ValidatorContextMap,
                _project_root: &Path,
                _permission_prompt: &PermissionPromptSpec,
            ) -> Result<SkillIterationResult> {
                *self.calls.borrow_mut() += 1;
                if skill.id == "b" {
                    return Err(miette::miette!("429 Too Many Requests"));
                }
                Ok(SkillIterationResult {
                    skill_id: skill.id.clone(),
                    status: "completed".to_string(),
                    findings: Vec::new(),
                    next_prompt: None,
                    skill_fingerprint: None,
                    provider: None,
                    duration_ms: None,
                    analysis_summary: None,
                    error: None,
                    token_usage: crate::model::TokenUsage::default(),
                })
            }
        }

        let temp = tempfile::tempdir().expect("temp dir");
        let state_out = temp.path().join("state.json");
        let provider = FlakyProvider {
            calls: std::cell::RefCell::new(0),
        };
        let skills: Vec<VulnerabilitySkill> =
            ["a", "b", "c"].iter().map(|id| skill_named(id)).collect();
        let mut state = empty_state();

        run_skill_loop(
            &skills,
            &mut state,
            SkillLoopContext {
                provider: &provider,
                source_files: &[],
                validator_context: &ValidatorContextMap::default(),
                project_root: temp.path(),
                permission_prompt: &permission_spec(temp.path()),
                ai_logs: LogLevel::Off,
                state_out: &state_out,
                resume_strict: false,
            },
        )
        .expect("one failing skill should not abort the run");

        assert_eq!(
            *provider.calls.borrow(),
            3,
            "every skill should be attempted once"
        );
        assert_eq!(state.iterations.len(), 3);
        assert_eq!(state.iterations[0].status, "completed");
        assert_eq!(state.iterations[1].status, "failed");
        assert!(
            state.iterations[1]
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("429")
        );
        assert_eq!(state.iterations[2].status, "completed");
    }

    #[test]
    fn skill_loop_stops_after_repeated_consecutive_failures() {
        struct DeadProvider;

        impl AnalysisProvider for DeadProvider {
            fn provider_spec(&self) -> crate::model::ProviderSpec {
                crate::model::ProviderSpec {
                    name: "test".to_string(),
                    model: None,
                    notes: String::new(),
                }
            }

            fn analyze_skill(
                &self,
                _skill: &VulnerabilitySkill,
                _prompt: &MiniPrompt,
                _source_references: &[String],
                _validator_context: &ValidatorContextMap,
                _project_root: &Path,
                _permission_prompt: &PermissionPromptSpec,
            ) -> Result<SkillIterationResult> {
                Err(miette::miette!("credit balance is too low"))
            }
        }

        let temp = tempfile::tempdir().expect("temp dir");
        let state_out = temp.path().join("state.json");
        let skills: Vec<VulnerabilitySkill> = ["a", "b", "c", "d", "e"]
            .iter()
            .map(|id| skill_named(id))
            .collect();
        let mut state = empty_state();

        let error = run_skill_loop(
            &skills,
            &mut state,
            SkillLoopContext {
                provider: &DeadProvider,
                source_files: &[],
                validator_context: &ValidatorContextMap::default(),
                project_root: temp.path(),
                permission_prompt: &permission_spec(temp.path()),
                ai_logs: LogLevel::Off,
                state_out: &state_out,
                resume_strict: false,
            },
        )
        .expect_err("a dead account should stop the run");

        assert!(error.to_string().contains("failed in a row"));
        assert_eq!(
            state.iterations.len(),
            MAX_CONSECUTIVE_SKILL_FAILURES,
            "the aborting failure must still be recorded"
        );
    }

    #[test]
    fn skill_loop_stops_entire_audit_when_user_declines_more_steps() {
        struct StoppedProvider {
            calls: std::cell::RefCell<usize>,
        }

        impl AnalysisProvider for StoppedProvider {
            fn provider_spec(&self) -> crate::model::ProviderSpec {
                crate::model::ProviderSpec {
                    name: "test".to_string(),
                    model: None,
                    notes: String::new(),
                }
            }

            fn analyze_skill(
                &self,
                _skill: &VulnerabilitySkill,
                _prompt: &MiniPrompt,
                _source_references: &[String],
                _validator_context: &ValidatorContextMap,
                _project_root: &Path,
                _permission_prompt: &PermissionPromptSpec,
            ) -> Result<SkillIterationResult> {
                *self.calls.borrow_mut() += 1;
                Err(miette::Report::new(UserStopped))
            }
        }

        let temp = tempfile::tempdir().expect("temp dir");
        let state_out = temp.path().join("state.json");
        let skills: Vec<VulnerabilitySkill> =
            ["a", "b", "c"].iter().map(|id| skill_named(id)).collect();
        let mut state = empty_state();
        let provider = StoppedProvider {
            calls: std::cell::RefCell::new(0),
        };

        let error = run_skill_loop(
            &skills,
            &mut state,
            SkillLoopContext {
                provider: &provider,
                source_files: &[],
                validator_context: &ValidatorContextMap::default(),
                project_root: temp.path(),
                permission_prompt: &permission_spec(temp.path()),
                ai_logs: LogLevel::Off,
                state_out: &state_out,
                resume_strict: false,
            },
        )
        .expect_err("declining more steps should stop the audit");

        assert!(error.to_string().contains("stopped by user"));
        assert_eq!(*provider.calls.borrow(), 1, "later skills must not run");
        assert_eq!(state.iterations.len(), 1);
        assert_eq!(state.iterations[0].status, "failed");
        assert!(
            state.iterations[0]
                .error
                .as_deref()
                .is_some_and(|error| error.contains("stopped by user"))
        );
    }

    fn ast_with_source_fingerprint(source_fingerprint: &str) -> AstMetadata {
        AstMetadata {
            path: ".tx3/audit/aiken-ast.json".to_string(),
            fingerprint: "sha256:snapshot".to_string(),
            source_fingerprint: source_fingerprint.to_string(),
            generated_at: "1970-01-01T00:00:00Z".to_string(),
            tool: crate::model::AstToolMetadata {
                name: "aiken".to_string(),
                version: "test".to_string(),
            },
        }
    }

    fn iteration_with_fingerprint(
        skill_id: &str,
        status: &str,
        fingerprint: &str,
        provider_name: Option<&str>,
        duration_ms: Option<u64>,
    ) -> SkillIterationResult {
        SkillIterationResult {
            skill_id: skill_id.to_string(),
            status: status.to_string(),
            findings: Vec::new(),
            next_prompt: None,
            skill_fingerprint: Some(fingerprint.to_string()),
            provider: provider_name.map(|name| ProviderSpec {
                name: name.to_string(),
                model: None,
                notes: String::new(),
            }),
            duration_ms,
            analysis_summary: None,
            error: None,
            token_usage: crate::model::TokenUsage::default(),
        }
    }

    fn completed_result(skill: &VulnerabilitySkill) -> SkillIterationResult {
        SkillIterationResult {
            skill_id: skill.id.clone(),
            status: "completed".to_string(),
            findings: Vec::new(),
            next_prompt: None,
            skill_fingerprint: None,
            provider: None,
            duration_ms: None,
            analysis_summary: None,
            error: None,
            token_usage: crate::model::TokenUsage::default(),
        }
    }

    #[test]
    fn load_resume_state_returns_none_when_state_file_is_missing() {
        let temp = tempfile::tempdir().expect("temp dir");
        let missing = temp.path().join("missing.json");
        let skills = vec![skill_named("a")];
        let ast = ast_with_source_fingerprint("sha256:src");

        let loaded = load_resume_state(&missing, &ast, &skills)
            .expect("a missing state file should not be an error");

        assert!(loaded.is_none());
    }

    #[test]
    fn load_resume_state_rejects_version_one() {
        let temp = tempfile::tempdir().expect("temp dir");
        let state_out = temp.path().join("state.json");
        let mut previous = empty_state();
        previous.version = "1".to_string();
        previous.ast = Some(ast_with_source_fingerprint("sha256:src"));
        std::fs::write(
            &state_out,
            serde_json::to_string(&previous).expect("serialize state"),
        )
        .expect("write state");

        let error = load_resume_state(
            &state_out,
            &ast_with_source_fingerprint("sha256:src"),
            &[skill_named("a")],
        )
        .expect_err("version 1 must be rejected");

        assert!(error.to_string().contains("unsupported version '1'"));
    }

    #[test]
    fn load_resume_state_rejects_changed_sources() {
        let temp = tempfile::tempdir().expect("temp dir");
        let state_out = temp.path().join("state.json");
        let mut previous = empty_state();
        previous.ast = Some(ast_with_source_fingerprint("sha256:old"));
        std::fs::write(
            &state_out,
            serde_json::to_string(&previous).expect("serialize state"),
        )
        .expect("write state");

        let error = load_resume_state(
            &state_out,
            &ast_with_source_fingerprint("sha256:new"),
            &[skill_named("a")],
        )
        .expect_err("changed sources must be rejected");

        assert!(error.to_string().contains("Sources changed"));
    }

    #[test]
    fn load_resume_state_keeps_one_iteration_per_current_skill() {
        let temp = tempfile::tempdir().expect("temp dir");
        let state_out = temp.path().join("state.json");
        let mut previous = empty_state();
        previous.ast = Some(ast_with_source_fingerprint("sha256:src"));
        previous.iterations = vec![
            iteration_with_fingerprint("a", "completed", "sha256:old-a", Some("test"), Some(10)),
            iteration_with_fingerprint(
                "removed",
                "completed",
                "sha256:removed",
                Some("test"),
                Some(10),
            ),
            iteration_with_fingerprint("a", "completed", "sha256:new-a", Some("test"), Some(20)),
        ];
        previous.total_token_usage = crate::model::TokenUsage {
            input_tokens: 42,
            ..Default::default()
        };
        previous.total_duration_ms = 30;
        std::fs::write(
            &state_out,
            serde_json::to_string(&previous).expect("serialize state"),
        )
        .expect("write state");

        let loaded = load_resume_state(
            &state_out,
            &ast_with_source_fingerprint("sha256:src"),
            &[skill_named("a"), skill_named("b")],
        )
        .expect("resume should parse")
        .expect("state should exist");

        assert_eq!(loaded.iterations.len(), 1);
        assert_eq!(loaded.iterations[0].skill_id, "a");
        assert_eq!(
            loaded.iterations[0].skill_fingerprint.as_deref(),
            Some("sha256:new-a"),
            "the latest attempt should win"
        );
        assert_eq!(loaded.total_token_usage.input_tokens, 42);
        assert_eq!(loaded.total_duration_ms, 30);
    }

    #[test]
    fn skill_loop_skips_matching_completed_skills_and_reruns_changed_ones() {
        struct CountingProvider {
            calls: std::cell::RefCell<Vec<String>>,
        }

        impl AnalysisProvider for CountingProvider {
            fn provider_spec(&self) -> ProviderSpec {
                ProviderSpec {
                    name: "test".to_string(),
                    model: None,
                    notes: String::new(),
                }
            }

            fn analyze_skill(
                &self,
                skill: &VulnerabilitySkill,
                _prompt: &MiniPrompt,
                _source_references: &[String],
                _validator_context: &ValidatorContextMap,
                _project_root: &Path,
                _permission_prompt: &PermissionPromptSpec,
            ) -> Result<SkillIterationResult> {
                self.calls.borrow_mut().push(skill.id.clone());
                Ok(completed_result(skill))
            }
        }

        let temp = tempfile::tempdir().expect("temp dir");
        let state_out = temp.path().join("state.json");
        let skills: Vec<VulnerabilitySkill> =
            ["a", "b", "c"].iter().map(|id| skill_named(id)).collect();
        let fingerprint_a = prompt_fingerprint(&build_mini_prompt(&skills[0]));
        let stale_fingerprint = "sha256:stale".to_string();

        let mut state = empty_state();
        state.iterations = vec![
            iteration_with_fingerprint("a", "completed", &fingerprint_a, Some("test"), Some(5)),
            iteration_with_fingerprint("b", "completed", &stale_fingerprint, Some("test"), Some(5)),
        ];

        let provider = CountingProvider {
            calls: std::cell::RefCell::new(Vec::new()),
        };

        run_skill_loop(
            &skills,
            &mut state,
            SkillLoopContext {
                provider: &provider,
                source_files: &[],
                validator_context: &ValidatorContextMap::default(),
                project_root: temp.path(),
                permission_prompt: &permission_spec(temp.path()),
                ai_logs: LogLevel::Off,
                state_out: &state_out,
                resume_strict: false,
            },
        )
        .expect("loop should succeed");

        assert_eq!(
            *provider.calls.borrow(),
            vec!["b".to_string(), "c".to_string()],
            "only changed and missing skills should run"
        );
        assert_eq!(state.iterations.len(), 3);
        assert_eq!(state.iterations[0].skill_id, "a");
    }

    #[test]
    fn skill_loop_reruns_failed_skills_loaded_from_state() {
        struct AlwaysOkProvider {
            calls: std::cell::RefCell<usize>,
        }

        impl AnalysisProvider for AlwaysOkProvider {
            fn provider_spec(&self) -> ProviderSpec {
                ProviderSpec {
                    name: "test".to_string(),
                    model: None,
                    notes: String::new(),
                }
            }

            fn analyze_skill(
                &self,
                skill: &VulnerabilitySkill,
                _prompt: &MiniPrompt,
                _source_references: &[String],
                _validator_context: &ValidatorContextMap,
                _project_root: &Path,
                _permission_prompt: &PermissionPromptSpec,
            ) -> Result<SkillIterationResult> {
                *self.calls.borrow_mut() += 1;
                Ok(completed_result(skill))
            }
        }

        let temp = tempfile::tempdir().expect("temp dir");
        let state_out = temp.path().join("state.json");
        let skill = skill_named("a");
        let fingerprint = prompt_fingerprint(&build_mini_prompt(&skill));
        let mut state = empty_state();
        state.iterations = vec![iteration_with_fingerprint(
            "a",
            "failed",
            &fingerprint,
            Some("test"),
            Some(10),
        )];

        let provider = AlwaysOkProvider {
            calls: std::cell::RefCell::new(0),
        };

        run_skill_loop(
            std::slice::from_ref(&skill),
            &mut state,
            SkillLoopContext {
                provider: &provider,
                source_files: &[],
                validator_context: &ValidatorContextMap::default(),
                project_root: temp.path(),
                permission_prompt: &permission_spec(temp.path()),
                ai_logs: LogLevel::Off,
                state_out: &state_out,
                resume_strict: false,
            },
        )
        .expect("failed skill should be retried");

        assert_eq!(*provider.calls.borrow(), 1);
        assert_eq!(state.iterations.len(), 1);
        assert_eq!(state.iterations[0].status, "completed");
    }

    #[test]
    fn resume_strict_reruns_when_provider_changes() {
        struct CountingProvider {
            calls: std::cell::RefCell<usize>,
        }

        impl AnalysisProvider for CountingProvider {
            fn provider_spec(&self) -> ProviderSpec {
                ProviderSpec {
                    name: "test".to_string(),
                    model: None,
                    notes: String::new(),
                }
            }

            fn analyze_skill(
                &self,
                skill: &VulnerabilitySkill,
                _prompt: &MiniPrompt,
                _source_references: &[String],
                _validator_context: &ValidatorContextMap,
                _project_root: &Path,
                _permission_prompt: &PermissionPromptSpec,
            ) -> Result<SkillIterationResult> {
                *self.calls.borrow_mut() += 1;
                Ok(completed_result(skill))
            }
        }

        let temp = tempfile::tempdir().expect("temp dir");
        let state_out = temp.path().join("state.json");
        let skill = skill_named("a");
        let fingerprint = prompt_fingerprint(&build_mini_prompt(&skill));
        let mut state = empty_state();
        state.iterations = vec![iteration_with_fingerprint(
            "a",
            "completed",
            &fingerprint,
            Some("other-provider"),
            Some(10),
        )];

        let provider = CountingProvider {
            calls: std::cell::RefCell::new(0),
        };

        run_skill_loop(
            std::slice::from_ref(&skill),
            &mut state,
            SkillLoopContext {
                provider: &provider,
                source_files: &[],
                validator_context: &ValidatorContextMap::default(),
                project_root: temp.path(),
                permission_prompt: &permission_spec(temp.path()),
                ai_logs: LogLevel::Off,
                state_out: &state_out,
                resume_strict: true,
            },
        )
        .expect("strict resume should re-run");

        assert_eq!(*provider.calls.borrow(), 1);
    }

    #[test]
    fn resume_strict_reruns_when_provider_notes_change() {
        struct CountingProvider {
            calls: std::cell::RefCell<usize>,
        }

        impl AnalysisProvider for CountingProvider {
            fn provider_spec(&self) -> ProviderSpec {
                ProviderSpec {
                    name: "test".to_string(),
                    model: None,
                    notes: "current endpoint".to_string(),
                }
            }

            fn analyze_skill(
                &self,
                skill: &VulnerabilitySkill,
                _prompt: &MiniPrompt,
                _source_references: &[String],
                _validator_context: &ValidatorContextMap,
                _project_root: &Path,
                _permission_prompt: &PermissionPromptSpec,
            ) -> Result<SkillIterationResult> {
                *self.calls.borrow_mut() += 1;
                Ok(completed_result(skill))
            }
        }

        let temp = tempfile::tempdir().expect("temp dir");
        let state_out = temp.path().join("state.json");
        let skill = skill_named("a");
        let fingerprint = prompt_fingerprint(&build_mini_prompt(&skill));
        let mut state = empty_state();
        let mut previous =
            iteration_with_fingerprint("a", "completed", &fingerprint, Some("test"), Some(10));
        previous.provider = Some(ProviderSpec {
            name: "test".to_string(),
            model: None,
            notes: "old endpoint".to_string(),
        });
        state.iterations = vec![previous];

        let provider = CountingProvider {
            calls: std::cell::RefCell::new(0),
        };

        run_skill_loop(
            std::slice::from_ref(&skill),
            &mut state,
            SkillLoopContext {
                provider: &provider,
                source_files: &[],
                validator_context: &ValidatorContextMap::default(),
                project_root: temp.path(),
                permission_prompt: &permission_spec(temp.path()),
                ai_logs: LogLevel::Off,
                state_out: &state_out,
                resume_strict: true,
            },
        )
        .expect("strict resume should re-run on endpoint change");

        assert_eq!(*provider.calls.borrow(), 1);
    }

    #[test]
    fn resume_reuses_by_default_when_provider_changes() {
        struct CountingProvider {
            calls: std::cell::RefCell<usize>,
        }

        impl AnalysisProvider for CountingProvider {
            fn provider_spec(&self) -> ProviderSpec {
                ProviderSpec {
                    name: "test".to_string(),
                    model: None,
                    notes: String::new(),
                }
            }

            fn analyze_skill(
                &self,
                skill: &VulnerabilitySkill,
                _prompt: &MiniPrompt,
                _source_references: &[String],
                _validator_context: &ValidatorContextMap,
                _project_root: &Path,
                _permission_prompt: &PermissionPromptSpec,
            ) -> Result<SkillIterationResult> {
                *self.calls.borrow_mut() += 1;
                Ok(completed_result(skill))
            }
        }

        let temp = tempfile::tempdir().expect("temp dir");
        let state_out = temp.path().join("state.json");
        let skill = skill_named("a");
        let fingerprint = prompt_fingerprint(&build_mini_prompt(&skill));
        let mut state = empty_state();
        state.iterations = vec![iteration_with_fingerprint(
            "a",
            "completed",
            &fingerprint,
            Some("other-provider"),
            Some(10),
        )];

        let provider = CountingProvider {
            calls: std::cell::RefCell::new(0),
        };

        run_skill_loop(
            std::slice::from_ref(&skill),
            &mut state,
            SkillLoopContext {
                provider: &provider,
                source_files: &[],
                validator_context: &ValidatorContextMap::default(),
                project_root: temp.path(),
                permission_prompt: &permission_spec(temp.path()),
                ai_logs: LogLevel::Off,
                state_out: &state_out,
                resume_strict: false,
            },
        )
        .expect("default resume should reuse");

        assert_eq!(*provider.calls.borrow(), 0);
    }

    #[test]
    fn duration_is_recorded_per_skill_and_total() {
        struct OkProvider;

        impl AnalysisProvider for OkProvider {
            fn provider_spec(&self) -> ProviderSpec {
                ProviderSpec {
                    name: "test".to_string(),
                    model: None,
                    notes: String::new(),
                }
            }

            fn analyze_skill(
                &self,
                skill: &VulnerabilitySkill,
                _prompt: &MiniPrompt,
                _source_references: &[String],
                _validator_context: &ValidatorContextMap,
                _project_root: &Path,
                _permission_prompt: &PermissionPromptSpec,
            ) -> Result<SkillIterationResult> {
                Ok(completed_result(skill))
            }
        }

        let temp = tempfile::tempdir().expect("temp dir");
        let state_out = temp.path().join("state.json");
        let skills: Vec<VulnerabilitySkill> = ["a", "b"].iter().map(|id| skill_named(id)).collect();
        let mut state = empty_state();

        run_skill_loop(
            &skills,
            &mut state,
            SkillLoopContext {
                provider: &OkProvider,
                source_files: &[],
                validator_context: &ValidatorContextMap::default(),
                project_root: temp.path(),
                permission_prompt: &permission_spec(temp.path()),
                ai_logs: LogLevel::Off,
                state_out: &state_out,
                resume_strict: false,
            },
        )
        .expect("loop should succeed");

        let durations = state
            .iterations
            .iter()
            .map(|iteration| iteration.duration_ms.expect("duration recorded"))
            .collect::<Vec<u64>>();
        assert_eq!(durations.len(), 2);
        assert_eq!(state.total_duration_ms, durations.iter().sum::<u64>());
        assert!(
            state
                .iterations
                .iter()
                .all(|iteration| iteration.provider.is_some())
        );
    }

    #[test]
    fn prompt_fingerprint_includes_audit_context() {
        let prompt = build_mini_prompt(&skill_named("a"));

        assert_ne!(
            prompt_fingerprint_with_context(&prompt, "context-a"),
            prompt_fingerprint_with_context(&prompt, "context-b")
        );
    }

    #[test]
    fn report_coverage_lists_statuses_models_times_and_summaries() {
        let skills = vec![skill_named("a"), skill_named("b"), skill_named("c")];
        let mut completed =
            iteration_with_fingerprint("a", "completed", "sha256:a", Some("openai"), Some(1_500));
        completed.provider = Some(ProviderSpec {
            name: "openai".to_string(),
            model: Some("gpt-5".to_string()),
            notes: String::new(),
        });
        completed.analysis_summary = Some("checked the spend path".to_string());

        let failed =
            iteration_with_fingerprint("b", "failed", "sha256:b", Some("anthropic"), Some(2_500));

        let mut state = empty_state();
        state.source_files = vec!["validators/spend.ak".to_string()];
        state.iterations = vec![completed, failed];
        state.total_duration_ms = 4_000;
        state.total_token_usage = crate::model::TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        };

        let report = build_report(&state, &skills, false);

        assert!(report.coverage_markdown.contains("1 completed"));
        assert!(report.coverage_markdown.contains("1 failed"));
        assert!(report.coverage_markdown.contains("1 not run"));
        assert!(report.coverage_markdown.contains("openai (gpt-5)"));
        assert!(report.coverage_markdown.contains("1.5s"));
        assert!(report.coverage_markdown.contains("anthropic"));
        assert!(report.coverage_markdown.contains("2.5s"));
        assert!(report.coverage_markdown.contains("checked the spend path"));
        assert!(report.coverage_markdown.contains("Total time: 4.0s"));

        let markdown = render_report_markdown(&report);
        assert!(markdown.contains("## Coverage"));
        assert!(markdown.contains("1 not run"));
    }

    #[test]
    fn report_coverage_includes_failure_reason() {
        let skills = vec![skill_named("a")];
        let mut failed =
            iteration_with_fingerprint("a", "failed", "sha256:a", Some("test"), Some(10));
        failed.error = Some("429 Too Many Requests".to_string());

        let mut state = empty_state();
        state.iterations = vec![failed];

        let report = build_report(&state, &skills, false);

        assert!(
            report
                .coverage_markdown
                .contains("error: 429 Too Many Requests")
        );
    }

    #[test]
    fn report_coverage_marks_incomplete_audits() {
        let skills = vec![skill_named("a"), skill_named("b")];
        let mut state = empty_state();
        state.iterations = vec![iteration_with_fingerprint(
            "a",
            "failed",
            "sha256:a",
            Some("test"),
            Some(10),
        )];

        let report = build_report(&state, &skills, true);

        assert!(report.coverage_markdown.contains("Status: INCOMPLETE"));
        assert!(report.coverage_markdown.contains("stopped early"));
        assert!(report.coverage_markdown.contains("1 failed"));
        assert!(report.coverage_markdown.contains("1 not run"));
    }

    #[test]
    fn report_coverage_marks_failed_skills_incomplete_without_abort() {
        let skills = vec![skill_named("a"), skill_named("b")];
        let mut state = empty_state();
        state.iterations = vec![
            iteration_with_fingerprint("a", "failed", "sha256:a", Some("test"), Some(10)),
            iteration_with_fingerprint("b", "completed", "sha256:b", Some("test"), Some(10)),
        ];

        let report = build_report(&state, &skills, false);

        assert!(report.coverage_markdown.contains("Status: INCOMPLETE"));
        assert!(report.coverage_markdown.contains("1 skill(s) failed"));
        assert!(!report.coverage_markdown.contains("stopped early"));
    }

    #[test]
    fn report_coverage_marks_not_run_skills_incomplete_without_abort() {
        let skills = vec![skill_named("a"), skill_named("b")];
        let mut state = empty_state();
        state.iterations = vec![iteration_with_fingerprint(
            "a",
            "completed",
            "sha256:a",
            Some("test"),
            Some(10),
        )];

        let report = build_report(&state, &skills, false);

        assert!(report.coverage_markdown.contains("Status: INCOMPLETE"));
        assert!(report.coverage_markdown.contains("1 skill(s) not run"));
        assert!(!report.coverage_markdown.contains("stopped early"));
    }

    #[test]
    fn report_coverage_omits_banner_when_complete() {
        let skills = vec![skill_named("a"), skill_named("b")];
        let mut state = empty_state();
        state.iterations = vec![
            iteration_with_fingerprint("a", "completed", "sha256:a", Some("test"), Some(10)),
            iteration_with_fingerprint("b", "completed", "sha256:b", Some("test"), Some(10)),
        ];

        let report = build_report(&state, &skills, false);

        assert!(!report.coverage_markdown.contains("Status: INCOMPLETE"));
    }

    #[test]
    fn report_coverage_lists_excluded_patterns() {
        let mut state = empty_state();
        state.excluded_patterns = vec!["*.tests.ak".to_string()];

        let report = build_report(&state, &[], false);

        assert!(
            report
                .coverage_markdown
                .contains("Excluded patterns: *.tests.ak")
        );
    }

    #[test]
    fn report_coverage_marks_unknown_provider_and_time() {
        let skills = vec![skill_named("a")];
        let mut state = empty_state();
        state.iterations = vec![iteration_with_fingerprint(
            "a",
            "completed",
            "sha256:a",
            None,
            None,
        )];

        let report = build_report(&state, &skills, false);

        assert!(report.coverage_markdown.contains("provider unknown"));
        assert!(report.coverage_markdown.contains("time unknown"));
    }

    #[test]
    fn duration_formatting_uses_readable_units() {
        assert_eq!(format_duration_ms(500), "500ms");
        assert_eq!(format_duration_ms(1_500), "1.5s");
        assert_eq!(format_duration_ms(65_000), "1m 5s");
    }

    #[test]
    fn truncate_error_text_keeps_both_ends() {
        let error = format!("{}{}{}", "start ".repeat(100), "x".repeat(1000), " cause");
        let truncated = truncate_error_text(&error);

        assert!(truncated.starts_with("start "));
        assert!(truncated.contains("omitted"));
        assert!(truncated.ends_with("cause"));
        assert!(truncated.len() < error.len());
    }

    #[test]
    fn truncate_error_text_handles_multibyte_input() {
        let error = format!("{}{}", "é".repeat(1000), " done");
        let truncated = truncate_error_text(&error);

        assert!(truncated.contains("omitted"));
        assert!(truncated.ends_with(" done"));
    }

    #[test]
    fn truncate_error_text_leaves_short_errors_unchanged() {
        assert_eq!(
            truncate_error_text("429 Too Many Requests"),
            "429 Too Many Requests"
        );
    }
}
