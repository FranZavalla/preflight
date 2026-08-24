use miette::{Context, IntoDiagnostic, Result};
use serde_json::{Value, json};
use std::path::Path;
use std::time::Duration;

use super::AnalysisProvider;
use super::shared::{
    ContentStreamState, ReasoningStreamState, block_on_runtime_aware, build_agent_system_prompt,
    build_initial_user_prompt, emit_reasoning_double_line_break, emit_reasoning_line_break,
    finalize_content_stdout, finalize_reasoning_stdout, log_agent_progress, run_agent_loop,
    stream_content_delta_to_stdout, stream_reasoning_delta_to_stdout,
};
use crate::model::{
    MiniPrompt, PermissionPromptSpec, ProviderSpec, SkillIterationResult, TokenUsage,
    ValidatorContextMap, VulnerabilitySkill,
};

const DEFAULT_MAX_TOKENS: u32 = 4000;
const THINKING_MAX_TOKENS: u32 = 1600;
const THINKING_BUDGET_TOKENS: u32 = 1024;
// Thinking tokens count against `max_tokens`, and adaptive thinking has no budget
// of its own: the model reasons until it is done or until `max_tokens` runs out,
// whichever comes first. Too small a ceiling and the entire budget goes to the
// reasoning pass - the turn ends with `stop_reason: "max_tokens"` and the response
// carries a thinking block and no text at all.
//
// A streamed request can afford the headroom. A non-streamed one cannot: the API
// rejects a large `max_tokens` outright without `stream`, so that path keeps the
// tighter ceiling and leans on the no-thinking fallback variant when it is not
// enough.
const ADAPTIVE_THINKING_MAX_TOKENS: u32 = 64000;
const ADAPTIVE_THINKING_NON_STREAM_MAX_TOKENS: u32 = 16000;
/// Attempts per payload variant when the API asks us to slow down.
///
/// Input-tokens-per-minute ceilings are the binding limit here, and cache reads
/// count against them: a late-stage step can read hundreds of thousands of cached
/// tokens, so the account can stay saturated for minutes at a time. The ladder
/// runs 15s -> 30s -> 60s -> 120s -> 240s -> 300s, well past a one-minute window.
const MAX_RATE_LIMIT_RETRIES: usize = 6;
const RATE_LIMIT_BASE_DELAY: Duration = Duration::from_secs(15);
const RATE_LIMIT_MAX_DELAY: Duration = Duration::from_secs(300);

#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub version: String,
    pub ai_logs: bool,
    pub reasoning_effort: Option<String>,
}

/// How a model accepts extended thinking.
///
/// The on-modes are not interchangeable and the wrong one is a hard 400, so the
/// fallback ladder has to be built per model family instead of tried blind:
/// `enabled`/`budget_tokens` is rejected on Sonnet 5 and the Opus 5 line, and
/// `adaptive` is rejected before 4.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingSupport {
    /// Thinking is always on; it can be neither disabled nor budgeted.
    AlwaysAdaptive,
    /// `adaptive` is the only on-mode. Omitting `thinking` still thinks on these,
    /// so the no-thinking fallback has to say `disabled` out loud to mean it.
    AdaptiveOnly,
    /// `adaptive` works, and the legacy fixed budget still works behind it.
    AdaptiveOrLegacy,
    /// Pre-4.6: `adaptive` is rejected and a fixed budget is the only on-mode.
    LegacyOnly,
}

fn thinking_support(model: &str) -> ThinkingSupport {
    let model = model.to_ascii_lowercase();

    if model.contains("fable") || model.contains("mythos") {
        return ThinkingSupport::AlwaysAdaptive;
    }

    if ["opus-5", "sonnet-5", "opus-4-8", "opus-4-7"]
        .iter()
        .any(|family| model.contains(family))
    {
        return ThinkingSupport::AdaptiveOnly;
    }

    if model.contains("opus-4-6") || model.contains("sonnet-4-6") {
        return ThinkingSupport::AdaptiveOrLegacy;
    }

    if model.contains("haiku") || model.contains("-4-5") || model.contains("-3-") {
        return ThinkingSupport::LegacyOnly;
    }

    // Unknown model: keep the historical try-everything ladder.
    ThinkingSupport::AdaptiveOrLegacy
}

fn build_anthropic_payload_variants(
    model: &str,
    system_prompt: &str,
    messages: &[Value],
    stream: bool,
    reasoning_effort: Option<&str>,
) -> Vec<Value> {
    let mut normalized_messages = normalize_anthropic_messages(messages);
    apply_cache_breakpoints(&mut normalized_messages);

    let mut base = json!({
        "model": model,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "system": system_prompt,
        "messages": normalized_messages,
    });

    if stream {
        base["stream"] = Value::Bool(true);
    }

    let adaptive_max_tokens = if stream {
        ADAPTIVE_THINKING_MAX_TOKENS
    } else {
        ADAPTIVE_THINKING_NON_STREAM_MAX_TOKENS
    };

    // `--reasoning-effort` only decides whether we also pin `output_config.effort`.
    let adaptive = |effort: Option<&str>| {
        let mut payload = base.clone();
        payload["max_tokens"] = Value::from(adaptive_max_tokens);
        payload["thinking"] = json!({
            "type": "adaptive",
            "display": "summarized"
        });
        if let Some(effort) = effort {
            payload["output_config"] = json!({
                "effort": effort
            });
        }
        payload
    };

    // Pre-4.6 models still need the legacy fixed budget.
    let legacy = || {
        let mut payload = base.clone();
        payload["max_tokens"] = Value::from(THINKING_MAX_TOKENS);
        payload["thinking"] = json!({
            "type": "enabled",
            "budget_tokens": THINKING_BUDGET_TOKENS
        });
        payload
    };

    // The last-resort variant: no reasoning pass at all, so the whole (small)
    // budget is available for the JSON action.
    //
    // `output_config.effort` is deliberately left off. On the Opus 5 line
    // `thinking: disabled` is only accepted at effort `high` or below, and
    // omitting the field leaves it at the API default of `high`.
    let no_thinking = |explicit: bool| {
        let mut payload = base.clone();
        if explicit {
            payload["thinking"] = json!({
                "type": "disabled"
            });
        }
        payload
    };

    match thinking_support(model) {
        ThinkingSupport::AlwaysAdaptive => {
            let mut variants = vec![adaptive(reasoning_effort)];
            // Nothing can be turned off, so the only retry left is a cheaper pass.
            if reasoning_effort != Some("low") {
                variants.push(adaptive(Some("low")));
            }
            variants
        }
        ThinkingSupport::AdaptiveOnly => vec![adaptive(reasoning_effort), no_thinking(true)],
        ThinkingSupport::AdaptiveOrLegacy => {
            vec![adaptive(reasoning_effort), legacy(), no_thinking(false)]
        }
        ThinkingSupport::LegacyOnly => vec![legacy(), no_thinking(false)],
    }
}

/// Builds the failure for a 200 response that carried no text block.
///
/// The usual cause is not a malformed payload but an exhausted budget: adaptive
/// thinking spent every one of `max_tokens` on the reasoning pass, so the turn
/// ended with a thinking block and nothing after it. Saying so keeps the caller
/// from reading a budgeting problem as a transport bug.
fn empty_output_failure(
    payload: &Value,
    stop_reason: Option<&str>,
    saw_thinking: bool,
    generic_message: &str,
) -> AttemptFailure {
    let budget = payload
        .get("max_tokens")
        .and_then(Value::as_u64)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unset".to_string());

    if stop_reason == Some("max_tokens") {
        return if saw_thinking {
            miette::miette!(
                "the model spent its entire max_tokens budget ({}) on extended thinking and \
                 never emitted a text block; raise max_tokens or lower --reasoning-effort",
                budget
            )
        } else {
            miette::miette!(
                "the response hit max_tokens ({}) before producing any text",
                budget
            )
        }
        .into();
    }

    if saw_thinking {
        return miette::miette!(
            "the response contained a thinking block and no text (stop_reason: {})",
            stop_reason.unwrap_or("unreported")
        )
        .into();
    }

    miette::miette!("{}", generic_message).into()
}

/// Marks the stable prefix and the rolling tail of the conversation as cacheable.
///
/// The agent loop replays the whole transcript on every step, so without this the
/// large initial prompt (skill text + validator context map) plus every prior tool
/// result is re-billed at full input price each step.
///
/// Two breakpoints, both well inside Anthropic's limit of four:
/// - `messages[0]` caches everything before it, the system prompt included. The
///   system prompt alone is under the ~1024-token minimum, so it is deliberately
///   not given its own breakpoint.
/// - the final message caches the transcript accumulated so far, so each step
///   reads the previous step's writes instead of resending them.
fn apply_cache_breakpoints(messages: &mut [Value]) {
    if messages.is_empty() {
        return;
    }

    let last_index = messages.len() - 1;
    let mut breakpoints = vec![0usize];
    if last_index != 0 {
        breakpoints.push(last_index);
    }

    for index in breakpoints {
        if let Some(blocks) = messages[index]
            .get_mut("content")
            .and_then(Value::as_array_mut)
            && let Some(block) = blocks.last_mut()
            && let Some(object) = block.as_object_mut()
        {
            object.insert("cache_control".to_string(), json!({ "type": "ephemeral" }));
        }
    }
}

fn normalize_anthropic_messages(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .map(|message| {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user")
                .to_ascii_lowercase();

            let role = if role == "assistant" {
                "assistant"
            } else {
                "user"
            };

            let content = normalize_anthropic_message_content(message.get("content"));

            json!({
                "role": role,
                "content": content,
            })
        })
        .collect()
}

fn normalize_anthropic_message_content(content: Option<&Value>) -> Value {
    let Some(content) = content else {
        return json!([
            {
                "type": "text",
                "text": ""
            }
        ]);
    };

    if let Some(text) = content.as_str() {
        return json!([
            {
                "type": "text",
                "text": text
            }
        ]);
    }

    if let Some(items) = content.as_array() {
        let normalized_items = items
            .iter()
            .map(|item| {
                if item.get("type").and_then(Value::as_str).is_some() {
                    return item.clone();
                }

                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    return json!({
                        "type": "text",
                        "text": text,
                    });
                }

                json!({
                    "type": "text",
                    "text": item.to_string(),
                })
            })
            .collect::<Vec<Value>>();

        return Value::Array(normalized_items);
    }

    json!([
        {
            "type": "text",
            "text": content.to_string()
        }
    ])
}

fn maybe_emit_reasoning_line_break_on_summary_change(
    enabled: bool,
    state: &mut ReasoningStreamState,
    summary_index: Option<i64>,
) {
    let Some(current_index) = summary_index else {
        return;
    };

    if let Some(previous_index) = state.last_summary_index
        && previous_index != current_index
    {
        emit_reasoning_double_line_break(enabled, state);
    }

    state.last_summary_index = Some(current_index);
}

fn extract_anthropic_reasoning_delta(event: &Value) -> Option<String> {
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();

    match event_type.as_str() {
        "content_block_start" => {
            let block_type = event
                .pointer("/content_block/type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_ascii_lowercase();

            if block_type.contains("thinking") || block_type.contains("reasoning") {
                return event
                    .pointer("/content_block/thinking")
                    .and_then(Value::as_str)
                    .or_else(|| event.pointer("/content_block/text").and_then(Value::as_str))
                    .map(ToString::to_string);
            }

            None
        }
        "content_block_delta" => {
            let delta_type = event
                .pointer("/delta/type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_ascii_lowercase();

            if delta_type.contains("thinking") || delta_type.contains("reasoning") {
                return event
                    .pointer("/delta/thinking")
                    .and_then(Value::as_str)
                    .or_else(|| event.pointer("/delta/text").and_then(Value::as_str))
                    .map(ToString::to_string);
            }

            None
        }
        _ => None,
    }
}

fn extract_anthropic_reasoning_index(event: &Value) -> Option<i64> {
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();

    if event_type == "content_block_start" || event_type == "content_block_delta" {
        return event.get("index").and_then(Value::as_i64);
    }

    None
}

fn extract_anthropic_content_delta(event: &Value) -> Option<String> {
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();

    match event_type.as_str() {
        "content_block_start" => {
            let block_type = event
                .pointer("/content_block/type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_ascii_lowercase();

            if block_type == "text" {
                return event
                    .pointer("/content_block/text")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
            }

            None
        }
        "content_block_delta" => {
            let delta_type = event
                .pointer("/delta/type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_ascii_lowercase();

            if delta_type == "text_delta" {
                return event
                    .pointer("/delta/text")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
            }

            None
        }
        _ => None,
    }
}

fn extract_anthropic_response_usage(response_json: &Value) -> TokenUsage {
    let Some(usage) = response_json.get("usage") else {
        return TokenUsage::default();
    };
    read_anthropic_usage_object(usage)
}

fn update_anthropic_usage_from_event(event: &Value, usage: &mut TokenUsage) {
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();

    match event_type {
        "message_start" => {
            if let Some(usage_obj) = event.pointer("/message/usage") {
                let parsed = read_anthropic_usage_object(usage_obj);
                usage.input_tokens = parsed.input_tokens;
                usage.output_tokens = parsed.output_tokens;
                if parsed.cache_read_input_tokens.is_some() {
                    usage.cache_read_input_tokens = parsed.cache_read_input_tokens;
                }
                if parsed.cache_creation_input_tokens.is_some() {
                    usage.cache_creation_input_tokens = parsed.cache_creation_input_tokens;
                }
            }
        }
        "message_delta" => {
            // Anthropic sends cumulative output_tokens in message_delta usage. Replace, don't add.
            if let Some(value) = event
                .pointer("/usage/output_tokens")
                .and_then(Value::as_u64)
            {
                usage.output_tokens = value;
            }
        }
        _ => {}
    }
}

fn read_anthropic_usage_object(usage_obj: &Value) -> TokenUsage {
    TokenUsage {
        input_tokens: usage_obj
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        output_tokens: usage_obj
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        cache_read_input_tokens: usage_obj
            .get("cache_read_input_tokens")
            .and_then(Value::as_u64),
        cache_creation_input_tokens: usage_obj
            .get("cache_creation_input_tokens")
            .and_then(Value::as_u64),
        reasoning_tokens: None,
    }
}

fn extract_anthropic_non_stream_content(response_json: &Value) -> Option<String> {
    let mut text_chunks = Vec::new();

    if let Some(blocks) = response_json.get("content").and_then(Value::as_array) {
        for block in blocks {
            let block_type = block
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();

            if block_type == "text"
                && let Some(text) = block.get("text").and_then(Value::as_str)
                && !text.trim().is_empty()
            {
                text_chunks.push(text.to_string());
            }
        }
    }

    if text_chunks.is_empty() {
        response_json
            .pointer("/content/0/text")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(ToString::to_string)
    } else {
        Some(text_chunks.join(""))
    }
}

fn extract_anthropic_non_stream_reasoning(response_json: &Value) -> Option<String> {
    let mut reasoning_chunks = Vec::new();

    if let Some(blocks) = response_json.get("content").and_then(Value::as_array) {
        for block in blocks {
            let block_type = block
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_ascii_lowercase();

            if (block_type.contains("thinking") || block_type.contains("reasoning"))
                && let Some(text) = block
                    .get("thinking")
                    .and_then(Value::as_str)
                    .or_else(|| block.get("text").and_then(Value::as_str))
                && !text.trim().is_empty()
            {
                reasoning_chunks.push(text.to_string());
            }
        }
    }

    if reasoning_chunks.is_empty() {
        None
    } else {
        Some(reasoning_chunks.join("\n"))
    }
}

/// A failed request attempt, tagged with whether trying the next payload variant
/// could possibly help.
///
/// The variant fallback exists for *capability* rejections — a model refusing a
/// thinking or effort parameter. Account-level failures (exhausted credits, a bad
/// key, a blocked workspace) also surface as 400/401/403, but no payload shape
/// will fix them, so retrying six times only buries the real cause.
/// What a caller should do next after a failed attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureKind {
    /// Nothing can succeed: exhausted credits, bad key, blocked workspace.
    Fatal,
    /// The payload is fine; the account is sending too fast. Wait, resend the
    /// *same* payload — switching variants here would only thrash the cache.
    RateLimited,
    /// This payload shape was rejected; the next variant may work.
    NextVariant,
}

struct AttemptFailure {
    report: miette::Report,
    kind: FailureKind,
    retry_after: Option<Duration>,
}

impl AttemptFailure {
    fn is_fatal(&self) -> bool {
        self.kind == FailureKind::Fatal
    }
}

impl From<miette::Report> for AttemptFailure {
    fn from(report: miette::Report) -> Self {
        Self {
            report,
            kind: FailureKind::NextVariant,
            retry_after: None,
        }
    }
}

/// Pulls `error.message` out of an Anthropic error body, falling back to the raw
/// body when it is not the shape we expect.
fn api_error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .as_ref()
        .and_then(|parsed| parsed.pointer("/error/message"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| body.trim().to_string())
}

fn mentions_account_level_problem(text: &str) -> bool {
    let lowered = text.to_ascii_lowercase();
    ["credit balance", "billing", "authentication", "permission"]
        .iter()
        .any(|marker| lowered.contains(marker))
}

fn is_account_level_failure(status: reqwest::StatusCode, body: &str) -> bool {
    if matches!(status.as_u16(), 401 | 403) {
        return true;
    }

    if status.as_u16() != 400 {
        return false;
    }

    mentions_account_level_problem(body)
}

/// Classifies an `error` event delivered *inside* a 200 stream.
///
/// The balance can hit zero after the response headers are sent, so a healthy
/// HTTP status says nothing about whether the turn will finish.
fn classify_stream_error_event(event: &Value) -> AttemptFailure {
    let message = event
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or("stream terminated with an unspecified error");

    let error_type = event
        .pointer("/error/type")
        .and_then(Value::as_str)
        .unwrap_or_default();

    let kind = if mentions_account_level_problem(message) {
        FailureKind::Fatal
    } else if matches!(
        error_type,
        "rate_limit_error" | "overloaded_error" | "api_error"
    ) || message.to_ascii_lowercase().contains("rate limit")
    {
        FailureKind::RateLimited
    } else {
        FailureKind::NextVariant
    };

    AttemptFailure {
        report: miette::miette!("Anthropic API ended the stream: {}", message),
        kind,
        retry_after: None,
    }
}

fn classify_request_failure(
    label: &str,
    status: reqwest::StatusCode,
    body: &str,
    retry_after: Option<Duration>,
) -> AttemptFailure {
    if is_account_level_failure(status, body) {
        return AttemptFailure {
            report: miette::miette!(
                "Anthropic API rejected the request ({}): {}",
                status,
                api_error_message(body)
            ),
            kind: FailureKind::Fatal,
            retry_after: None,
        };
    }

    // 429 and 5xx are timing problems, not payload problems.
    if status.as_u16() == 429 || status.is_server_error() {
        return AttemptFailure {
            report: miette::miette!(
                "Anthropic API is throttling or unavailable ({}): {}",
                status,
                api_error_message(body)
            ),
            kind: FailureKind::RateLimited,
            retry_after,
        };
    }

    AttemptFailure {
        report: miette::miette!("{} failed with status {}: {}", label, status, body),
        kind: FailureKind::NextVariant,
        retry_after: None,
    }
}

/// Reads the `retry-after` header, which the API sends in seconds.
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

/// Runs one payload variant, waiting out rate limits instead of falling through
/// to a different variant.
///
/// Switching payload shape on a 429 is actively harmful: a changed `thinking`
/// config invalidates the cached prefix, so the retry resends the whole
/// transcript uncached and pushes the account *further* past its input-tokens
/// per minute ceiling.
async fn attempt_with_backoff<'a, F, Fut>(
    ai_logs: bool,
    label: &str,
    attempt_idx: usize,
    mut run: F,
) -> std::result::Result<(String, TokenUsage), AttemptFailure>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<(String, TokenUsage), AttemptFailure>>
        + 'a,
{
    let mut last = match run().await {
        Ok(value) => return Ok(value),
        Err(failure) => failure,
    };

    for retry in 0..MAX_RATE_LIMIT_RETRIES {
        if last.kind != FailureKind::RateLimited {
            return Err(last);
        }

        let delay = last
            .retry_after
            .unwrap_or_else(|| RATE_LIMIT_BASE_DELAY * 2u32.pow(retry.min(31) as u32))
            .min(RATE_LIMIT_MAX_DELAY);

        log_agent_progress(
            ai_logs,
            format!(
                "⏳ {} attempt {} rate limited; waiting {}s before retry {}/{}",
                label,
                attempt_idx + 1,
                delay.as_secs(),
                retry + 1,
                MAX_RATE_LIMIT_RETRIES
            ),
        );
        tokio::time::sleep(delay).await;

        last = match run().await {
            Ok(value) => return Ok(value),
            Err(failure) => failure,
        };
    }

    Err(last)
}

async fn stream_attempt(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    version: &str,
    payload: &Value,
    ai_logs: bool,
) -> std::result::Result<(String, TokenUsage), AttemptFailure> {
    let mut response = client
        .post(endpoint)
        .header("x-api-key", api_key)
        .header("anthropic-version", version)
        .json(payload)
        .send()
        .await
        .into_diagnostic()?;

    let status = response.status();
    if !status.is_success() {
        let retry_after = parse_retry_after(response.headers());
        let body = response.text().await.into_diagnostic()?;
        return Err(classify_request_failure(
            "Streaming request",
            status,
            &body,
            retry_after,
        ));
    }

    // Byte buffer, not String: a chunk boundary can land mid-UTF-8-sequence, and
    // decoding each partial chunk would replace the split character with U+FFFD.
    let mut pending: Vec<u8> = Vec::new();
    let mut model_output = String::new();
    let mut reasoning_stream_state = ReasoningStreamState::default();
    let mut content_stream_state = ContentStreamState::default();
    let mut usage = TokenUsage::default();
    let mut stream_error: Option<AttemptFailure> = None;
    let mut saw_message_stop = false;
    let mut saw_thinking = false;
    let mut stop_reason: Option<String> = None;

    'stream: while let Some(chunk) = response.chunk().await.into_diagnostic()? {
        pending.extend_from_slice(&chunk);

        while let Some(newline_index) = pending.iter().position(|&byte| byte == b'\n') {
            let line = String::from_utf8_lossy(&pending[..newline_index]).into_owned();
            pending.drain(..=newline_index);

            let line = line.trim_end_matches('\r').trim();
            if line.is_empty() || !line.starts_with("data:") {
                continue;
            }

            let event_data = line[5..].trim();
            if event_data == "[DONE]" {
                break;
            }

            let event: Value = match serde_json::from_str(event_data) {
                Ok(parsed) => parsed,
                Err(_) => continue,
            };

            match event
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
            {
                "error" => {
                    stream_error = Some(classify_stream_error_event(&event));
                    break 'stream;
                }
                "message_stop" => saw_message_stop = true,
                _ => {}
            }

            // `message_delta` carries the terminal `stop_reason`; without it an
            // empty turn is indistinguishable from a budget that ran out.
            if let Some(reason) = event
                .get("delta")
                .and_then(|delta| delta.get("stop_reason"))
                .and_then(Value::as_str)
            {
                stop_reason = Some(reason.to_string());
            }

            update_anthropic_usage_from_event(&event, &mut usage);

            if let Some(reasoning_delta) = extract_anthropic_reasoning_delta(&event) {
                saw_thinking = true;
                maybe_emit_reasoning_line_break_on_summary_change(
                    ai_logs,
                    &mut reasoning_stream_state,
                    extract_anthropic_reasoning_index(&event),
                );
                stream_reasoning_delta_to_stdout(
                    ai_logs,
                    &mut reasoning_stream_state,
                    &reasoning_delta,
                );
            }

            if let Some(content_delta) = extract_anthropic_content_delta(&event) {
                emit_reasoning_line_break(ai_logs, &mut reasoning_stream_state);
                model_output.push_str(&content_delta);
                stream_content_delta_to_stdout(ai_logs, &mut content_stream_state, &content_delta);
            }
        }
    }

    finalize_content_stdout(ai_logs, &mut content_stream_state);
    finalize_reasoning_stdout(ai_logs, &mut reasoning_stream_state);

    if let Some(failure) = stream_error {
        return Err(failure);
    }

    // Without `message_stop` the turn was cut off mid-flight. Any text captured so
    // far is a fragment; returning it would hand the agent loop a truncated action
    // to parse instead of reporting that the request never finished.
    if !saw_message_stop {
        return Err(miette::miette!(
            "Streaming response ended before message_stop after {} character(s) of output; \
             the response was truncated",
            model_output.chars().count()
        )
        .into());
    }

    if model_output.is_empty() {
        return Err(empty_output_failure(
            payload,
            stop_reason.as_deref(),
            saw_thinking,
            "Streaming response did not include output text deltas",
        ));
    }

    Ok((model_output, usage))
}

async fn non_stream_attempt(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    version: &str,
    payload: &Value,
    ai_logs: bool,
) -> std::result::Result<(String, TokenUsage), AttemptFailure> {
    let response = client
        .post(endpoint)
        .header("x-api-key", api_key)
        .header("anthropic-version", version)
        .json(payload)
        .send()
        .await
        .into_diagnostic()?;

    let status = response.status();
    if !status.is_success() {
        let retry_after = parse_retry_after(response.headers());
        let body = response.text().await.into_diagnostic()?;
        return Err(classify_request_failure(
            "Request",
            status,
            &body,
            retry_after,
        ));
    }

    let response_json = response.json::<Value>().await.into_diagnostic()?;

    let content = match extract_anthropic_non_stream_content(&response_json) {
        Some(content) => content,
        None => {
            let saw_thinking = response_json
                .get("content")
                .and_then(Value::as_array)
                .is_some_and(|blocks| {
                    blocks
                        .iter()
                        .any(|block| block.get("type").and_then(Value::as_str) == Some("thinking"))
                });

            return Err(empty_output_failure(
                payload,
                response_json.get("stop_reason").and_then(Value::as_str),
                saw_thinking,
                "Anthropic provider returned an unexpected response payload",
            ));
        }
    };

    if let Some(reasoning_text) = extract_anthropic_non_stream_reasoning(&response_json) {
        log_agent_progress(
            ai_logs,
            format!("🧠 Model reasoning output:\n{}", reasoning_text),
        );
    }

    let usage = extract_anthropic_response_usage(&response_json);
    Ok((content, usage))
}

impl AnalysisProvider for AnthropicProvider {
    fn provider_spec(&self) -> ProviderSpec {
        ProviderSpec {
            name: "anthropic".to_string(),
            model: Some(self.model.clone()),
            notes: format!("Endpoint: {}", self.endpoint),
        }
    }

    fn analyze_skill(
        &self,
        skill: &VulnerabilitySkill,
        prompt: &MiniPrompt,
        source_references: &[String],
        validator_context: &ValidatorContextMap,
        project_root: &Path,
        permission_prompt: &PermissionPromptSpec,
    ) -> Result<SkillIterationResult> {
        let canonical_root = project_root
            .canonicalize()
            .into_diagnostic()
            .with_context(|| {
                format!(
                    "Failed to canonicalize project root {}",
                    project_root.display()
                )
            })?;

        let system_prompt = build_agent_system_prompt();
        let initial_user_prompt = build_initial_user_prompt(
            prompt,
            source_references,
            validator_context,
            permission_prompt,
        );

        let mut messages = vec![serde_json::json!({
            "role": "user",
            "content": initial_user_prompt,
        })];

        run_agent_loop(
            skill,
            super::shared::AgentLoopContext {
                endpoint: &self.endpoint,
                ai_logs: self.ai_logs,
                project_root: &canonical_root,
                permission_prompt,
                provider_label: "Anthropic provider",
            },
            &mut messages,
            |messages| {
                block_on_runtime_aware(async {
                    let client = reqwest::Client::new();

                    // Streaming is not only for the live log. It is the only
                    // path the API accepts a large `max_tokens` on, and adaptive
                    // thinking needs that headroom to have anything left for a
                    // text block after the reasoning pass. Gating it on
                    // `--ai-logs` silently pushed quiet runs onto the tighter
                    // non-stream ceiling, where the reasoning pass eats the whole
                    // budget. The non-stream pass below is the fallback, not the
                    // default.
                    {
                        let mut last_stream_error: Option<String> = None;
                        let stream_payloads = build_anthropic_payload_variants(
                            &self.model,
                            system_prompt,
                            messages,
                            true,
                            self.reasoning_effort.as_deref(),
                        );

                        for (attempt_idx, payload) in stream_payloads.iter().enumerate() {
                            match attempt_with_backoff(
                                self.ai_logs,
                                "Streaming",
                                attempt_idx,
                                || {
                                    stream_attempt(
                                        &client,
                                        &self.endpoint,
                                        &self.api_key,
                                        &self.version,
                                        payload,
                                        self.ai_logs,
                                    )
                                },
                            )
                            .await
                            {
                                Ok(content) => return Ok(content),
                                // No payload variant can recover an account-level
                                // rejection, and neither can the non-stream pass.
                                Err(failure) if failure.is_fatal() => return Err(failure.report),
                                Err(failure) => {
                                    last_stream_error = Some(failure.report.to_string());
                                    log_agent_progress(
                                        self.ai_logs,
                                        format!(
                                            "⚠️ Streaming attempt {} failed: {}",
                                            attempt_idx + 1,
                                            failure.report
                                        ),
                                    );
                                }
                            }
                        }

                        if let Some(error) = last_stream_error {
                            log_agent_progress(
                                self.ai_logs,
                                format!(
                                    "⚠️ Streaming unavailable, falling back to non-stream request: {}",
                                    error
                                ),
                            );
                        }
                    }

                    let non_stream_payloads = build_anthropic_payload_variants(
                        &self.model,
                        system_prompt,
                        messages,
                        false,
                        self.reasoning_effort.as_deref(),
                    );
                    let mut last_non_stream_error: Option<String> = None;

                    for (attempt_idx, payload) in non_stream_payloads.iter().enumerate() {
                        match attempt_with_backoff(self.ai_logs, "Non-stream", attempt_idx, || {
                            non_stream_attempt(
                                &client,
                                &self.endpoint,
                                &self.api_key,
                                &self.version,
                                payload,
                                self.ai_logs,
                            )
                        })
                        .await
                        {
                            Ok(content) => return Ok(content),
                            Err(failure) if failure.is_fatal() => return Err(failure.report),
                            Err(failure) => {
                                last_non_stream_error = Some(failure.report.to_string());
                                log_agent_progress(
                                    self.ai_logs,
                                    format!(
                                        "⚠️ Non-stream attempt {} failed: {}",
                                        attempt_idx + 1,
                                        failure.report
                                    ),
                                );
                            }
                        }
                    }

                    Err(miette::miette!(
                        "All non-stream model request attempts failed for model '{}': {}",
                        self.model,
                        last_non_stream_error.unwrap_or_else(|| "unknown error".to_string())
                    ))
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_messages() -> Vec<Value> {
        vec![json!({"role": "user", "content": "hello"})]
    }

    #[test]
    fn payload_variants_without_reasoning_effort_still_lead_with_adaptive_thinking() {
        let messages = sample_messages();
        let variants =
            build_anthropic_payload_variants("claude-sonnet-4-6", "system", &messages, false, None);

        assert_eq!(variants.len(), 3);

        let first = &variants[0];
        assert_eq!(first["model"], "claude-sonnet-4-6");
        assert_eq!(first["max_tokens"], ADAPTIVE_THINKING_NON_STREAM_MAX_TOKENS);
        assert_eq!(first["thinking"]["type"], "adaptive");
        assert_eq!(first["thinking"]["display"], "summarized");
        assert!(first["thinking"].get("budget_tokens").is_none());
        // No `--reasoning-effort` means we leave effort at the API default.
        assert!(first.get("output_config").is_none());

        let legacy = &variants[1];
        assert_eq!(legacy["max_tokens"], THINKING_MAX_TOKENS);
        assert_eq!(legacy["thinking"]["type"], "enabled");
        assert_eq!(legacy["thinking"]["budget_tokens"], THINKING_BUDGET_TOKENS);
        assert!(legacy.get("output_config").is_none());

        let fallback = &variants[2];
        assert_eq!(fallback["max_tokens"], DEFAULT_MAX_TOKENS);
        assert!(fallback.get("thinking").is_none());
        assert!(fallback.get("output_config").is_none());
    }

    #[test]
    fn payload_variants_with_reasoning_effort_use_adaptive_thinking() {
        let messages = sample_messages();
        let variants = build_anthropic_payload_variants(
            "claude-opus-4-7",
            "system",
            &messages,
            false,
            Some("high"),
        );

        // Adaptive-only family: no legacy variant, because `enabled` is a 400.
        assert_eq!(variants.len(), 2);

        let first = &variants[0];
        assert_eq!(first["max_tokens"], ADAPTIVE_THINKING_NON_STREAM_MAX_TOKENS);
        assert_eq!(first["thinking"]["type"], "adaptive");
        assert_eq!(first["thinking"]["display"], "summarized");
        assert_eq!(first["output_config"]["effort"], "high");
        assert!(first["thinking"].get("budget_tokens").is_none());

        let fallback = &variants[1];
        assert_eq!(fallback["max_tokens"], DEFAULT_MAX_TOKENS);
        // Omitting `thinking` is not the same as turning it off on this family.
        assert_eq!(fallback["thinking"]["type"], "disabled");
        // `thinking: disabled` is rejected above effort `high`, so effort is left
        // at the API default rather than pinned to the requested one.
        assert!(fallback.get("output_config").is_none());
    }

    #[test]
    fn sonnet_5_ladder_never_sends_a_legacy_thinking_budget() {
        let messages = sample_messages();
        let variants = build_anthropic_payload_variants(
            "claude-sonnet-5",
            "system",
            &messages,
            true,
            Some("low"),
        );

        assert_eq!(variants.len(), 2);
        for variant in &variants {
            assert_ne!(variant["thinking"]["type"], "enabled");
            assert!(variant["thinking"].get("budget_tokens").is_none());
        }

        // Streaming gets the headroom the reasoning pass needs; the previous
        // 16000 ceiling was consumed by thinking before any text was emitted.
        assert_eq!(variants[0]["max_tokens"], ADAPTIVE_THINKING_MAX_TOKENS);
        assert_eq!(variants[0]["output_config"]["effort"], "low");
        assert_eq!(variants[1]["thinking"]["type"], "disabled");
    }

    #[test]
    fn fable_5_ladder_never_disables_thinking() {
        let messages = sample_messages();
        let variants = build_anthropic_payload_variants(
            "claude-fable-5",
            "system",
            &messages,
            true,
            Some("high"),
        );

        assert_eq!(variants.len(), 2);
        for variant in &variants {
            assert_eq!(variant["thinking"]["type"], "adaptive");
        }
        assert_eq!(variants[0]["output_config"]["effort"], "high");
        assert_eq!(variants[1]["output_config"]["effort"], "low");
    }

    #[test]
    fn pre_4_6_models_skip_the_adaptive_variant() {
        let messages = sample_messages();
        let variants =
            build_anthropic_payload_variants("claude-haiku-4-5", "system", &messages, false, None);

        assert_eq!(variants.len(), 2);
        assert_eq!(variants[0]["thinking"]["type"], "enabled");
        assert_eq!(
            variants[0]["thinking"]["budget_tokens"],
            THINKING_BUDGET_TOKENS
        );
        assert!(variants[1].get("thinking").is_none());
    }

    #[test]
    fn empty_output_with_exhausted_budget_names_the_cause() {
        let payload = json!({"max_tokens": 16000});

        let failure = empty_output_failure(&payload, Some("max_tokens"), true, "generic");
        let message = failure.report.to_string();
        assert!(message.contains("16000"), "{message}");
        assert!(message.contains("extended thinking"), "{message}");

        // Nothing to diagnose: fall back to the generic wording.
        let failure = empty_output_failure(&payload, Some("end_turn"), false, "generic");
        assert_eq!(failure.report.to_string(), "generic");
    }

    #[test]
    fn payload_variants_set_stream_flag_when_requested() {
        let messages = sample_messages();
        let variants = build_anthropic_payload_variants(
            "claude-sonnet-4-6",
            "system",
            &messages,
            true,
            Some("medium"),
        );

        for variant in &variants {
            assert_eq!(variant["stream"], true);
        }
    }

    #[test]
    fn extract_response_usage_reads_input_and_output_tokens() {
        let response = json!({
            "usage": {
                "input_tokens": 1234,
                "output_tokens": 56
            }
        });
        let usage = extract_anthropic_response_usage(&response);
        assert_eq!(usage.input_tokens, 1234);
        assert_eq!(usage.output_tokens, 56);
        assert_eq!(usage.cache_read_input_tokens, None);
        assert_eq!(usage.cache_creation_input_tokens, None);
    }

    #[test]
    fn extract_response_usage_reads_cache_fields_when_present() {
        let response = json!({
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 50
            }
        });
        let usage = extract_anthropic_response_usage(&response);
        assert_eq!(usage.cache_read_input_tokens, Some(100));
        assert_eq!(usage.cache_creation_input_tokens, Some(50));
    }

    #[test]
    fn extract_response_usage_returns_empty_when_missing() {
        let response = json!({"id": "msg_x"});
        let usage = extract_anthropic_response_usage(&response);
        assert!(usage.is_empty());
    }

    #[test]
    fn update_usage_from_message_start_event_sets_initial_input_tokens() {
        let event = json!({
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "usage": {
                    "input_tokens": 500,
                    "output_tokens": 1
                }
            }
        });
        let mut usage = TokenUsage::default();
        update_anthropic_usage_from_event(&event, &mut usage);
        assert_eq!(usage.input_tokens, 500);
        assert_eq!(usage.output_tokens, 1);
    }

    #[test]
    fn update_usage_from_message_start_event_captures_cache_fields() {
        let event = json!({
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 7,
                    "cache_creation_input_tokens": 3
                }
            }
        });
        let mut usage = TokenUsage::default();
        update_anthropic_usage_from_event(&event, &mut usage);
        assert_eq!(usage.cache_read_input_tokens, Some(7));
        assert_eq!(usage.cache_creation_input_tokens, Some(3));
    }

    #[test]
    fn update_usage_from_message_delta_event_replaces_output_tokens() {
        let mut usage = TokenUsage {
            input_tokens: 500,
            output_tokens: 5,
            ..Default::default()
        };
        let event = json!({
            "type": "message_delta",
            "usage": {
                "output_tokens": 320
            }
        });
        update_anthropic_usage_from_event(&event, &mut usage);
        // Anthropic sends cumulative output_tokens in message_delta — replace, don't add.
        assert_eq!(usage.output_tokens, 320);
        // input_tokens stays as it was.
        assert_eq!(usage.input_tokens, 500);
    }

    #[test]
    fn update_usage_from_unrelated_event_leaves_usage_unchanged() {
        let mut usage = TokenUsage {
            input_tokens: 5,
            output_tokens: 7,
            ..Default::default()
        };
        let event = json!({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "x"}
        });
        update_anthropic_usage_from_event(&event, &mut usage);
        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 7);
    }

    #[test]
    fn payload_variants_preserve_string_message_content_as_text_block() {
        let messages = sample_messages();
        let variants =
            build_anthropic_payload_variants("claude-sonnet-4-6", "system", &messages, false, None);

        let first_message = &variants[0]["messages"][0];
        assert_eq!(first_message["role"], "user");
        assert_eq!(first_message["content"][0]["type"], "text");
        assert_eq!(first_message["content"][0]["text"], "hello");
    }

    #[test]
    fn payload_variants_mark_first_and_last_message_as_cacheable() {
        let messages = vec![
            json!({"role": "user", "content": "initial prompt"}),
            json!({"role": "assistant", "content": "{\"action\":\"read_file\"}"}),
            json!({"role": "user", "content": "tool result"}),
        ];
        let variants =
            build_anthropic_payload_variants("claude-opus-5", "system", &messages, false, None);

        for payload in &variants {
            let rendered = &payload["messages"];
            assert_eq!(
                rendered[0]["content"][0]["cache_control"]["type"], "ephemeral",
                "the stable prefix must carry a breakpoint"
            );
            assert!(
                rendered[1]["content"][0].get("cache_control").is_none(),
                "middle turns must not consume breakpoints"
            );
            assert_eq!(
                rendered[2]["content"][0]["cache_control"]["type"], "ephemeral",
                "the rolling tail must carry a breakpoint"
            );
        }
    }

    #[test]
    fn payload_variants_use_a_single_breakpoint_for_a_lone_message() {
        let messages = vec![json!({"role": "user", "content": "only turn"})];
        let variants =
            build_anthropic_payload_variants("claude-opus-5", "system", &messages, false, None);

        let rendered = &variants[0]["messages"];
        assert_eq!(rendered.as_array().map(Vec::len), Some(1));
        assert_eq!(
            rendered[0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn auth_failures_are_fatal() {
        for status in [
            reqwest::StatusCode::UNAUTHORIZED,
            reqwest::StatusCode::FORBIDDEN,
        ] {
            let failure = classify_request_failure("Request", status, "{}", None);
            assert!(failure.is_fatal(), "{} should not be retried", status);
        }
    }

    #[test]
    fn api_error_message_falls_back_to_the_raw_body() {
        assert_eq!(api_error_message("not json at all"), "not json at all");
        assert_eq!(
            api_error_message(r#"{"error":{"message":"clean message"}}"#),
            "clean message"
        );
    }

    #[test]
    fn mid_stream_credit_exhaustion_is_fatal() {
        // Delivered inside a 200 response, so the HTTP status cannot catch it.
        let event = json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Your credit balance is too low to access the Anthropic API."
            }
        });
        let failure = classify_stream_error_event(&event);

        assert!(failure.is_fatal());
        assert!(
            failure
                .report
                .to_string()
                .contains("credit balance is too low")
        );
    }

    #[test]
    fn mid_stream_error_without_a_message_is_still_reported() {
        let failure = classify_stream_error_event(&json!({"type": "error"}));

        assert!(!failure.is_fatal());
        assert!(failure.report.to_string().contains("unspecified error"));
    }

    #[test]
    fn account_level_markers_are_shared_by_both_paths() {
        assert!(mentions_account_level_problem(
            "Your credit balance is too low"
        ));
        assert!(mentions_account_level_problem("invalid authentication"));
        assert!(!mentions_account_level_problem(
            "\"thinking.type.enabled\" is not supported for this model"
        ));
    }

    #[test]
    fn rate_limits_wait_rather_than_switching_payload_variant() {
        let body = r#"{"type":"error","error":{"type":"rate_limit_error","message":"This request would exceed your rate limit of 500,000 input tokens per minute"}}"#;
        let failure = classify_request_failure(
            "Request",
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            body,
            Some(Duration::from_secs(12)),
        );

        assert_eq!(failure.kind, FailureKind::RateLimited);
        assert!(!failure.is_fatal());
        assert_eq!(failure.retry_after, Some(Duration::from_secs(12)));
        assert!(failure.report.to_string().contains("throttling"));
    }

    #[test]
    fn server_errors_are_waited_out_too() {
        let failure = classify_request_failure(
            "Request",
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            "upstream boom",
            None,
        );

        assert_eq!(failure.kind, FailureKind::RateLimited);
        assert_eq!(failure.retry_after, None);
    }

    #[test]
    fn unsupported_parameter_advances_to_the_next_variant() {
        let body = r#"{"type":"error","error":{"message":"\"thinking.type.enabled\" is not supported for this model."}}"#;
        let failure =
            classify_request_failure("Request", reqwest::StatusCode::BAD_REQUEST, body, None);

        assert_eq!(failure.kind, FailureKind::NextVariant);
    }

    #[test]
    fn credit_exhaustion_remains_fatal_under_the_new_kinds() {
        let body = r#"{"type":"error","error":{"message":"Your credit balance is too low to access the Anthropic API."}}"#;
        let failure =
            classify_request_failure("Request", reqwest::StatusCode::BAD_REQUEST, body, None);

        assert!(failure.is_fatal());
    }

    #[test]
    fn mid_stream_overload_is_waited_out() {
        let event = json!({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"}
        });
        let failure = classify_stream_error_event(&event);

        assert_eq!(failure.kind, FailureKind::RateLimited);
        assert!(!failure.is_fatal());
    }

    #[test]
    fn mid_stream_rate_limit_is_waited_out_not_fatal() {
        let event = json!({
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}
        });
        let failure = classify_stream_error_event(&event);

        assert_eq!(failure.kind, FailureKind::RateLimited);
        assert!(!failure.is_fatal());
    }

    #[test]
    fn retry_after_header_is_read_in_seconds() {
        let mut headers = reqwest::header::HeaderMap::new();
        assert_eq!(parse_retry_after(&headers), None);

        headers.insert(reqwest::header::RETRY_AFTER, "30".parse().unwrap());
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(30)));

        headers.insert(
            reqwest::header::RETRY_AFTER,
            "Wed, 21 Oct 2026 07:28:00 GMT".parse().unwrap(),
        );
        assert_eq!(
            parse_retry_after(&headers),
            None,
            "http-date form falls back to exponential backoff"
        );
    }
}
