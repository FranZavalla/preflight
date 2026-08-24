use miette::{Context, IntoDiagnostic, Result};
use serde_json::{Value, json};
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

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

#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub ai_logs: bool,
    pub reasoning_effort: Option<String>,
    pub ollama_compat: bool,
    /// Minimum spacing between outbound requests, or `None` to send freely.
    pub min_request_interval: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApiFamily {
    ChatCompletions,
    Responses,
}

fn detect_api_family(endpoint: &str, ollama_compat: bool) -> ApiFamily {
    if ollama_compat {
        return ApiFamily::ChatCompletions;
    }

    if endpoint.contains("/responses") {
        ApiFamily::Responses
    } else {
        ApiFamily::ChatCompletions
    }
}

/// The ladder runs 15s -> 30s -> 60s -> 120s -> 240s -> 300s, which clears any
/// per-minute window even when the gateway sends no hint of its own.
const MAX_RATE_LIMIT_RETRIES: usize = 6;
const RATE_LIMIT_BASE_DELAY: Duration = Duration::from_secs(15);
const RATE_LIMIT_MAX_DELAY: Duration = Duration::from_secs(300);

/// What a caller should do next after a failed attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureKind {
    /// Nothing can succeed: exhausted credits, bad key, unroutable model.
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

/// Earliest instant the next request may leave, shared across every request in
/// the process so the pacing survives the per-skill agent loops.
static NEXT_REQUEST_SLOT: LazyLock<Mutex<Option<Instant>>> = LazyLock::new(|| Mutex::new(None));

/// Claims the next send slot and waits for it.
///
/// Reserving the slot before releasing the lock keeps two callers from claiming
/// the same instant, so the spacing holds even if requests ever overlap.
async fn pace_request(min_interval: Option<Duration>) {
    let Some(interval) = min_interval.filter(|value| !value.is_zero()) else {
        return;
    };

    let now = Instant::now();
    let wait = {
        let mut slot = NEXT_REQUEST_SLOT
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let send_at = slot.map_or(now, |next| next.max(now));
        *slot = Some(send_at + interval);
        send_at.saturating_duration_since(now)
    };

    if !wait.is_zero() {
        tokio::time::sleep(wait).await;
    }
}

/// Pulls `error.message` out of an OpenAI-shaped error body, falling back to the
/// raw body when it is not the shape we expect.
fn api_error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .as_ref()
        .and_then(|parsed| parsed.pointer("/error/message"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| body.trim().to_string())
}

/// Reads how long to wait before resending.
///
/// `retry-after` is seconds. OpenRouter sends `x-ratelimit-reset` instead, as
/// the epoch instant the window reopens — in milliseconds, though seconds and
/// plain relative values show up across gateways, so all three are accepted.
fn parse_retry_delay(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    if let Some(seconds) = headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
    {
        return Some(Duration::from_secs(seconds));
    }

    let reset = headers
        .get("x-ratelimit-reset")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())?;

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_millis() as u64;

    let millis = if reset > 1_000_000_000_000 {
        reset.saturating_sub(now_ms)
    } else if reset > 1_000_000_000 {
        reset.saturating_mul(1_000).saturating_sub(now_ms)
    } else {
        reset.saturating_mul(1_000)
    };

    (millis > 0).then(|| Duration::from_millis(millis).min(RATE_LIMIT_MAX_DELAY))
}

fn classify_request_failure(
    label: &str,
    status: reqwest::StatusCode,
    body: &str,
    retry_after: Option<Duration>,
) -> AttemptFailure {
    // 402 is how a gateway reports an empty balance; 401/403 are key and routing
    // problems. No payload variant and no amount of waiting fixes any of them.
    if matches!(status.as_u16(), 401 | 402 | 403) {
        return AttemptFailure {
            report: miette::miette!(
                "AI provider rejected the request ({}): {}",
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
                "AI provider is throttling or unavailable ({}): {}",
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

/// Runs one payload variant, waiting out rate limits instead of falling through
/// to a different variant.
///
/// Switching payload shape on a 429 is actively harmful: a changed `reasoning`
/// config invalidates the cached prefix, so the retry resends the whole
/// transcript uncached and pushes the account *further* past its ceiling.
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

/// Sends one payload, respecting the configured pacing.
async fn post_payload(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    payload: &Value,
    min_interval: Option<Duration>,
) -> std::result::Result<reqwest::Response, AttemptFailure> {
    pace_request(min_interval).await;

    client
        .post(endpoint)
        .bearer_auth(api_key)
        .json(payload)
        .send()
        .await
        .into_diagnostic()
        .map_err(AttemptFailure::from)
}

fn build_chat_payload_variants(
    model: &str,
    messages: &[Value],
    stream: bool,
    reasoning_effort: Option<&str>,
    ollama_compat: bool,
) -> Vec<Value> {
    let mut base = json!({
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_object"
        }
    });

    if stream {
        base["stream"] = Value::Bool(true);
        if !ollama_compat {
            base["stream_options"] = json!({"include_usage": true});
        }
    }

    let mut variants = vec![base.clone()];

    if ollama_compat {
        let mut with_ollama_think = base.clone();
        with_ollama_think["think"] = Value::Bool(true);
        variants.insert(0, with_ollama_think);
    }

    let Some(effort_raw) = reasoning_effort else {
        return variants;
    };

    let effort = effort_raw.trim();
    if effort.is_empty() {
        return variants;
    }

    let mut with_reasoning_object = base.clone();
    with_reasoning_object["reasoning"] = json!({ "effort": effort });

    let mut with_reasoning_effort = base.clone();
    with_reasoning_effort["reasoning_effort"] = Value::String(effort.to_string());

    let mut with_reasoning_object_and_ollama = with_reasoning_object.clone();
    with_reasoning_object_and_ollama["think"] = Value::Bool(true);

    let mut with_reasoning_effort_and_ollama = with_reasoning_effort.clone();
    with_reasoning_effort_and_ollama["think"] = Value::Bool(true);

    if ollama_compat {
        vec![
            with_reasoning_object_and_ollama,
            with_reasoning_effort_and_ollama,
            with_reasoning_object,
            with_reasoning_effort,
            base,
        ]
    } else {
        vec![with_reasoning_object, with_reasoning_effort, base]
    }
}

fn build_responses_payload_variants(
    model: &str,
    messages: &[Value],
    stream: bool,
    reasoning_effort: Option<&str>,
) -> Vec<Value> {
    let input = messages_to_responses_input(messages);

    let mut base = json!({
        "model": model,
        "input": input,
        "text": {
            "format": {
                "type": "json_object"
            }
        }
    });

    if stream {
        base["stream"] = Value::Bool(true);
    }

    let Some(effort_raw) = reasoning_effort else {
        return vec![base];
    };

    let effort = effort_raw.trim();
    if effort.is_empty() {
        return vec![base];
    }

    let mut with_reasoning_summary = base.clone();
    with_reasoning_summary["reasoning"] = json!({
        "effort": effort,
        "summary": "auto"
    });

    let mut with_reasoning_effort = base.clone();
    with_reasoning_effort["reasoning"] = json!({ "effort": effort });

    vec![with_reasoning_summary, with_reasoning_effort, base]
}

fn messages_to_responses_input(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .map(|message| {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user");
            let content = message.get("content").unwrap_or(&Value::Null);

            json!({
                "role": role,
                "content": normalize_responses_input_content(role, content)
            })
        })
        .collect()
}

fn text_block_type_for_role(role: &str) -> &'static str {
    if role.eq_ignore_ascii_case("assistant") {
        "output_text"
    } else {
        "input_text"
    }
}

fn normalize_responses_input_content(role: &str, content: &Value) -> Value {
    let text_block_type = text_block_type_for_role(role);

    if let Some(text) = content.as_str() {
        return json!([
            {
                "type": text_block_type,
                "text": text
            }
        ]);
    }

    if let Some(chunks) = content.as_array() {
        let normalized = chunks
            .iter()
            .map(|chunk| {
                if let Some(text) = chunk.get("text").and_then(Value::as_str) {
                    json!({
                        "type": text_block_type,
                        "text": text
                    })
                } else {
                    chunk.clone()
                }
            })
            .collect::<Vec<Value>>();

        return Value::Array(normalized);
    }

    json!([
        {
            "type": text_block_type,
            "text": content.to_string()
        }
    ])
}

fn extract_summary_index(event: &Value) -> Option<i64> {
    event
        .get("summary_index")
        .and_then(Value::as_i64)
        .or_else(|| event.pointer("/summary/index").and_then(Value::as_i64))
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

fn extract_chat_reasoning_delta(event: &Value) -> Option<String> {
    event
        .pointer("/choices/0/delta/reasoning_content")
        .and_then(Value::as_str)
        .or_else(|| {
            event
                .pointer("/choices/0/delta/reasoning")
                .and_then(Value::as_str)
        })
        .or_else(|| {
            event
                .pointer("/choices/0/delta/thinking")
                .and_then(Value::as_str)
        })
        .map(ToString::to_string)
}

fn extract_chat_content_delta(event: &Value) -> Option<String> {
    event
        .pointer("/choices/0/delta/content")
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn extract_responses_reasoning_delta(event: &Value) -> Option<String> {
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();

    let is_delta_event = event_type.ends_with(".delta");
    let is_reasoning_event = event_type.contains("reasoning") || event_type.contains("summary");

    if !(is_delta_event && is_reasoning_event) {
        return None;
    }

    event
        .get("delta")
        .and_then(Value::as_str)
        .or_else(|| event.get("text").and_then(Value::as_str))
        .or_else(|| event.pointer("/summary/text").and_then(Value::as_str))
        .map(ToString::to_string)
}

fn extract_responses_content_delta(event: &Value) -> Option<String> {
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();

    if !event_type.ends_with(".delta") {
        return None;
    }

    if event_type.contains("reasoning") || event_type.contains("summary") {
        return None;
    }

    if event_type.contains("output_text") || event_type.contains("message") {
        return event
            .get("delta")
            .and_then(Value::as_str)
            .or_else(|| event.get("text").and_then(Value::as_str))
            .or_else(|| event.pointer("/content/delta").and_then(Value::as_str))
            .map(ToString::to_string);
    }

    None
}

fn extract_responses_output_text(response_json: &Value) -> Option<String> {
    if let Some(output_text) = response_json.get("output_text").and_then(Value::as_str)
        && !output_text.trim().is_empty()
    {
        return Some(output_text.to_string());
    }

    let mut chunks = Vec::new();

    if let Some(outputs) = response_json.get("output").and_then(Value::as_array) {
        for item in outputs {
            let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();

            if (item_type == "output_text" || item_type == "text")
                && item.get("text").and_then(Value::as_str).is_some()
            {
                if let Some(text) = item.get("text").and_then(Value::as_str)
                    && !text.trim().is_empty()
                {
                    chunks.push(text.to_string());
                }

                continue;
            }

            if let Some(content) = item.get("content").and_then(Value::as_array) {
                for block in content {
                    let block_type = block
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if (block_type == "output_text" || block_type == "text")
                        && block.get("text").and_then(Value::as_str).is_some()
                        && let Some(text) = block.get("text").and_then(Value::as_str)
                        && !text.trim().is_empty()
                    {
                        chunks.push(text.to_string());
                    }
                }
            }
        }
    }

    if chunks.is_empty() {
        None
    } else {
        Some(chunks.join(""))
    }
}

fn extract_responses_reasoning_summary(response_json: &Value) -> Option<String> {
    let mut chunks = Vec::new();

    if let Some(outputs) = response_json.get("output").and_then(Value::as_array) {
        for item in outputs {
            let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();

            if item_type != "reasoning" {
                continue;
            }

            if let Some(summary_text) = item.get("summary").and_then(Value::as_str)
                && !summary_text.trim().is_empty()
            {
                chunks.push(summary_text.to_string());
            }

            if let Some(summary_items) = item.get("summary").and_then(Value::as_array) {
                for entry in summary_items {
                    if let Some(text) = entry.get("text").and_then(Value::as_str)
                        && !text.trim().is_empty()
                    {
                        chunks.push(text.to_string());
                    }
                }
            }
        }
    }

    if chunks.is_empty() {
        None
    } else {
        Some(chunks.join("\n"))
    }
}

fn extract_openai_chat_usage(response_json: &Value) -> Option<TokenUsage> {
    let usage = response_json.get("usage")?;
    if usage.is_null() {
        return None;
    }

    let input_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read_input_tokens = usage
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(Value::as_u64);
    let reasoning_tokens = usage
        .pointer("/completion_tokens_details/reasoning_tokens")
        .and_then(Value::as_u64);

    if input_tokens == 0
        && output_tokens == 0
        && cache_read_input_tokens.is_none()
        && reasoning_tokens.is_none()
    {
        return None;
    }

    Some(TokenUsage {
        input_tokens,
        output_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens: None,
        reasoning_tokens,
    })
}

fn extract_openai_responses_usage(response_json: &Value) -> Option<TokenUsage> {
    let usage = response_json.get("usage")?;
    if usage.is_null() {
        return None;
    }

    let input_tokens = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read_input_tokens = usage
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(Value::as_u64);
    let reasoning_tokens = usage
        .pointer("/output_tokens_details/reasoning_tokens")
        .and_then(Value::as_u64);

    if input_tokens == 0
        && output_tokens == 0
        && cache_read_input_tokens.is_none()
        && reasoning_tokens.is_none()
    {
        return None;
    }

    Some(TokenUsage {
        input_tokens,
        output_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens: None,
        reasoning_tokens,
    })
}

fn extract_openai_responses_event_usage(event: &Value) -> Option<TokenUsage> {
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();

    if event_type != "response.completed" && event_type != "response.incomplete" {
        return None;
    }

    let response = event.get("response")?;
    extract_openai_responses_usage(response)
}

async fn stream_chat_attempt(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    payload: &Value,
    ai_logs: bool,
    min_interval: Option<Duration>,
) -> std::result::Result<(String, TokenUsage), AttemptFailure> {
    let mut response = post_payload(client, endpoint, api_key, payload, min_interval).await?;

    let status = response.status();
    if !status.is_success() {
        let retry_after = parse_retry_delay(response.headers());
        let body = response.text().await.into_diagnostic()?;
        return Err(classify_request_failure(
            "Streaming request",
            status,
            &body,
            retry_after,
        ));
    }

    let mut pending = String::new();
    let mut model_output = String::new();
    let mut reasoning_stream_state = ReasoningStreamState::default();
    let mut content_stream_state = ContentStreamState::default();
    let mut usage = TokenUsage::default();

    while let Some(chunk) = response.chunk().await.into_diagnostic()? {
        pending.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(newline_index) = pending.find('\n') {
            let line = pending[..newline_index].trim_end_matches('\r').to_string();
            pending.drain(..=newline_index);

            let line = line.trim();
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

            if let Some(chunk_usage) = extract_openai_chat_usage(&event) {
                usage = chunk_usage;
            }

            if let Some(reasoning_delta) = extract_chat_reasoning_delta(&event) {
                stream_reasoning_delta_to_stdout(
                    ai_logs,
                    &mut reasoning_stream_state,
                    &reasoning_delta,
                );
            }

            if let Some(content_delta) = extract_chat_content_delta(&event) {
                emit_reasoning_line_break(ai_logs, &mut reasoning_stream_state);
                model_output.push_str(&content_delta);
                stream_content_delta_to_stdout(ai_logs, &mut content_stream_state, &content_delta);
            }
        }
    }

    finalize_content_stdout(ai_logs, &mut content_stream_state);
    finalize_reasoning_stdout(ai_logs, &mut reasoning_stream_state);

    if model_output.is_empty() {
        return Err(miette::miette!("Streaming response did not include content deltas").into());
    }

    Ok((model_output, usage))
}

async fn stream_responses_attempt(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    payload: &Value,
    ai_logs: bool,
    min_interval: Option<Duration>,
) -> std::result::Result<(String, TokenUsage), AttemptFailure> {
    let mut response = post_payload(client, endpoint, api_key, payload, min_interval).await?;

    let status = response.status();
    if !status.is_success() {
        let retry_after = parse_retry_delay(response.headers());
        let body = response.text().await.into_diagnostic()?;
        return Err(classify_request_failure(
            "Streaming request",
            status,
            &body,
            retry_after,
        ));
    }

    let mut pending = String::new();
    let mut model_output = String::new();
    let mut reasoning_stream_state = ReasoningStreamState::default();
    let mut content_stream_state = ContentStreamState::default();
    let mut usage = TokenUsage::default();

    while let Some(chunk) = response.chunk().await.into_diagnostic()? {
        pending.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(newline_index) = pending.find('\n') {
            let line = pending[..newline_index].trim_end_matches('\r').to_string();
            pending.drain(..=newline_index);

            let line = line.trim();
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

            if let Some(event_usage) = extract_openai_responses_event_usage(&event) {
                usage = event_usage;
            }

            if let Some(reasoning_delta) = extract_responses_reasoning_delta(&event) {
                maybe_emit_reasoning_line_break_on_summary_change(
                    ai_logs,
                    &mut reasoning_stream_state,
                    extract_summary_index(&event),
                );
                stream_reasoning_delta_to_stdout(
                    ai_logs,
                    &mut reasoning_stream_state,
                    &reasoning_delta,
                );
            }

            if let Some(content_delta) = extract_responses_content_delta(&event) {
                emit_reasoning_line_break(ai_logs, &mut reasoning_stream_state);
                model_output.push_str(&content_delta);
                stream_content_delta_to_stdout(ai_logs, &mut content_stream_state, &content_delta);
            }
        }
    }

    finalize_content_stdout(ai_logs, &mut content_stream_state);
    finalize_reasoning_stdout(ai_logs, &mut reasoning_stream_state);

    if model_output.is_empty() {
        return Err(
            miette::miette!("Streaming response did not include output text deltas").into(),
        );
    }

    Ok((model_output, usage))
}

async fn non_stream_chat_attempt(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    payload: &Value,
    ai_logs: bool,
    min_interval: Option<Duration>,
) -> std::result::Result<(String, TokenUsage), AttemptFailure> {
    let response = post_payload(client, endpoint, api_key, payload, min_interval).await?;

    // `error_for_status` throws the body away, and the body is where the gateway
    // explains *why* it refused — which decides whether waiting can help.
    let status = response.status();
    if !status.is_success() {
        let retry_after = parse_retry_delay(response.headers());
        let body = response.text().await.into_diagnostic()?;
        return Err(classify_request_failure(
            "Non-stream request",
            status,
            &body,
            retry_after,
        ));
    }

    let response_json = response.json::<Value>().await.into_diagnostic()?;

    let content = response_json
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            AttemptFailure::from(miette::miette!(
                "AI provider returned an unexpected response payload"
            ))
        })?;

    if let Some(reasoning_text) = response_json
        .pointer("/choices/0/message/reasoning_content")
        .and_then(Value::as_str)
        .or_else(|| {
            response_json
                .pointer("/choices/0/message/reasoning")
                .and_then(Value::as_str)
        })
        .or_else(|| {
            response_json
                .pointer("/choices/0/message/thinking")
                .and_then(Value::as_str)
        })
    {
        log_agent_progress(
            ai_logs,
            format!("🧠 Model reasoning output:\n{}", &reasoning_text),
        );
    }

    let usage = extract_openai_chat_usage(&response_json).unwrap_or_default();
    Ok((content.to_string(), usage))
}

async fn non_stream_responses_attempt(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    payload: &Value,
    ai_logs: bool,
    min_interval: Option<Duration>,
) -> std::result::Result<(String, TokenUsage), AttemptFailure> {
    let response = post_payload(client, endpoint, api_key, payload, min_interval).await?;

    // `error_for_status` throws the body away, and the body is where the gateway
    // explains *why* it refused — which decides whether waiting can help.
    let status = response.status();
    if !status.is_success() {
        let retry_after = parse_retry_delay(response.headers());
        let body = response.text().await.into_diagnostic()?;
        return Err(classify_request_failure(
            "Non-stream request",
            status,
            &body,
            retry_after,
        ));
    }

    let response_json = response.json::<Value>().await.into_diagnostic()?;

    let content = extract_responses_output_text(&response_json).ok_or_else(|| {
        AttemptFailure::from(miette::miette!(
            "AI provider returned an unexpected response payload"
        ))
    })?;

    if let Some(reasoning_summary) = extract_responses_reasoning_summary(&response_json) {
        log_agent_progress(
            ai_logs,
            format!("🧠 Model reasoning summary:\n{}", &reasoning_summary),
        );
    }

    let usage = extract_openai_responses_usage(&response_json).unwrap_or_default();
    Ok((content, usage))
}

impl AnalysisProvider for OpenAiProvider {
    fn provider_spec(&self) -> ProviderSpec {
        let api_family = detect_api_family(&self.endpoint, self.ollama_compat);
        let api_note = match api_family {
            ApiFamily::ChatCompletions => "chat-completions",
            ApiFamily::Responses => "responses",
        };

        let reasoning_note = self
            .reasoning_effort
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(|value| format!(", reasoning_effort={}", value))
            .unwrap_or_default();

        ProviderSpec {
            name: "openai-compatible".to_string(),
            model: Some(self.model.clone()),
            notes: format!(
                "Endpoint: {} (api={}){}",
                self.endpoint, api_note, reasoning_note
            ),
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

        let api_family = detect_api_family(&self.endpoint, self.ollama_compat);

        let system_prompt = build_agent_system_prompt();
        let initial_user_prompt = build_initial_user_prompt(
            prompt,
            source_references,
            validator_context,
            permission_prompt,
        );

        let mut messages = vec![
            json!({
                "role": "system",
                "content": system_prompt,
            }),
            json!({
                "role": "user",
                "content": initial_user_prompt,
            }),
        ];

        run_agent_loop(
            skill,
            super::shared::AgentLoopContext {
                endpoint: &self.endpoint,
                ai_logs: self.ai_logs,
                project_root: &canonical_root,
                permission_prompt,
                provider_label: "AI provider",
            },
            &mut messages,
            |messages| {
                block_on_runtime_aware(async {
                    let client = reqwest::Client::new();
                    let reasoning_effort = self.reasoning_effort.as_deref();
                    let min_interval = self.min_request_interval;

                    if self.ai_logs {
                        let mut last_stream_error: Option<String> = None;
                        let stream_payloads = match api_family {
                            ApiFamily::ChatCompletions => build_chat_payload_variants(
                                &self.model,
                                messages,
                                true,
                                reasoning_effort,
                                self.ollama_compat,
                            ),
                            ApiFamily::Responses => build_responses_payload_variants(
                                &self.model,
                                messages,
                                true,
                                reasoning_effort,
                            ),
                        };

                        for (attempt_idx, stream_payload) in stream_payloads.iter().enumerate() {
                            match attempt_with_backoff(
                                self.ai_logs,
                                "Streaming",
                                attempt_idx,
                                || async {
                                    match api_family {
                                        ApiFamily::ChatCompletions => {
                                            stream_chat_attempt(
                                                &client,
                                                &self.endpoint,
                                                &self.api_key,
                                                stream_payload,
                                                self.ai_logs,
                                                min_interval,
                                            )
                                            .await
                                        }
                                        ApiFamily::Responses => {
                                            stream_responses_attempt(
                                                &client,
                                                &self.endpoint,
                                                &self.api_key,
                                                stream_payload,
                                                self.ai_logs,
                                                min_interval,
                                            )
                                            .await
                                        }
                                    }
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

                    let non_stream_payloads = match api_family {
                        ApiFamily::ChatCompletions => build_chat_payload_variants(
                            &self.model,
                            messages,
                            false,
                            reasoning_effort,
                            self.ollama_compat,
                        ),
                        ApiFamily::Responses => build_responses_payload_variants(
                            &self.model,
                            messages,
                            false,
                            reasoning_effort,
                        ),
                    };

                    let mut last_non_stream_error: Option<String> = None;

                    for (attempt_idx, payload) in non_stream_payloads.iter().enumerate() {
                        match attempt_with_backoff(
                            self.ai_logs,
                            "Non-stream",
                            attempt_idx,
                            || async {
                                match api_family {
                                    ApiFamily::ChatCompletions => {
                                        non_stream_chat_attempt(
                                            &client,
                                            &self.endpoint,
                                            &self.api_key,
                                            payload,
                                            self.ai_logs,
                                            min_interval,
                                        )
                                        .await
                                    }
                                    ApiFamily::Responses => {
                                        non_stream_responses_attempt(
                                            &client,
                                            &self.endpoint,
                                            &self.api_key,
                                            payload,
                                            self.ai_logs,
                                            min_interval,
                                        )
                                        .await
                                    }
                                }
                            },
                        )
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
                        "All non-stream model request attempts failed: {}",
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
    use crate::model::TokenUsage;

    fn headers_with(name: &'static str, value: String) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(name, value.parse().expect("valid header value"));
        headers
    }

    #[test]
    fn retry_after_header_is_read_in_seconds() {
        let headers = headers_with("retry-after", "12".to_string());
        assert_eq!(parse_retry_delay(&headers), Some(Duration::from_secs(12)));
    }

    #[test]
    fn ratelimit_reset_is_read_as_epoch_milliseconds() {
        // What OpenRouter actually sends: the instant the window reopens.
        let reset_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_millis() as u64
            + 30_000;

        let delay = parse_retry_delay(&headers_with("x-ratelimit-reset", reset_ms.to_string()))
            .expect("a delay");

        // Allow slack for the clock read between building and parsing.
        assert!(
            delay <= Duration::from_secs(30) && delay >= Duration::from_secs(25),
            "unexpected delay: {:?}",
            delay
        );
    }

    #[test]
    fn a_reset_already_in_the_past_yields_no_delay() {
        let elapsed_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_millis() as u64
            - 5_000;

        assert_eq!(
            parse_retry_delay(&headers_with("x-ratelimit-reset", elapsed_ms.to_string())),
            None
        );
    }

    #[test]
    fn throttling_is_retried_not_escalated_to_the_next_variant() {
        let failure = classify_request_failure(
            "Streaming request",
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"message":"Rate limit exceeded: new-account-rpm"}}"#,
            Some(Duration::from_secs(9)),
        );

        assert_eq!(failure.kind, FailureKind::RateLimited);
        assert_eq!(failure.retry_after, Some(Duration::from_secs(9)));
        assert!(!failure.is_fatal());
        assert!(failure.report.to_string().contains("new-account-rpm"));
    }

    #[test]
    fn an_empty_balance_is_fatal_so_it_is_not_retried_six_times() {
        let failure = classify_request_failure(
            "Non-stream request",
            reqwest::StatusCode::PAYMENT_REQUIRED,
            r#"{"error":{"message":"Insufficient credits"}}"#,
            None,
        );

        assert!(failure.is_fatal());
        assert!(failure.report.to_string().contains("Insufficient credits"));
    }

    #[test]
    fn a_rejected_payload_shape_falls_through_to_the_next_variant() {
        let failure = classify_request_failure(
            "Non-stream request",
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"Unsupported parameter: reasoning"}}"#,
            None,
        );

        assert_eq!(failure.kind, FailureKind::NextVariant);
    }

    #[tokio::test]
    async fn pacing_spaces_consecutive_requests() {
        let interval = Duration::from_millis(120);
        let started = Instant::now();

        for _ in 0..3 {
            pace_request(Some(interval)).await;
        }

        // First send is free, the next two wait one interval each.
        assert!(
            started.elapsed() >= interval * 2,
            "requests were not spaced: {:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn no_ceiling_means_no_waiting() {
        let started = Instant::now();
        pace_request(None).await;
        assert!(started.elapsed() < Duration::from_millis(50));
    }

    #[test]
    fn extract_chat_usage_reads_prompt_and_completion_tokens() {
        let response = json!({
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 250, "completion_tokens": 80, "total_tokens": 330}
        });
        let usage = extract_openai_chat_usage(&response).expect("usage present");
        assert_eq!(usage.input_tokens, 250);
        assert_eq!(usage.output_tokens, 80);
    }

    #[test]
    fn extract_chat_usage_returns_none_when_missing() {
        let response = json!({"choices": []});
        assert!(extract_openai_chat_usage(&response).is_none());
    }

    #[test]
    fn extract_chat_usage_returns_none_when_usage_is_null() {
        // Streaming chat completions deliver `usage: null` in non-final chunks.
        let response = json!({"choices": [], "usage": null});
        assert!(extract_openai_chat_usage(&response).is_none());
    }

    #[test]
    fn extract_chat_usage_picks_up_cached_input_tokens() {
        let response = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 40}
            }
        });
        let usage = extract_openai_chat_usage(&response).expect("usage present");
        assert_eq!(usage.cache_read_input_tokens, Some(40));
    }

    #[test]
    fn extract_chat_usage_picks_up_reasoning_tokens() {
        let response = json!({
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 80,
                "completion_tokens_details": {"reasoning_tokens": 60}
            }
        });
        let usage = extract_openai_chat_usage(&response).expect("usage present");
        assert_eq!(usage.reasoning_tokens, Some(60));
    }

    #[test]
    fn extract_responses_usage_reads_input_and_output_tokens() {
        let response = json!({
            "usage": {"input_tokens": 1500, "output_tokens": 90, "total_tokens": 1590}
        });
        let usage = extract_openai_responses_usage(&response).expect("usage present");
        assert_eq!(usage.input_tokens, 1500);
        assert_eq!(usage.output_tokens, 90);
    }

    #[test]
    fn extract_responses_usage_picks_up_cached_and_reasoning() {
        let response = json!({
            "usage": {
                "input_tokens": 100,
                "output_tokens": 80,
                "input_tokens_details": {"cached_tokens": 25},
                "output_tokens_details": {"reasoning_tokens": 50}
            }
        });
        let usage = extract_openai_responses_usage(&response).expect("usage present");
        assert_eq!(usage.cache_read_input_tokens, Some(25));
        assert_eq!(usage.reasoning_tokens, Some(50));
    }

    #[test]
    fn extract_responses_usage_returns_none_when_missing() {
        let response = json!({"id": "resp_x"});
        assert!(extract_openai_responses_usage(&response).is_none());
    }

    #[test]
    fn extract_responses_event_usage_reads_response_completed() {
        let event = json!({
            "type": "response.completed",
            "response": {
                "usage": {"input_tokens": 200, "output_tokens": 40}
            }
        });
        let usage = extract_openai_responses_event_usage(&event).expect("usage present");
        assert_eq!(usage.input_tokens, 200);
        assert_eq!(usage.output_tokens, 40);
    }

    #[test]
    fn extract_responses_event_usage_returns_none_for_other_events() {
        let event = json!({"type": "response.output_text.delta", "delta": "x"});
        assert!(extract_openai_responses_event_usage(&event).is_none());
    }

    #[test]
    fn chat_payload_includes_stream_options_include_usage_when_streaming() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let variants = build_chat_payload_variants("gpt-4.1-mini", &messages, true, None, false);
        let payload = &variants[0];
        assert_eq!(payload["stream"], true);
        assert_eq!(payload["stream_options"]["include_usage"], true);
    }

    #[test]
    fn chat_payload_without_stream_does_not_include_stream_options() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let variants = build_chat_payload_variants("gpt-4.1-mini", &messages, false, None, false);
        for variant in &variants {
            assert!(variant.get("stream_options").is_none());
        }
    }

    #[test]
    fn ollama_compat_chat_payload_omits_stream_options() {
        // Ollama does not support stream_options. Avoid sending it for Ollama-compat clients.
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let variants = build_chat_payload_variants("llama3.1", &messages, true, None, true);
        for variant in &variants {
            assert!(variant.get("stream_options").is_none());
        }
    }

    #[test]
    fn responses_payload_with_stream_includes_usage_in_response_completed() {
        // Responses API automatically includes usage in `response.completed`; no flag needed.
        // Sanity check: stream flag is set when requested.
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let variants = build_responses_payload_variants("gpt-4.1-mini", &messages, true, None);
        assert_eq!(variants[0]["stream"], true);
    }

    fn _ensure_token_usage_in_scope() -> TokenUsage {
        TokenUsage::default()
    }
}
