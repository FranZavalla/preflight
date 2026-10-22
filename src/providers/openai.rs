use miette::{Context, IntoDiagnostic, Result};
use serde_json::{Value, json};
use std::path::Path;
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

/// Total attempts per payload variant, including the initial one.
const MAX_RATE_LIMIT_ATTEMPTS: usize = 3;
const RATE_LIMIT_BASE_DELAY: Duration = Duration::from_secs(5);
const RATE_LIMIT_MAX_DELAY: Duration = Duration::from_secs(60);
/// Total time one model turn may spend waiting between retries, shared across
/// every payload variant and across streaming/non-streaming.
const RATE_LIMIT_TOTAL_WAIT_BUDGET: Duration = Duration::from_secs(120);

/// What a caller should do next after a failed attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureKind {
    /// Nothing can succeed: exhausted credits, bad key, unroutable model.
    Fatal,
    /// The payload is fine; the account is sending too fast. Wait, resend the
    /// *same* payload — switching variants here would only thrash the cache.
    RateLimited,
    /// Transport-level trouble (timeout, reset, dropped stream, 5xx). Waiting
    /// and resending the same payload is worth trying.
    Transient,
    /// This payload shape was rejected; the next variant may work.
    NextVariant,
}

#[derive(Debug)]
struct AttemptFailure {
    report: miette::Report,
    kind: FailureKind,
    retry_after: Option<Duration>,
}

impl AttemptFailure {
    fn transient(report: miette::Report) -> Self {
        Self {
            report,
            kind: FailureKind::Transient,
            retry_after: None,
        }
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

fn transport_failure(error: reqwest::Error) -> AttemptFailure {
    // A stalled read is unlikely to recover by resending the same payload;
    // move on to the next variant instead of burning the skill budget on
    // repeated timeouts. Connection resets remain retryable transients.
    if error.is_timeout() {
        AttemptFailure {
            report: miette::miette!("request timed out: {error}"),
            kind: FailureKind::NextVariant,
            retry_after: None,
        }
    } else {
        AttemptFailure::transient(miette::miette!("request failed: {error}"))
    }
}

fn is_retryable(failure: &AttemptFailure) -> bool {
    matches!(
        failure.kind,
        FailureKind::RateLimited | FailureKind::Transient
    )
}

fn should_abort_turn(failure: &AttemptFailure) -> bool {
    matches!(failure.kind, FailureKind::Fatal)
}

/// Time budget shared by every retry inside one model turn.
struct RetryBudget {
    remaining: Duration,
}

impl RetryBudget {
    fn new(total: Duration) -> Self {
        Self { remaining: total }
    }

    fn take(&mut self, delay: Duration) -> bool {
        if delay > self.remaining {
            return false;
        }
        self.remaining -= delay;
        true
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

/// Converts an HTTP-date (IMF-fixdate) into a delay from now.
fn parse_http_date_delay(value: &str) -> Option<Duration> {
    let target = chrono::DateTime::parse_from_rfc2822(value)
        .ok()
        .map(|date| date.with_timezone(&chrono::Utc))
        .or_else(|| {
            chrono::NaiveDateTime::parse_from_str(value, "%a, %d %b %Y %H:%M:%S GMT")
                .ok()
                .map(|date| date.and_utc())
        })?;

    let seconds = target
        .signed_duration_since(chrono::Utc::now())
        .num_seconds();
    Some(Duration::from_secs(seconds.max(0) as u64))
}

/// Reads how long to wait before resending.
///
/// `retry-after` is either seconds or an HTTP-date. OpenRouter sends
/// `x-ratelimit-reset` instead, as the epoch instant the window reopens — in
/// milliseconds, though seconds and plain relative values show up across
/// gateways, so all three are accepted.
fn parse_retry_delay(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    if let Some(raw) = headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
    {
        if let Ok(seconds) = raw.parse::<u64>() {
            return Some(Duration::from_secs(seconds).min(RATE_LIMIT_MAX_DELAY));
        }
        if let Some(delay) = parse_http_date_delay(raw) {
            return Some(delay.min(RATE_LIMIT_MAX_DELAY));
        }
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

    Some(Duration::from_millis(millis).min(RATE_LIMIT_MAX_DELAY))
}

struct ProviderError {
    message: String,
    code: Option<String>,
    error_type: Option<String>,
}

fn json_scalar_to_string(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(ToString::to_string)
        .or_else(|| value.as_u64().map(|number| number.to_string()))
}

fn extract_provider_error(value: &Value) -> Option<ProviderError> {
    let event_type = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let explicit_event = event_type == "error" || event_type == "response.failed";

    let error_value = value
        .get("error")
        .filter(|error| !error.is_null())
        .or_else(|| {
            value
                .pointer("/response/error")
                .filter(|error| !error.is_null())
        });

    let (message, code, error_type) = match error_value {
        Some(Value::Object(error)) => {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| value.get("message").and_then(Value::as_str))
                .unwrap_or("provider reported an error")
                .trim()
                .to_string();
            let code = error
                .get("code")
                .and_then(json_scalar_to_string)
                .or_else(|| error.get("status").and_then(json_scalar_to_string));
            let error_type = error
                .get("type")
                .and_then(Value::as_str)
                .or_else(|| explicit_event.then_some(event_type))
                .map(ToString::to_string);

            // Some providers always include `error: {}` in healthy events.
            if message == "provider reported an error" && code.is_none() && error_type.is_none() {
                return None;
            }

            (message, code, error_type)
        }
        Some(Value::String(message)) => {
            let message = message.trim();
            if message.is_empty() && !explicit_event {
                return None;
            }
            (
                if message.is_empty() {
                    "provider reported an error".to_string()
                } else {
                    message.to_string()
                },
                None,
                explicit_event.then(|| event_type.to_string()),
            )
        }
        Some(_) => (
            "provider reported an error".to_string(),
            None,
            explicit_event.then(|| event_type.to_string()),
        ),
        None if explicit_event => (
            "provider reported an error".to_string(),
            None,
            Some(event_type.to_string()),
        ),
        None => return None,
    };

    Some(ProviderError {
        message,
        code,
        error_type,
    })
}

fn classify_error_code(value: &str) -> Option<FailureKind> {
    let normalized = value.trim().to_ascii_lowercase();

    if let Ok(status) = normalized.parse::<u16>() {
        return match status {
            401..=403 => Some(FailureKind::Fatal),
            408 => Some(FailureKind::Transient),
            429 => Some(FailureKind::RateLimited),
            500..=599 => Some(FailureKind::Transient),
            400..=499 => Some(FailureKind::NextVariant),
            _ => None,
        };
    }

    let fatal = [
        "insufficient_quota",
        "billing",
        "authentication",
        "permission_error",
        "invalid_api_key",
        "account_deactivated",
        "payment_required",
        "credit_balance",
    ];
    let rate_limited = ["rate_limit", "too_many_requests", "rate-limit"];
    let transient = [
        "server_error",
        "overloaded",
        "service_unavailable",
        "timeout",
        "temporarily_unavailable",
        "upstream_error",
    ];
    let next_variant = ["invalid_request", "bad_request", "unsupported"];

    if fatal.iter().any(|marker| normalized.contains(marker)) {
        Some(FailureKind::Fatal)
    } else if rate_limited
        .iter()
        .any(|marker| normalized.contains(marker))
    {
        Some(FailureKind::RateLimited)
    } else if transient.iter().any(|marker| normalized.contains(marker)) {
        Some(FailureKind::Transient)
    } else if next_variant
        .iter()
        .any(|marker| normalized.contains(marker))
    {
        Some(FailureKind::NextVariant)
    } else {
        None
    }
}

fn classify_error_message(message: &str) -> Option<FailureKind> {
    let lowered = message.to_ascii_lowercase();

    let fatal = [
        "credit balance",
        "insufficient credits",
        "insufficient_quota",
        "billing",
        "payment required",
        "invalid api key",
        "authentication",
        "permission denied",
        "account deactivated",
    ];
    let rate_limited = ["rate limit", "too many requests", "rate_limit"];
    let transient = [
        "overloaded",
        "server error",
        "service unavailable",
        "temporarily unavailable",
        "timeout",
    ];

    if fatal.iter().any(|marker| lowered.contains(marker)) {
        Some(FailureKind::Fatal)
    } else if rate_limited.iter().any(|marker| lowered.contains(marker)) {
        Some(FailureKind::RateLimited)
    } else if transient.iter().any(|marker| lowered.contains(marker)) {
        Some(FailureKind::Transient)
    } else {
        None
    }
}

fn classify_provider_error(error: &ProviderError, retry_after: Option<Duration>) -> AttemptFailure {
    let kind = error
        .code
        .as_deref()
        .and_then(classify_error_code)
        .or_else(|| error.error_type.as_deref().and_then(classify_error_code))
        .or_else(|| classify_error_message(&error.message))
        .unwrap_or(FailureKind::NextVariant);

    AttemptFailure {
        report: miette::miette!("AI provider error: {}", error.message),
        kind,
        retry_after,
    }
}

/// Detects provider errors delivered inside a successful (200) response, both
/// SSE `error` events and JSON bodies with an `/error` pointer.
fn classify_in_band_error(value: &Value, retry_after: Option<Duration>) -> Option<AttemptFailure> {
    let error = extract_provider_error(value)?;
    Some(classify_provider_error(&error, retry_after))
}

fn classify_request_failure(
    label: &str,
    status: reqwest::StatusCode,
    body: &str,
    retry_after: Option<Duration>,
) -> AttemptFailure {
    // 402 is how a gateway reports an empty balance; 401/403 are key and routing
    // problems. No payload variant and no amount of waiting fixes any of them.
    if matches!(
        status,
        reqwest::StatusCode::UNAUTHORIZED
            | reqwest::StatusCode::PAYMENT_REQUIRED
            | reqwest::StatusCode::FORBIDDEN
    ) {
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
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
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

/// Returns the delay to use before the next attempt, or `None` when the failure
/// is not retryable.
fn retry_delay_after_failure(failure: &AttemptFailure, retry: usize) -> Option<Duration> {
    if !is_retryable(failure) {
        return None;
    }

    Some(
        failure
            .retry_after
            .unwrap_or_else(|| RATE_LIMIT_BASE_DELAY * 2u32.pow(retry.min(31) as u32))
            .min(RATE_LIMIT_MAX_DELAY),
    )
}

/// Runs one payload variant, waiting out retryable failures instead of falling
/// through to a different variant or failing the skill.
async fn attempt_with_backoff<'a, F, Fut>(
    ai_logs: bool,
    label: &str,
    attempt_idx: usize,
    budget: &mut RetryBudget,
    deadline: Instant,
    run: F,
) -> std::result::Result<(String, TokenUsage), AttemptFailure>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<(String, TokenUsage), AttemptFailure>>
        + 'a,
{
    let mut last = match run().await {
        Ok(value) => return Ok(value),
        Err(failure) => failure,
    };

    for retry in 0..MAX_RATE_LIMIT_ATTEMPTS.saturating_sub(1) {
        // Non-retryable failures return quietly; only a retryable failure that
        // cannot fit in the remaining budget should report budget exhaustion.
        let Some(delay) = retry_delay_after_failure(&last, retry) else {
            return Err(last);
        };

        if Instant::now() + delay >= deadline {
            log_agent_progress(
                ai_logs,
                format!(
                    "⏳ {} attempt {} hit the skill deadline; giving up",
                    label,
                    attempt_idx + 1
                ),
            );
            return Err(last);
        }

        if !budget.take(delay) {
            log_agent_progress(
                ai_logs,
                format!(
                    "⏳ {} attempt {} stopped retrying (turn budget {}s exhausted); giving up",
                    label,
                    attempt_idx + 1,
                    RATE_LIMIT_TOTAL_WAIT_BUDGET.as_secs()
                ),
            );
            return Err(last);
        }

        log_agent_progress(
            ai_logs,
            format!(
                "⏳ {} attempt {} rate limited or transient; waiting {}s before retry {}/{}",
                label,
                attempt_idx + 1,
                delay.as_secs(),
                retry + 1,
                MAX_RATE_LIMIT_ATTEMPTS.saturating_sub(1)
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

async fn send_request(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    payload: &Value,
) -> std::result::Result<reqwest::Response, AttemptFailure> {
    client
        .post(endpoint)
        .bearer_auth(api_key)
        .json(payload)
        .send()
        .await
        .map_err(transport_failure)
}

/// Checks the HTTP status and preserves why the gateway refused, so the caller
/// can decide whether waiting or another payload can help.
async fn ensure_success(
    response: reqwest::Response,
    label: &str,
) -> std::result::Result<(reqwest::Response, Option<Duration>), AttemptFailure> {
    let status = response.status();
    let retry_after = parse_retry_delay(response.headers());

    if status.is_success() {
        return Ok((response, retry_after));
    }

    let body = response.text().await.map_err(transport_failure)?;
    Err(classify_request_failure(label, status, &body, retry_after))
}

async fn run_attempt(
    api_family: ApiFamily,
    stream: bool,
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    payload: &Value,
    ai_logs: bool,
) -> std::result::Result<(String, TokenUsage), AttemptFailure> {
    match (api_family, stream) {
        (ApiFamily::ChatCompletions, true) => {
            stream_chat_attempt(client, endpoint, api_key, payload, ai_logs).await
        }
        (ApiFamily::Responses, true) => {
            stream_responses_attempt(client, endpoint, api_key, payload, ai_logs).await
        }
        (ApiFamily::ChatCompletions, false) => {
            non_stream_chat_attempt(client, endpoint, api_key, payload, ai_logs).await
        }
        (ApiFamily::Responses, false) => {
            non_stream_responses_attempt(client, endpoint, api_key, payload, ai_logs).await
        }
    }
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
) -> std::result::Result<(String, TokenUsage), AttemptFailure> {
    let response = send_request(client, endpoint, api_key, payload).await?;
    let (mut response, retry_after_hint) = ensure_success(response, "Streaming request").await?;

    let mut pending = String::new();
    let mut model_output = String::new();
    let mut reasoning_stream_state = ReasoningStreamState::default();
    let mut content_stream_state = ContentStreamState::default();
    let mut usage = TokenUsage::default();
    let mut stream_error: Option<AttemptFailure> = None;

    'stream: while let Some(chunk) = response.chunk().await.map_err(transport_failure)? {
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

            if let Some(failure) = classify_in_band_error(&event, retry_after_hint) {
                stream_error = Some(failure);
                break 'stream;
            }

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

    if let Some(failure) = stream_error {
        return Err(failure);
    }

    if model_output.is_empty() {
        return Err(AttemptFailure::transient(miette::miette!(
            "Streaming response did not include content deltas"
        )));
    }

    Ok((model_output, usage))
}

async fn stream_responses_attempt(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    payload: &Value,
    ai_logs: bool,
) -> std::result::Result<(String, TokenUsage), AttemptFailure> {
    let response = send_request(client, endpoint, api_key, payload).await?;
    let (mut response, retry_after_hint) = ensure_success(response, "Streaming request").await?;

    let mut pending = String::new();
    let mut model_output = String::new();
    let mut reasoning_stream_state = ReasoningStreamState::default();
    let mut content_stream_state = ContentStreamState::default();
    let mut usage = TokenUsage::default();
    let mut stream_error: Option<AttemptFailure> = None;

    'stream: while let Some(chunk) = response.chunk().await.map_err(transport_failure)? {
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

            if let Some(failure) = classify_in_band_error(&event, retry_after_hint) {
                stream_error = Some(failure);
                break 'stream;
            }

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

    if let Some(failure) = stream_error {
        return Err(failure);
    }

    if model_output.is_empty() {
        return Err(AttemptFailure::transient(miette::miette!(
            "Streaming response did not include output text deltas"
        )));
    }

    Ok((model_output, usage))
}

async fn non_stream_chat_attempt(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    payload: &Value,
    ai_logs: bool,
) -> std::result::Result<(String, TokenUsage), AttemptFailure> {
    let response = send_request(client, endpoint, api_key, payload).await?;
    let (response, retry_after_hint) = ensure_success(response, "Non-stream request").await?;
    let response_json = response.json::<Value>().await.map_err(transport_failure)?;

    if let Some(failure) = classify_in_band_error(&response_json, retry_after_hint) {
        return Err(failure);
    }

    let content = response_json
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| miette::miette!("AI provider returned an unexpected response payload"))?;

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
) -> std::result::Result<(String, TokenUsage), AttemptFailure> {
    let response = send_request(client, endpoint, api_key, payload).await?;
    let (response, retry_after_hint) = ensure_success(response, "Non-stream request").await?;
    let response_json = response.json::<Value>().await.map_err(transport_failure)?;

    if let Some(failure) = classify_in_band_error(&response_json, retry_after_hint) {
        return Err(failure);
    }

    let content = extract_responses_output_text(&response_json)
        .ok_or_else(|| miette::miette!("AI provider returned an unexpected response payload"))?;

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

        let skill_deadline = Instant::now() + super::shared::DEFAULT_SKILL_TIMEOUT;

        run_agent_loop(
            skill,
            super::shared::AgentLoopContext {
                endpoint: &self.endpoint,
                ai_logs: self.ai_logs,
                project_root: &canonical_root,
                permission_prompt,
                provider_label: "AI provider",
                skill_timeout: super::shared::DEFAULT_SKILL_TIMEOUT,
            },
            &mut messages,
            |messages| {
                block_on_runtime_aware(async {
                    let client = reqwest::Client::builder()
                        .read_timeout(super::shared::DEFAULT_REQUEST_READ_TIMEOUT)
                        .connect_timeout(super::shared::DEFAULT_REQUEST_CONNECT_TIMEOUT)
                        .build()
                        .into_diagnostic()?;
                    let reasoning_effort = self.reasoning_effort.as_deref();
                    let mut budget = RetryBudget::new(RATE_LIMIT_TOTAL_WAIT_BUDGET);

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
                                &mut budget,
                                skill_deadline,
                                || {
                                    run_attempt(
                                        api_family,
                                        true,
                                        &client,
                                        &self.endpoint,
                                        &self.api_key,
                                        stream_payload,
                                        self.ai_logs,
                                    )
                                },
                            )
                            .await
                            {
                                Ok(content) => return Ok(content),
                                // Account-level failures cannot be fixed by another
                                // payload variant or by falling back to non-streaming.
                                Err(failure) if should_abort_turn(&failure) => {
                                    return Err(failure.report);
                                }
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
                            &mut budget,
                            skill_deadline,
                            || {
                                run_attempt(
                                    api_family,
                                    false,
                                    &client,
                                    &self.endpoint,
                                    &self.api_key,
                                    payload,
                                    self.ai_logs,
                                )
                            },
                        )
                        .await
                        {
                            Ok(content) => return Ok(content),
                            // Account-level failures cannot be fixed by another
                            // payload variant or by a non-stream retry.
                            Err(failure) if should_abort_turn(&failure) => {
                                return Err(failure.report);
                            }
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
    use std::sync::atomic::{AtomicUsize, Ordering};

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
    fn a_reset_already_in_the_past_retries_immediately() {
        let elapsed_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_millis() as u64
            - 5_000;

        assert_eq!(
            parse_retry_delay(&headers_with("x-ratelimit-reset", elapsed_ms.to_string())),
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn retry_after_http_date_is_supported() {
        let future = chrono::Utc::now() + chrono::Duration::seconds(30);
        let value = future.format("%a, %d %b %Y %H:%M:%S GMT").to_string();

        let delay = parse_retry_delay(&headers_with("retry-after", value)).expect("a delay");
        assert!(
            delay <= Duration::from_secs(30) && delay >= Duration::from_secs(25),
            "unexpected delay: {:?}",
            delay
        );
    }

    #[test]
    fn retry_after_in_the_past_retries_immediately() {
        let past = chrono::Utc::now() - chrono::Duration::seconds(30);
        let value = past.format("%a, %d %b %Y %H:%M:%S GMT").to_string();

        assert_eq!(
            parse_retry_delay(&headers_with("retry-after", value)),
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn throttling_is_classified_as_rate_limited() {
        let failure = classify_request_failure(
            "Streaming request",
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"message":"Rate limit exceeded: new-account-rpm"}}"#,
            Some(Duration::from_secs(9)),
        );

        assert_eq!(failure.kind, FailureKind::RateLimited);
        assert_eq!(failure.retry_after, Some(Duration::from_secs(9)));
        assert!(!should_abort_turn(&failure));
        assert!(failure.report.to_string().contains("new-account-rpm"));
    }

    #[test]
    fn an_empty_balance_is_fatal() {
        let failure = classify_request_failure(
            "Non-stream request",
            reqwest::StatusCode::PAYMENT_REQUIRED,
            r#"{"error":{"message":"Insufficient credits"}}"#,
            None,
        );

        assert!(should_abort_turn(&failure));
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
    async fn retries_a_rate_limited_request_without_switching_variants() {
        let attempts = AtomicUsize::new(0);
        let mut budget = RetryBudget::new(Duration::from_secs(120));

        let result = attempt_with_backoff(
            false,
            "Test",
            0,
            &mut budget,
            Instant::now() + Duration::from_secs(120),
            || {
                let attempt = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                async move {
                    if attempt < 3 {
                        Err(AttemptFailure {
                            report: miette::miette!("429 Too Many Requests"),
                            kind: FailureKind::RateLimited,
                            retry_after: Some(Duration::ZERO),
                        })
                    } else {
                        Ok(("done".to_string(), TokenUsage::default()))
                    }
                }
            },
        )
        .await
        .expect("should recover after retries");

        assert_eq!(result.0, "done");
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn fatal_failures_are_not_retried() {
        let attempts = AtomicUsize::new(0);
        let mut budget = RetryBudget::new(Duration::from_secs(120));

        let error = attempt_with_backoff(
            false,
            "Test",
            0,
            &mut budget,
            Instant::now() + Duration::from_secs(120),
            || {
                attempts.fetch_add(1, Ordering::SeqCst);
                async {
                    Err(AttemptFailure {
                        report: miette::miette!("credit balance is too low"),
                        kind: FailureKind::Fatal,
                        retry_after: None,
                    })
                }
            },
        )
        .await
        .expect_err("fatal failures should stop immediately");

        assert!(error.report.to_string().contains("credit balance"));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn skill_deadline_stops_retries() {
        let attempts = AtomicUsize::new(0);
        let mut budget = RetryBudget::new(Duration::from_secs(120));
        let deadline = Instant::now() - Duration::from_secs(1);

        let error = attempt_with_backoff(false, "Test", 0, &mut budget, deadline, || {
            attempts.fetch_add(1, Ordering::SeqCst);
            async {
                Err(AttemptFailure {
                    report: miette::miette!("429 Too Many Requests"),
                    kind: FailureKind::RateLimited,
                    retry_after: Some(Duration::ZERO),
                })
            }
        })
        .await
        .expect_err("the skill deadline should stop retries");

        assert!(error.report.to_string().contains("429"));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn retry_budget_is_shared_across_calls() {
        let mut budget = RetryBudget::new(Duration::from_secs(10));

        assert!(budget.take(Duration::from_secs(6)));
        assert!(!budget.take(Duration::from_secs(6)));
        assert!(budget.take(Duration::from_secs(4)));
    }

    #[tokio::test(start_paused = true)]
    async fn shared_budget_is_consumed_across_variants() {
        let mut budget = RetryBudget::new(Duration::from_secs(120));
        let first_attempts = AtomicUsize::new(0);

        // Two 60s waits consume the whole turn budget.
        let first = attempt_with_backoff(
            false,
            "Variant 1",
            0,
            &mut budget,
            Instant::now() + Duration::from_secs(120),
            || {
                first_attempts.fetch_add(1, Ordering::SeqCst);
                async {
                    Err(AttemptFailure {
                        report: miette::miette!("429 Too Many Requests"),
                        kind: FailureKind::RateLimited,
                        retry_after: Some(Duration::from_secs(60)),
                    })
                }
            },
        )
        .await;

        assert!(first.is_err());
        assert_eq!(
            first_attempts.load(Ordering::SeqCst),
            MAX_RATE_LIMIT_ATTEMPTS
        );

        let second_attempts = AtomicUsize::new(0);
        let second = attempt_with_backoff(
            false,
            "Variant 2",
            0,
            &mut budget,
            Instant::now() + Duration::from_secs(120),
            || {
                second_attempts.fetch_add(1, Ordering::SeqCst);
                async {
                    Err(AttemptFailure {
                        report: miette::miette!("429 Too Many Requests"),
                        kind: FailureKind::RateLimited,
                        retry_after: Some(Duration::from_secs(10)),
                    })
                }
            },
        )
        .await;

        assert!(second.is_err());
        assert_eq!(
            second_attempts.load(Ordering::SeqCst),
            1,
            "the second variant must see the exhausted turn budget"
        );
    }

    #[test]
    fn retry_delay_is_capped_per_attempt() {
        let failure = AttemptFailure {
            report: miette::miette!("429 Too Many Requests"),
            kind: FailureKind::RateLimited,
            retry_after: Some(Duration::from_secs(121)),
        };

        assert_eq!(
            retry_delay_after_failure(&failure, 0),
            Some(Duration::from_secs(60))
        );
    }

    #[test]
    fn transient_failures_are_retryable() {
        let failure = AttemptFailure::transient(miette::miette!("connection reset"));

        assert_eq!(failure.kind, FailureKind::Transient);
        assert!(retry_delay_after_failure(&failure, 0).is_some());
    }

    #[test]
    fn next_variant_failures_are_not_retryable() {
        let failure = AttemptFailure {
            report: miette::miette!("unsupported parameter"),
            kind: FailureKind::NextVariant,
            retry_after: None,
        };

        assert!(retry_delay_after_failure(&failure, 0).is_none());
    }

    #[test]
    fn fatal_failures_abort_the_turn() {
        let fatal = AttemptFailure {
            report: miette::miette!("no credits"),
            kind: FailureKind::Fatal,
            retry_after: None,
        };
        let rate_limited = AttemptFailure {
            report: miette::miette!("429"),
            kind: FailureKind::RateLimited,
            retry_after: None,
        };
        let transient = AttemptFailure::transient(miette::miette!("timeout"));
        let next_variant = AttemptFailure {
            report: miette::miette!("bad payload"),
            kind: FailureKind::NextVariant,
            retry_after: None,
        };

        assert!(should_abort_turn(&fatal));
        assert!(!should_abort_turn(&rate_limited));
        assert!(!should_abort_turn(&transient));
        assert!(!should_abort_turn(&next_variant));
    }

    #[test]
    fn in_band_rate_limit_is_retryable() {
        let event = json!({
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error"
            }
        });

        let failure = classify_in_band_error(&event, Some(Duration::from_secs(7)))
            .expect("should detect in-band error");

        assert_eq!(failure.kind, FailureKind::RateLimited);
        assert_eq!(failure.retry_after, Some(Duration::from_secs(7)));
    }

    #[test]
    fn in_band_credit_exhaustion_is_fatal() {
        let event = json!({
            "error": {
                "message": "Insufficient credits"
            }
        });

        let failure = classify_in_band_error(&event, None).expect("should detect in-band error");

        assert!(should_abort_turn(&failure));
    }

    #[test]
    fn responses_failed_event_is_detected() {
        let event = json!({
            "type": "response.failed",
            "response": {
                "error": {
                    "message": "upstream overloaded",
                    "code": "overloaded_error"
                }
            }
        });

        let failure = classify_in_band_error(&event, None).expect("should detect failure event");

        assert_eq!(failure.kind, FailureKind::Transient);
        assert!(is_retryable(&failure));
    }

    #[test]
    fn numeric_error_code_429_is_rate_limited() {
        let event = json!({
            "error": {
                "message": "slow down",
                "code": 429
            }
        });

        let failure = classify_in_band_error(&event, None).expect("should detect in-band error");
        assert_eq!(failure.kind, FailureKind::RateLimited);
    }

    #[test]
    fn structured_insufficient_quota_is_fatal() {
        let event = json!({
            "error": {
                "message": "quota exhausted",
                "code": "insufficient_quota"
            }
        });

        let failure = classify_in_band_error(&event, None).expect("should detect in-band error");
        assert!(should_abort_turn(&failure));
    }

    #[test]
    fn insufficient_capacity_is_not_fatal() {
        let event = json!({
            "error": {
                "message": "insufficient capacity right now"
            }
        });

        let failure = classify_in_band_error(&event, None).expect("should detect in-band error");
        assert!(!should_abort_turn(&failure));
        assert_eq!(failure.kind, FailureKind::NextVariant);
    }

    #[test]
    fn ordinary_stream_events_are_not_errors() {
        let event = json!({"choices": [{"delta": {"content": "{}"}}]});
        assert!(classify_in_band_error(&event, None).is_none());
    }

    #[test]
    fn null_error_fields_are_ignored() {
        let event = json!({"choices": [], "error": null});
        assert!(classify_in_band_error(&event, None).is_none());
    }

    #[test]
    fn empty_error_objects_are_ignored() {
        let event = json!({"choices": [], "error": {}});
        assert!(classify_in_band_error(&event, None).is_none());
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
