use miette::{Context, IntoDiagnostic, Result};
use serde_json::{Value, json};
use std::path::Path;

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

const DEFAULT_MAX_TOKENS: u32 = 1200;
const THINKING_MAX_TOKENS: u32 = 1600;
const THINKING_BUDGET_TOKENS: u32 = 1024;
const ADAPTIVE_THINKING_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub version: String,
    pub ai_logs: bool,
    pub reasoning_effort: Option<String>,
}

fn build_anthropic_payload_variants(
    model: &str,
    system_prompt: &str,
    messages: &[Value],
    stream: bool,
    reasoning_effort: Option<&str>,
) -> Vec<Value> {
    let normalized_messages = normalize_anthropic_messages(messages);

    let mut base = json!({
        "model": model,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "system": system_prompt,
        "messages": normalized_messages,
    });

    if stream {
        base["stream"] = Value::Bool(true);
    }

    if let Some(effort) = reasoning_effort {
        let mut with_adaptive = base.clone();
        with_adaptive["max_tokens"] = Value::from(ADAPTIVE_THINKING_MAX_TOKENS);
        with_adaptive["thinking"] = json!({
            "type": "adaptive",
            "display": "summarized"
        });
        with_adaptive["output_config"] = json!({
            "effort": effort
        });

        return vec![with_adaptive, base];
    }

    let mut with_thinking = base.clone();
    with_thinking["max_tokens"] = Value::from(THINKING_MAX_TOKENS);
    with_thinking["thinking"] = json!({
        "type": "enabled",
        "budget_tokens": THINKING_BUDGET_TOKENS
    });

    vec![with_thinking, base]
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

async fn stream_attempt(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    version: &str,
    payload: &Value,
    ai_logs: bool,
) -> Result<(String, TokenUsage)> {
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
        let body = response.text().await.into_diagnostic()?;
        return Err(miette::miette!(
            "Streaming request failed with status {}: {}",
            status,
            body
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

            update_anthropic_usage_from_event(&event, &mut usage);

            if let Some(reasoning_delta) = extract_anthropic_reasoning_delta(&event) {
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

    if model_output.is_empty() {
        return Err(miette::miette!(
            "Streaming response did not include output text deltas"
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
) -> Result<(String, TokenUsage)> {
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
        let body = response.text().await.into_diagnostic()?;
        return Err(miette::miette!(
            "Request failed with status {}: {}",
            status,
            body
        ));
    }

    let response_json = response.json::<Value>().await.into_diagnostic()?;

    let content = extract_anthropic_non_stream_content(&response_json).ok_or_else(|| {
        miette::miette!("Anthropic provider returned an unexpected response payload")
    })?;

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

                    if self.ai_logs {
                        let mut last_stream_error: Option<String> = None;
                        let stream_payloads = build_anthropic_payload_variants(
                            &self.model,
                            system_prompt,
                            messages,
                            true,
                            self.reasoning_effort.as_deref(),
                        );

                        for (attempt_idx, payload) in stream_payloads.iter().enumerate() {
                            let stream_result = stream_attempt(
                                &client,
                                &self.endpoint,
                                &self.api_key,
                                &self.version,
                                payload,
                                self.ai_logs,
                            )
                            .await;

                            match stream_result {
                                Ok(content) => return Ok(content),
                                Err(error) => {
                                    last_stream_error = Some(error.to_string());
                                    log_agent_progress(
                                        self.ai_logs,
                                        format!(
                                            "⚠️ Streaming attempt {} failed: {}",
                                            attempt_idx + 1,
                                            error
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
                        let request_result = non_stream_attempt(
                            &client,
                            &self.endpoint,
                            &self.api_key,
                            &self.version,
                            payload,
                            self.ai_logs,
                        )
                        .await;

                        match request_result {
                            Ok(content) => return Ok(content),
                            Err(error) => {
                                last_non_stream_error = Some(error.to_string());
                                log_agent_progress(
                                    self.ai_logs,
                                    format!(
                                        "⚠️ Non-stream attempt {} failed: {}",
                                        attempt_idx + 1,
                                        error
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
    fn payload_variants_without_reasoning_effort_use_legacy_thinking() {
        let messages = sample_messages();
        let variants = build_anthropic_payload_variants(
            "claude-sonnet-4-6",
            "system",
            &messages,
            false,
            None,
        );

        assert_eq!(variants.len(), 2);

        let first = &variants[0];
        assert_eq!(first["model"], "claude-sonnet-4-6");
        assert_eq!(first["max_tokens"], THINKING_MAX_TOKENS);
        assert_eq!(first["thinking"]["type"], "enabled");
        assert_eq!(first["thinking"]["budget_tokens"], THINKING_BUDGET_TOKENS);
        assert!(first.get("output_config").is_none());

        let fallback = &variants[1];
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

        assert_eq!(variants.len(), 2);

        let first = &variants[0];
        assert_eq!(first["max_tokens"], ADAPTIVE_THINKING_MAX_TOKENS);
        assert_eq!(first["thinking"]["type"], "adaptive");
        assert_eq!(first["thinking"]["display"], "summarized");
        assert_eq!(first["output_config"]["effort"], "high");
        assert!(first["thinking"].get("budget_tokens").is_none());

        let fallback = &variants[1];
        assert_eq!(fallback["max_tokens"], DEFAULT_MAX_TOKENS);
        assert!(fallback.get("thinking").is_none());
        assert!(fallback.get("output_config").is_none());
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
        let variants = build_anthropic_payload_variants(
            "claude-sonnet-4-6",
            "system",
            &messages,
            false,
            None,
        );

        let first_message = &variants[0]["messages"][0];
        assert_eq!(first_message["role"], "user");
        assert_eq!(first_message["content"][0]["type"], "text");
        assert_eq!(first_message["content"][0]["text"], "hello");
    }
}
