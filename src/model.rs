use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilitySkill {
    pub id: String,
    pub name: String,
    pub severity: String,
    pub description: String,
    pub prompt_fragment: String,
    pub examples: Vec<String>,
    pub false_positives: Vec<String>,
    pub references: Vec<String>,
    pub tags: Vec<String>,
    pub confidence_hint: Option<String>,
    pub guidance_markdown: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniPrompt {
    pub skill_id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillIterationResult {
    pub skill_id: String,
    pub status: String,
    pub findings: Vec<VulnerabilityFinding>,
    pub next_prompt: Option<MiniPrompt>,
    #[serde(default, skip_serializing_if = "TokenUsage::is_empty")]
    pub token_usage: TokenUsage,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
}

impl TokenUsage {
    pub fn is_empty(&self) -> bool {
        self.input_tokens == 0
            && self.output_tokens == 0
            && self.cache_read_input_tokens.is_none()
            && self.cache_creation_input_tokens.is_none()
            && self.reasoning_tokens.is_none()
    }

    pub fn add(&mut self, other: &TokenUsage) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cache_read_input_tokens =
            sum_optional(self.cache_read_input_tokens, other.cache_read_input_tokens);
        self.cache_creation_input_tokens = sum_optional(
            self.cache_creation_input_tokens,
            other.cache_creation_input_tokens,
        );
        self.reasoning_tokens = sum_optional(self.reasoning_tokens, other.reasoning_tokens);
    }
}

fn sum_optional(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.saturating_add(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityFinding {
    pub title: String,
    pub severity: String,
    pub summary: String,
    pub evidence: Vec<String>,
    pub recommendation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisStateJson {
    pub version: String,
    pub source_files: Vec<String>,
    pub provider: ProviderSpec,
    pub permission_prompt: PermissionPromptSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ast: Option<AstMetadata>,
    #[serde(default)]
    pub validator_context: ValidatorContextMap,
    pub iterations: Vec<SkillIterationResult>,
    #[serde(default, skip_serializing_if = "TokenUsage::is_empty")]
    pub total_token_usage: TokenUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstMetadata {
    pub path: String,
    pub fingerprint: String,
    pub generated_at: String,
    pub tool: AstToolMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstToolMetadata {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidatorContextMap {
    #[serde(default)]
    pub validators: Vec<ValidatorContextEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorContextEntry {
    pub id: String,
    pub module: String,
    pub source_file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_span: Option<SourceSpan>,
    pub handlers: Vec<ValidatorHandlerContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSpan {
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorHandlerContext {
    pub name: String,
    pub parameters: Vec<ValidatorParameterContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorParameterContext {
    pub name: String,
    pub r#type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderSpec {
    pub name: String,
    pub model: Option<String>,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionPromptSpec {
    pub shell: String,
    pub allowed_commands: Vec<String>,
    pub scope_rules: Vec<String>,
    #[serde(default = "default_workspace_root")]
    pub workspace_root: String,
    #[serde(default = "default_read_scope")]
    pub read_scope: String,
    #[serde(default)]
    pub interactive_permissions: bool,
    #[serde(default)]
    pub allowed_paths: Vec<String>,
}

fn default_workspace_root() -> String {
    ".".to_string()
}

fn default_read_scope() -> String {
    "workspace".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityReportSpec {
    pub title: String,
    pub generated_at: String,
    pub findings: Vec<VulnerabilityFinding>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_usage_default_is_empty() {
        let usage = TokenUsage::default();
        assert!(usage.is_empty());
    }

    #[test]
    fn token_usage_with_input_tokens_is_not_empty() {
        let usage = TokenUsage {
            input_tokens: 1,
            ..Default::default()
        };
        assert!(!usage.is_empty());
    }

    #[test]
    fn token_usage_with_output_tokens_is_not_empty() {
        let usage = TokenUsage {
            output_tokens: 1,
            ..Default::default()
        };
        assert!(!usage.is_empty());
    }

    #[test]
    fn token_usage_with_cache_field_is_not_empty() {
        let usage = TokenUsage {
            cache_read_input_tokens: Some(0),
            ..Default::default()
        };
        assert!(!usage.is_empty());
    }

    #[test]
    fn token_usage_add_sums_required_fields() {
        let mut a = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        let b = TokenUsage {
            input_tokens: 200,
            output_tokens: 25,
            ..Default::default()
        };
        a.add(&b);
        assert_eq!(a.input_tokens, 300);
        assert_eq!(a.output_tokens, 75);
    }

    #[test]
    fn token_usage_add_sums_optional_fields_when_both_present() {
        let mut a = TokenUsage {
            cache_read_input_tokens: Some(10),
            cache_creation_input_tokens: Some(5),
            reasoning_tokens: Some(7),
            ..Default::default()
        };
        let b = TokenUsage {
            cache_read_input_tokens: Some(20),
            cache_creation_input_tokens: Some(15),
            reasoning_tokens: Some(3),
            ..Default::default()
        };
        a.add(&b);
        assert_eq!(a.cache_read_input_tokens, Some(30));
        assert_eq!(a.cache_creation_input_tokens, Some(20));
        assert_eq!(a.reasoning_tokens, Some(10));
    }

    #[test]
    fn token_usage_add_keeps_present_when_other_is_none() {
        let mut a = TokenUsage {
            cache_read_input_tokens: Some(10),
            ..Default::default()
        };
        let b = TokenUsage::default();
        a.add(&b);
        assert_eq!(a.cache_read_input_tokens, Some(10));
    }

    #[test]
    fn token_usage_add_adopts_other_when_self_is_none() {
        let mut a = TokenUsage::default();
        let b = TokenUsage {
            cache_read_input_tokens: Some(42),
            ..Default::default()
        };
        a.add(&b);
        assert_eq!(a.cache_read_input_tokens, Some(42));
    }

    #[test]
    fn token_usage_skipped_when_empty_in_serialization() {
        let usage = TokenUsage::default();
        let json = serde_json::to_string(&usage).expect("serialize");
        // Default ints/options serialize to {input_tokens:0, output_tokens:0} but the
        // skip_serializing_if guard is checked at the parent level. Make sure
        // serializing a SkillIterationResult with empty usage omits the field.
        let iter = SkillIterationResult {
            skill_id: "s1".to_string(),
            status: "completed".to_string(),
            findings: vec![],
            next_prompt: None,
            token_usage: TokenUsage::default(),
        };
        let json_iter = serde_json::to_string(&iter).expect("serialize");
        assert!(!json_iter.contains("token_usage"));
        // raw struct still serializes to a non-empty json object (sanity)
        assert!(json.contains("input_tokens"));
    }

    #[test]
    fn skill_iteration_result_serializes_usage_when_non_empty() {
        let iter = SkillIterationResult {
            skill_id: "s1".to_string(),
            status: "completed".to_string(),
            findings: vec![],
            next_prompt: None,
            token_usage: TokenUsage {
                input_tokens: 123,
                output_tokens: 45,
                ..Default::default()
            },
        };
        let json = serde_json::to_string(&iter).expect("serialize");
        assert!(json.contains("\"token_usage\""));
        assert!(json.contains("\"input_tokens\":123"));
        assert!(json.contains("\"output_tokens\":45"));
    }
}
