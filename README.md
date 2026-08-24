# preflight

> ⚠️ **EXPERIMENTAL** — not yet ready for production audits. Findings should be reviewed by a human auditor.

`preflight` is a CLI tool that audits [Aiken](https://aiken-lang.org) smart contracts for vulnerabilities. It parses your validator code into an AST, then runs a configurable set of **vulnerability skills** against it using either local heuristics or an AI provider (OpenAI, Anthropic, Ollama). Output is a markdown report plus a structured JSON state file for tooling.

## How it works

1. **Source discovery** — recursively scans the working directory for `.ak` files (skipping `target/`, `build/`, `.git/`, `.tx3/`, and nested projects with their own `aiken.toml`).
2. **AST generation** — parses each source file with `aiken-lang`, builds a per-validator context (handlers, parameters, source spans), and caches the snapshot to `.tx3/audit/aiken-ast.json`.
3. **Skill loop** — for each vulnerability skill (markdown files with YAML frontmatter), invokes the configured provider with the skill prompt and AST context. The provider returns findings (or asks for additional file reads via a tool-call loop).
4. **Persistence** — after every skill iteration, the analysis state is written to `.tx3/audit/state.json` so partial progress survives interruptions.
5. **Report** — once all skills finish, a markdown report is rendered to `.tx3/audit/vulnerabilities.md`.

## Installation

### From a release binary

Download the binary for your platform from the [Releases page](https://github.com/tx3-lang/preflight/releases) and place it on your `PATH`.

### From source

```bash
git clone https://github.com/tx3-lang/preflight
cd preflight
cargo install --path .
```

## Usage

Run `preflight` from the root of an Aiken project:

```bash
preflight --provider openai
```

By default the tool writes:

| Path | Description |
| --- | --- |
| `.tx3/audit/state.json` | Incremental analysis state (one entry per skill iteration). |
| `.tx3/audit/vulnerabilities.md` | Final markdown report. |
| `.tx3/audit/aiken-ast.json` | Cached AST snapshot. |

### Providers

Select with `--provider`:

| Provider | Description | Required env var |
| --- | --- | --- |
| `scaffold` | No-op stub — useful for smoke testing the pipeline. | — |
| `heuristic` | Local pattern-matching detectors (no network). | — |
| `openai` | Uses the OpenAI Responses API. Default model: `gpt-4.1-mini`. | `OPENAI_API_KEY` |
| `anthropic` | Uses the Anthropic Messages API. Default model: `claude-sonnet-4-6`. | `ANTHROPIC_API_KEY` |
| `ollama` | OpenAI-compatible local Ollama endpoint. Default model: `llama3.1`. | — |

Override the endpoint, model, or API-key env var with `--endpoint`, `--model`, `--api-key-env`.

### Reasoning effort

`--reasoning-effort` hints to thinking-capable models how much to deliberate. Accepted values and wiring differ by provider:

| Provider | Accepted values | How it's wired |
| --- | --- | --- |
| `openai` | `low` \| `medium` \| `high` | Sets `reasoning.effort` (or `reasoning_effort`) on the Responses API request. Multiple request shapes are tried for compatibility. |
| `ollama` | `low` \| `medium` \| `high` | Same as OpenAI plus `think: true` for Ollama's reasoning toggle. |
| `anthropic` | `low` \| `medium` \| `high` \| `max` | Enables adaptive extended thinking (`thinking: {type: "adaptive", display: "summarized"}`) plus `output_config.effort`. `max` is Opus-tier only — Sonnet 4.6 rejects it; Haiku 4.5 rejects thinking entirely. |
| `scaffold`, `heuristic` | — | Ignored. |

If the API rejects the thinking variant (e.g., `max` on Sonnet, or any value on Haiku), the Anthropic provider falls back to a request without thinking automatically. With `--ai-logs` you'll see `⚠️ Streaming attempt 1 failed` followed by a successful retry.

### Examples

**Smoke test with no AI**

```bash
preflight --provider scaffold
```

**Local heuristic scan**

```bash
preflight --provider heuristic
```

**OpenAI with progress logs**

```bash
export OPENAI_API_KEY=sk-...
preflight --provider openai --ai-logs
```

**Anthropic with a specific model**

```bash
export ANTHROPIC_API_KEY=...
preflight --provider anthropic --model claude-opus-4-7
```

**Anthropic with adaptive thinking**

```bash
export ANTHROPIC_API_KEY=...
preflight --provider anthropic --reasoning-effort high --ai-logs

# Opus-tier maximum reasoning:
preflight --provider anthropic --model claude-opus-4-7 --reasoning-effort max
```

**Ollama (local)**

```bash
preflight --provider ollama --model llama3.1
```

### Permission scoping

When the AI provider asks to read additional files, requests are gated:

- `--read-scope workspace` (default) — any path under the project root may be read. Excluded directories (`.git/`, `target/`, `.tx3/`, `build/`, and nested projects) are always pruned and unreachable, for every scope.
- `--read-scope strict` — only the source files discovered up front are readable.
- `--interactive-permissions` — prompt the user before each AI-requested read.

## Vulnerability skills

Each skill is a markdown file with YAML frontmatter. Place additional skills in `skills/vulnerabilities/` (or pass `--skills-dir <path>`). Frontmatter schema:

```yaml
---
id: my-vuln-001                  # required, unique
name: my-vuln                    # required
severity: low|medium|high|critical
description: Short summary used by the audit prompt.
prompt_fragment: Concrete detection objective for the model.
examples: []                     # optional
false_positives: []              # optional — what NOT to flag
references: []                   # optional
tags: []                         # optional
confidence_hint: medium          # optional
---

# Detailed guidance (markdown body)

The body is passed to the model as authoritative guidance for this skill.
```

If `--skills-dir` is left at its default (`skills/vulnerabilities`) and the directory does not exist on disk, `preflight` falls back to three embedded seed skills.

### Built-in seed skills

| ID | Severity | What it detects |
| --- | --- | --- |
| `strict-value-equality-001` | high | Strict equality checks on ADA or full output values that can become unsatisfiable due to minUTxO / datum / reference-script changes. |
| `missing-address-validation-002` | high | Outputs selected by datum/token/value but never checked against the destination address. |
| `unvalidated-datum-003` | high | Outputs at script addresses whose datum is missing or never validated. |

## CLI reference

```
preflight [OPTIONS]

  --provider <PROVIDER>           scaffold | heuristic | openai | anthropic | ollama  [default: scaffold]
  --endpoint <URL>                Override the provider endpoint.
  --model <MODEL>                 Override the provider model.
  --api-key-env <VAR>             Override the env var read for the API key.
  --reasoning-effort <LEVEL>      Reasoning effort hint. OpenAI/Ollama: low|medium|high. Anthropic: low|medium|high|max (max is Opus-tier only; rejected by Sonnet/Haiku).
  --ai-logs                       Print chat-style progress while auditing.

  --skills-dir <PATH>             Vulnerability skills directory.            [default: skills/vulnerabilities]
  --state-out <PATH>              Incremental state JSON path.               [default: .tx3/audit/state.json]
  --report-out <PATH>             Final markdown report path.                [default: .tx3/audit/vulnerabilities.md]
  --ast-out <PATH>                AST snapshot JSON path.                    [default: .tx3/audit/aiken-ast.json]
  --no-ast-cache                  Force regeneration of the AST snapshot.

  --read-scope <SCOPE>            workspace | strict                         [default: workspace]
  --interactive-permissions       Confirm every AI-requested read.
  --main-source <PATH>            Fallback source file when no .ak files are discovered.
```

## Releases

Releases are produced via [`cargo-dist`](https://github.com/axodotdev/cargo-dist). The `Release` GitHub Action triggers on tag pushes matching `v*.*.*` and builds binaries for:

- `aarch64-apple-darwin`
- `x86_64-unknown-linux-gnu`
- `aarch64-unknown-linux-gnu`
- `x86_64-pc-windows-msvc`

Cutting a release locally:

```bash
# One-time
cargo install cargo-release

# From main, working tree clean
cargo release patch --execute        # 0.1.0 → 0.1.1
git push --follow-tags origin main
```

`cargo-release` config lives in `release.toml`. Tag pushes trigger `.github/workflows/release.yml`, which compiles the binaries and creates the GitHub Release.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
