# AGENTS.md — Agent & Developer Quick Reference

This file serves as a single source of truth for AI agents and developers working on the `Agent Harness` codebase. Refer to this document to understand project structure, key components, execution requirements, and how to run commands without searching through files.

---

## 1. Python Environment & Execution Guide

### The Python Path & `.venv` Setup
- **Windows System Python**: `python.exe` (Python 3.13 at `C:\Python313\python.exe`).
- **Dependencies Directory**: `.venv\Lib\site-packages` (contains `groq`, `sentence-transformers`, `torch`, `pydantic`, `dotenv`, etc.).
- **Why `.venv\Scripts\python.exe` fails**:
  The virtual environment was initialized via `uv`. In this sandbox environment, running `.venv\Scripts\python.exe` directly emits `Failed to find real location of C:\Python313\python.exe`.
- **The Reliable Execution Method**:
  Run standard `python` with `PYTHONPATH` set to include both `src` and `.venv\Lib\site-packages`:

```powershell
$env:PYTHONPATH="src;.venv\Lib\site-packages"
python <script.py>
```

> **Note**: If you see a one-line warning `Failed to find real location of C:\Python313\python.exe`, it is harmless — the script will continue execution normally as long as `PYTHONPATH` is configured as shown above.

---

## 2. Common Commands Cheat Sheet

| Task | Command (PowerShell) |
|---|---|
| **Run Streamlit Dashboard** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python -m streamlit run dashboard.py` |
| **Clear Semantic Cache** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python clear_cache.py` |
| **Run Dashboard Unit Test** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python tests/test_dashboard.py` |
| **Run Interactive CLI** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python chat.py` |
| **Run CLI (Verbose Mode)** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python chat.py --verbose` |
| **Run Validator Unit Tests** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python tests/test_validators.py` |
| **Run Stalled Loop Unit Test** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python tests/test_stalled_loop.py` |
| **Run Retry Feedback Unit Test** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python tests/test_retry_feedback.py` |
| **Run LLM Clients Unit Test** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python tests/test_llm_clients.py` |
| **Run Evaluate Metrics Unit Test** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python tests/test_evaluate.py` |
| **Run Benchmark Integrity Test** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python tests/test_samples.py` |
| **Run Challenges Integrity Test** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python tests/test_challenges.py` |
| **Generate Challenges Dataset** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python -m agentic_harness.benchmark.make_challenges` |
| **Run Benchmark Suite (Tune)** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python -m agentic_harness.benchmark.run_bench --suite benchmark --split tune --mode harness` |
| **Run Challenge Suite (Test)** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python -m agentic_harness.benchmark.run_bench --suite challenge --split test --mode harness` |
| **Run Full Evaluation** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python evaluate.py` |
| **Run Evaluation (Smoke Test)** | `$env:PYTHONPATH="src;.venv\Lib\site-packages"; python evaluate.py --suite benchmark --limit 5` |

---

## 3. Project Architecture & File Map

```
Agent Harness/
├── AGENTS.md                               # This reference guide
├── chat.py                                 # Interactive CLI entry point
├── clear_cache.py                          # Utility script to wipe semantic cache table
├── dashboard.py                            # Streamlit multi-page analytics & live testing UI
├── evaluate.py                             # Evaluation runner: baseline vs harness, summary table
├── config.yaml                             # Provider & model configuration (generator vs. judge)
├── pyproject.toml                          # Project configuration and dependencies
├── .env                                    # Environment variables (GROQ_API_KEY)
├── agent_harness.db                        # SQLite database (runs, cache)
├── results/
│   ├── summary.json                        # Comprehensive metrics JSON output
│   └── summary.md                          # Summary markdown report with comparison table
├── src/agentic_harness/
│   ├── harness.py                          # Core engine: retry loops, scoring, stalled-loop exit
│   ├── llm/                                # Multi-provider LLM abstraction
│   │   ├── base.py                         # Abstract LLMClient and standardized LLMResult
│   │   ├── ollama_client.py                # Local Ollama client (generator)
│   │   ├── groq_client.py                  # Groq cloud client (judge)
│   │   └── __init__.py                     # get_client(role) factory from config.yaml
│   ├── constraint_extractor.py             # LLM-based constraint extraction (uses judge)
│   ├── validators.py                       # Deterministic checkers & robust sentence splitter
│   ├── critic.py                           # LLM-as-a-judge critique parser & prompt (uses judge)
│   ├── semantic.py                         # Embeddings and cosine similarity calculations
│   ├── db.py                               # SQLite persistence (runs log & vector cache)
│   ├── cache.py                            # Semantic cache lookup (>0.90 similarity)
│   └── benchmark/
│       ├── run_bench.py                    # Benchmark runner with --suite and --split flags
│       ├── make_challenges.py              # Programmatic generator for 200+ harder prompts
│       ├── samples.json                    # 54 benchmark samples across 6 categories (60/40 split)
│       └── challenges.json                 # 225 challenge samples (60/40 split)
└── tests/
    ├── test_validators.py                  # Unit tests for sentence splitter & checkers
    ├── test_stalled_loop.py                # Unit tests for stalled retry loop detection
    ├── test_retry_feedback.py              # Unit tests for 1.5B feedback, temp scaling, judge prompts
    ├── test_llm_clients.py                 # Unit tests for multi-provider client abstraction
    ├── test_evaluate.py                    # Unit tests for evaluation metrics & table formatting
    ├── test_dashboard.py                   # Unit tests for dashboard data aggregation & loaders
    ├── test_samples.py                     # Unit tests for samples.json integrity
    ├── test_challenges.py                  # Unit tests for challenges.json integrity
    └── test_benchmark_cli.py               # Unit tests for run_bench suite/split loader
```

---

## 4. Key Modules & Datasets

### Datasets: `samples.json` & `challenges.json`
- **`samples.json`**: Exactly 54 samples across 6 categories (9 per category):
  1. `json_extraction` (s001–s009)
  2. `arithmetic_word_problems` (s010–s018)
  3. `format_constraints` (s019–s027)
  4. `factual_qa` (s028–s036)
  5. `instruction_following` (s037–s045)
  6. `length_limited_summarization` (s046–s054)
  - Split: 32 `tune` (59.3%), 22 `test` (40.7%).
- **`make_challenges.py` & `challenges.json`**:
  - Programmatically generates 225 challenge samples from templates: conflicting constraints (80), nested formats (60), tricky edge cases (45), and paraphrased variants (40).
  - Split: 135 `tune` (60.0%), 90 `test` (40.0%).

### `benchmark/run_bench.py` CLI Flags
- `--suite benchmark|challenge`: Selects `samples.json` (default) or `challenges.json`.
- `--split tune|test`: Filters dataset by split. If omitted, runs all samples.
- `--mode baseline|harness`: Single LLM call vs agentic retry harness.
- `--max-retries N`: Correction turns per sample.
- `--no-cache`: Disables semantic caching.

### `src/agentic_harness/harness.py`
- `answer(query, constraints=None, ...)`: Answering pipeline without reference answers.
- `run(...)`: Sample-based evaluation against reference answers.
- `combine(...)`: Reliability scoring combining deterministic, semantic relevance, and critic scores.
- `build_retry_message(...)`: Formats short, direct feedback for 1.5B generator (caps at 3 errors; critic issues only if no validator errors; appends `"Your last answer was identical. Change it."` if output repeated).
- Context Window on Retries: Retains strictly 3 messages (`[original_query, last_answer, retry_message]`).
- Temperature Scaling: Raises generator temperature by `+0.1` on each retry attempt.
- Stalled loop detection: Aborts early with `failure_reason = "possible checker error"` if 2 consecutive attempts produce identical output and identical validator error.

### `src/agentic_harness/llm/` & `config.yaml`
- **`config.yaml`**: Defines provider, model, temperature, and token caps for `generator` (default: Ollama with `qwen2.5:1.5b`) and `judge` (default: Groq with `openai/gpt-oss-20b`).
- **`get_client(role)`**: Returns client configured for role (`"generator"` or `"judge"`).
- **`OllamaClient`**: Local chat client connecting to `http://localhost:11434/api/chat`. Returns typed error `"ollama_unreachable"` if server is not running.
- **`GroqClient`**: Cloud judge client. Automatically retries with doubled tokens on truncation or empty response; handles rate limits with exponential backoff and enforces a minimum 3.0-second gap between consecutive API calls.

---

## 5. Environment Variables
- `GROQ_API_KEY`: API key for Groq LLM calls (defined in `.env`).
