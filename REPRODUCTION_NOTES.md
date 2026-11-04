# Reproduction notes: A-Mem on LoCoMo

Working notes for the reproduction of *A-Mem: Agentic Memory for LLM Agents* (arXiv 2502.12110, latest version)
with the authors' repository. Setup and commands are in the [README](README.md#reproduction-study-a-mem-on-locomo-with-and-without-memory-evolution).

## Target and modification

- **Reproduction target:** paper Table 1, A-Mem row for Qwen2.5-3B: F1 / BLEU-1 (%) on LoCoMo per category.
  Multi-hop 12.57 / 9.01, temporal 27.59 / 25.07, open-domain 7.12 / 7.28, single-hop 17.23 / 13.12,
  adversarial 27.91 / 25.15.
- **Modification:** disable memory evolution (`--no_evolution`). The paper's own ablation (Table 3, GPT-4o-mini
  only) reports that removing it lowers every category, e.g. multi-hop F1 27.02 → 21.35, temporal 45.85 → 31.24.
- **Second model:** DeepSeek (`deepseek-chat`) in place of GPT-4o-mini. Results are not directly comparable to the
  paper's GPT-4o-mini numbers because the underlying LLM differs.
- **Metrics:** token-level F1 and BLEU-1 between prediction and reference, per category, as computed by the
  repository (`utils.calculate_metrics`). Category 5 is reported both as the paper scored it and corrected (below).

## Deviations from the authors' setup

| Item | Authors | Here | Effect on results |
|---|---|---|---|
| Evaluation script | Paper numbers: original strict-JSON pipeline (`test_advanced.py`) | `test_advanced_robust.py`, which the README now recommends (plain-text prompts, section parsing, evolution split into 3 calls) | Prompts differ from the paper's runs, so some gap is expected |
| Data | Full LoCoMo (10 conversations, 1,986 questions) | `--ratio 0.3`: first 3 conversations, 497 questions (one optional full run) | Subset; per-category counts are smaller |
| Trials | One run per setting | 3 seeds per condition, mean ± std | – |
| vLLM | 0.8.5, `--enforce-eager`, `--gpu-memory-utilization 0.95` | 0.33.0, CUDA graphs on, 0.55, PyTorch sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`) | None intended: speed and VRAM only. Same `--dtype float16 --max-model-len 8192` |
| k | Tuned per category (Appendix Table 8; Qwen 10/10/50/10/10) | One k from a sweep (10-50) on the same data; per-category k optional | k chosen on the evaluation data, as the authors' `run_k_sweep.sh` also does |
| GPT-4o-mini | OpenAI API | DeepSeek `deepseek-chat` | Different model |

## Hidden assumptions and findings

1. **Context overflows are not reported.** The paper never mentions a context limit, truncation or failed
   answers. The authors' own later commit (b4890dc, 20 Oct 2026) says: *"The robust answer path scored a failed
   call as an empty prediction without saying so. A prompt longer than a small server's context window (e.g. vLLM
   --max-model-len 8192 with k=10 plus linked neighbours) fails this way."* The answer prompt holds the k retrieved
   notes plus up to k linked notes each, about 100-120 tokens per note. With k = 50 (the paper's open-domain k for
   Qwen) it cannot fit in 8,192 tokens. We keep the authors' 8,192 and count overflows per run
   (`llm_failures.answer_context_overflow`, `answer_error` per question).
   Version detail: vLLM 0.33 rejects only prompts over 8,192 tokens and shortens the completion when prompt +
   `max_tokens` (1000) exceeds the window. Older versions such as the authors' 0.8.5 may have rejected those too
   (not verified).
2. **Category 5 scoring changed after the paper.** The paper's evaluation used the adversarial (wrong) answer as
   the reference for category-5 questions, so choosing the trap option scored high; upstream fixed it in Oct 2026
   (score against "Not mentioned in the conversation", `--legacy_c5_reference` restores the old way). Every run
   here saves both.
3. **Category numbering.** The data, README and Table 1 column order agree: 1 multi-hop, 2 temporal,
   3 open-domain, 4 single-hop, 5 adversarial. The paper's text lists them as "(1) single-hop, (2) multi-hop,
   (3) temporal ...", which does not match.
4. **Sampling.** Every LLM call (note construction, evolution, keyword queries, answers) samples at temperature
   0.7 (0.5 for category-5 answers); upstream sent no seed. Runs here send `--seed` with each request.
   DeepSeek accepts the `seed` parameter but does not document whether it is used.
5. **Robust parser guesses evolution decisions.** If the model's reply has no valid `DECISION:` line, the
   decision is inferred from words anywhere in the reply, e.g. "no update needed" becomes UPDATE_NEIGHBOR
   (counted as `parse_fallbacks.decision_inferred_from_text`).
6. **Keyword retry never runs.** `analyze_content` retries with a focused prompt when keywords are empty, but the
   parser has already filled them from the content heuristically, so the retry is unreachable
   (counted as `parse_fallbacks.analysis_heuristic_keywords`). Left unchanged, as it changes results.
7. **Evolution can rewrite old notes away from their content.** In a 10-turn test on Qwen2.5-3B, evolution changed
   the context of the opening greeting ("Hey Mel! Good to see you! How have you been?") from *"The speaker greets a
   friend by asking how they are doing."* (evolution off) to *"The speaker expresses intent to pursue further
   education and explore career options, highlighting the associated excitement."*
8. **Evolution and prompt length are linked.** Evolution changes notes' tags and context, and with it what gets
   retrieved and linked. If one condition overflows the context more often, part of a score difference can come
   from overflows; the aggregate table lists overflows per condition.
9. **Dataset statistic.** `load_dataset.py` prints that 1,954 of 1,986 questions have image evidence; its check
   (`dia_id.endswith(turn_id)`) matches far too broadly. Printing only; it does not affect evaluation.

## Debug diary

| Problem | Cause | Resolution |
|---|---|---|
| `NotADirectoryError` for model `qwen2.5:3b` on Windows | Model name used in the cache directory name; `:` is not allowed on Windows | Path-safe names (commit 3126741) |
| Log lines missing on Windows | Log file written in the locale encoding (cp950) | UTF-8 log files (3126741) |
| `TypeError: process() takes no keyword arguments` calling the API from Windows Anaconda | Anaconda's `httpx2` incompatible with an installed decompression package | Not fixed; all runs in a WSL venv |
| `pip install -r requirements.txt` | `pathlib` on PyPI is a Python 2 backport | Skipped in the install command |
| No `pip` in a WSL venv | Ubuntu Python without `ensurepip`; `sudo apt` needed | `uv` (no sudo) |
| vLLM: `fatal error: Python.h: No such file or directory` | Triton compiles a CUDA helper; Ubuntu's Python has no headers | venv on uv-managed CPython 3.12, which ships them |
| vLLM: `Could not find nvcc` | FlashInfer's sampler JIT-compiles with the CUDA toolkit, absent in WSL | `VLLM_USE_FLASHINFER_SAMPLER=0` (PyTorch sampler) |
| First vLLM start took ~16 min | Model download (5.8 GB) | Cached afterwards (model loads in seconds) |
| Shell commands fail on `\r` | CRLF line endings on a Windows checkout | `.gitattributes` keeps `*.sh` LF; README strips `\r` from `requirements.txt` |

## Results

To be filled from `results/summary.md` after the runs (`bash scripts/run_experiments.sh aggregate`).

- Chosen k (sweep):
- Qwen2.5-3B, evolution on vs off, vs paper Table 1:
- DeepSeek, evolution on vs off, vs paper Table 1 / Table 3 (GPT-4o-mini):
- Context overflows per condition:
- Cost (LLM calls, tokens, time) per condition:
