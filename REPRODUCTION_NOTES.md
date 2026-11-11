# Reproduction notes: A-Mem on LoCoMo

Results and notes of the reproduction of *A-Mem: Agentic Memory for LLM Agents* (arXiv 2502.12110, latest version)
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
| Model server and context window | **Paper** (Sec. 4.2): Ollama; window not stated. The released code (Feb 2025) sets none, so Ollama's default applied: 2,048 tokens before v0.6.7 (Apr 2025), longer prompts silently cut from the start. **Repository (2026):** vLLM, `--max-model-len` 4,096 (`run_all_experiments.sh`) or 8,192 (`run_k_sweep.sh`, README) | vLLM, 8,192 | The largest of the authors' settings. Longer prompts are rejected instead of silently cut, scored 0 and counted (`answer_context_overflow`). The paper's Qwen runs may have answered from shortened prompts without any sign of it |
| vLLM version and flags | 0.8.5, `--enforce-eager`, `--gpu-memory-utilization 0.95`, Turing (sm_75) GPUs | 0.33.0, CUDA graphs on, 0.55, PyTorch sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`), RTX 4080 Super 16 GB | None intended: speed and VRAM only. Same `--dtype float16` |
| k | Tuned per category (Appendix Table 8; Qwen 10/10/50/10/10) | k = 10 for all categories, from a sweep (10-50) on the same data; per-category k optional | k chosen on the evaluation data, as the authors' `run_k_sweep.sh` also does |
| GPT-4o-mini | OpenAI API | DeepSeek `deepseek-chat` | Different model |

## Hidden assumptions and findings

1. **The context window is undocumented, and overflows were not reported.** The paper never states a context
   limit: the full text of v11, appendices included, mentions no context length, token limit or `num_ctx`, and the
   `OllamaController` in the code released with it (Feb 2025) passes no context option. Its Qwen results came from Ollama (Sec. 4.2), which before v0.6.7 used 2,048 tokens and silently dropped
   the start of longer prompts. The repository's later vLLM scripts use 4,096 or 8,192, and the authors' commit
   b4890dc (20 Oct 2026) admits: *"The robust answer path scored a failed call as an empty prediction without
   saying so. A prompt longer than a small server's context window (e.g. vLLM --max-model-len 8192 with k=10 plus
   linked neighbours) fails this way."* We use 8,192, the most generous documented setting, and count every
   overflow (`llm_failures.answer_context_overflow`, `answer_error` per question). The answer prompt holds the k
   retrieved notes plus their linked notes, about 100-120 tokens per note. In our k-sweep (Qwen, evolution on,
   seed 1, 497 questions):

   | k | 10 | 20 | 30 | 40 | 50 |
   |---|---|---|---|---|---|
   | Questions overflowed | 0 | 2 (0.4%) | 198 (40%) | 442 (89%) | 497 (100%) |
   | Longest answer prompt (tokens) | 5,158 | 8,038 | 8,188 | 8,172 | – |
   | Overall F1 (paper's c5 scoring) | 30.36 | 32.97 | 22.13 | 3.55 | 0.00 |

   The paper's open-domain k = 50 for Qwen therefore cannot fit even in 8,192 tokens.
   **Small-model baselines.** Table 1 also reports LoCoMo, ReadAgent, MemoryBank and MemGPT on the small models.
   LoCoMo and MemGPT use about 17,000 tokens per question, far beyond a 2,048-token Ollama window, while A-Mem uses
   about 1,100-1,400. If the baselines ran under the same Ollama default, they lost most of their prompt and A-Mem's
   lead on small models would be inflated. The paper does not say how the baselines were run and the repository
   contains no baseline code, so this cannot be checked; we reproduce only A-Mem's own numbers.
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
10. **`deepseek-chat` is an alias.** In Nov 2026 the DeepSeek API lists only `deepseek-flash` and
    `deepseek-v4-pro`; requests for `deepseek-chat` are answered by `deepseek-flash` (response `model` field,
    recorded in `settings.served_models`). Report the served model, not the requested name.
11. **The two models use evolution very differently** (smoke test, first 60 turns of conversation 1, seed 1):
    Qwen2.5-3B chose UPDATE_NEIGHBOR / STRENGTHEN_AND_UPDATE in 38 of 59 decisions and rewrote 187 neighbour notes;
    DeepSeek chose STRENGTHEN (links only) in 57 of 59 and rewrote 10. Disabling evolution can therefore only have
    a small effect on DeepSeek's memories.
12. **Answer prompts grow with links.** In the same test the longest answer prompt was 2,859 tokens for Qwen
    (evolution on), 4,376 (off) and 9,564 for DeepSeek, which links far more. Over a full conversation Qwen's
    prompts will be longer; DeepSeek's context window is large enough that it does not overflow, so the two
    models do not face the same limit.
13. **Category-5 scoring flips the adversarial result.** In the smoke test Qwen answered "Not mentioned in the
    conversation" to both adversarial questions: F1 0 under the paper's scoring, 100 under the corrected one.
14. **Qwen's evolution writes the prompt's placeholders into memory.** `UPDATE_NEIGHBORS_PROMPT` shows its
    output format with the example values `CONTEXT: updated context sentence` and `TAGS: tag1, tag2, tag3`.
    Qwen2.5-3B often copies them literally, and the parser accepts them, so real notes are overwritten. After
    building the three conversations (evolution on, seed 1), 131 of 419, 121 of 369 and 252 of 663 notes have the
    placeholder context or only placeholder tags: 504 of 1,451 (about 35%). Notes that mix placeholder and real tags
    are not counted. Example: *"I went to a LGBTQ support group yesterday ..."* has context "updated context
    sentence". It is not caused by the context window: all construction requests finished normally (no `length`
    stops) and almost all prompts were under 2,000 tokens. Left unchanged, as fixing it would change the method;
    it bears directly on the evolution ablation, which the paper ran only on GPT-4o-mini.

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
| Not enough VRAM for 3 parallel runs + vLLM | Each evaluation process loads MiniLM and BERTScore's RoBERTa-large on the GPU (~1.8 GB); peak 15.7 / 16 GB | `PARALLEL=2` while vLLM runs |
| Shell commands fail on `\r` | CRLF line endings on a Windows checkout | `.gitattributes` keeps `*.sh` LF; README strips `\r` from `requirements.txt` |
| DeepSeek HTTP 500 / 503 and empty replies during the main runs | Server load (memory building slowed to ~11.6 s per turn for an hour) | Built-in retries (3 attempts); 0-4 calls per run gave up and were counted in `llm_failures`; no answer failed |

## Results

12 main runs (2 models × evolution on/off × seeds 1, 2, 3), k = 10, first 3 conversations (497 questions); tables
from `results/summary.md` (`bash scripts/run_experiments.sh aggregate`). F1 in %, mean ± sample std over seeds.
Every run answered all 497 questions: 0 failed answers and 0 context overflows in all 12 runs.

**Chosen k: 10** (sweep table in finding 1). k = 20 scored 2.6 F1 higher on one seed, but its longest answer prompt
was 154 tokens below the limit, and prompts without evolution are longer (main runs: 6,942 vs 5,305 tokens at
k = 10), so the no-evolution condition would overflow more and the comparison would partly measure overflows.

### Qwen2.5-3B, evolution on, vs paper Table 1

| Category | Ours F1 | Paper F1 | Ours BLEU-1 | Paper BLEU-1 |
|---|---|---|---|---|
| Multi-hop | 23.79 ± 3.03 | 12.57 | 14.99 ± 3.22 | 9.01 |
| Temporal | 25.20 ± 2.42 | 27.59 | 18.93 ± 1.72 | 25.07 |
| Open-domain | 9.85 ± 3.06 | 7.12 | 7.41 ± 2.09 | 7.28 |
| Single-hop | 32.55 ± 0.21 | 17.23 | 27.87 ± 0.21 | 13.12 |
| Adversarial | 40.71 ± 2.26 | 27.91 | 38.47 ± 2.30 | 25.15 |

Same order of magnitude, not the same numbers: higher in 4 of 5 categories, by more than the seed spread.

### Evolution on vs off (the modification)

| Model | Evolution | Multi-hop | Temporal | Open-domain | Single-hop | Adversarial | Overall | Overall (corrected c5) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | on | 23.79 ± 3.03 | 25.20 ± 2.42 | 9.85 ± 3.06 | 32.55 ± 0.21 | 40.71 ± 2.26 | 30.79 ± 0.34 | 35.96 ± 1.32 |
| Qwen2.5-3B | off | 19.86 ± 0.53 | 25.83 ± 2.62 | 8.63 ± 2.58 | 32.24 ± 0.78 | 36.96 ± 2.10 | 29.31 ± 1.01 | 36.25 ± 0.41 |
| DeepSeek | on | 35.37 ± 2.09 | 30.19 ± 4.50 | 10.13 ± 4.03 | 46.23 ± 1.20 | 26.84 ± 4.62 | 35.81 ± 2.30 | 47.21 ± 0.51 |
| DeepSeek | off | 32.28 ± 3.19 | 29.86 ± 2.08 | 11.61 ± 0.84 | 45.16 ± 2.08 | 27.94 ± 6.24 | 35.17 ± 2.58 | 45.97 ± 0.88 |
| Paper, GPT-4o-mini (Table 3) | on | 27.02 | 45.85 | 12.14 | 44.65 | 50.03 | – | – |
| Paper, GPT-4o-mini (Table 3) | off | 21.35 | 31.24 | 10.13 | 39.17 | 44.16 | – | – |

- Disabling evolution changes overall F1 by +1.5 (Qwen) and +0.6 (DeepSeek) with the paper's scoring, −0.3 and +1.2
  corrected. No difference is significant (Welch's t-test, n = 3 per condition, all p > 0.1).
- The paper's temporal gain from evolution (+14.6 F1 for GPT-4o-mini) does not appear (−0.6 and +0.3). With a seed
  spread of 2-4.5 F1, an effect that size would be clear; effects of 1-4 points cannot be excluded with 3 seeds.
- Multi-hop is higher with evolution in all 6 seed pairs (Qwen +1.3, +6.4, +4.1; DeepSeek +2.8, +4.5, +2.0; sign
  test p ≈ 0.03). Exploratory, as many categories were tested.
- Under the corrected category-5 scoring, evolution's adversarial effect for Qwen reverses (63.6 on vs 67.8 off).
- DeepSeek against the paper's GPT-4o-mini: higher on multi-hop (35.4 vs 27.0), similar on single-hop, lower on
  temporal (30.2 vs 45.9). Its low adversarial score under the paper's scoring comes from answering "Not mentioned in
  the conversation"; corrected, it scores 77.4.

### Cost of evolution (mean per run)

| Model | Evolution | Build LLM calls | Build tokens | Build min | QA tokens | Longest answer prompt | Neighbour notes rewritten |
|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | on | 4,284 | 3.10 M | 73 | 1.41 M | 5,305 | 3,900 |
| Qwen2.5-3B | off | 3,421 | 2.23 M | 42 | 1.43 M | 6,942 | 0 |
| DeepSeek | on | 4,440 | 3.88 M | 134 | 5.14 M | 16,307 | 544 |
| DeepSeek | off | 4,328 | 3.41 M | 107 | 4.01 M | 11,215 | 0 |

For Qwen, evolution adds 75% build time and 39% build tokens. DeepSeek rarely chooses to rewrite neighbours, and
its answer prompts are far longer than Qwen's window would allow. DeepSeek times depend on API load.

### Run notes

- Qwen's seed-1 evolution run reused the memory built in the k-sweep (same settings, seed and run tag); only its
  answers were regenerated, so its build cost is the mean of seeds 2 and 3.
- DeepSeek requests for `deepseek-chat` were all served by `deepseek-flash` (`settings.served_models`).
- Results, caches and logs are gitignored; the commands in the README regenerate them.
