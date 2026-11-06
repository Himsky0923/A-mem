# Agentic Memory 🧠

A novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way.

> **Note:** This repository is specifically designed to reproduce the results presented in our paper. If you want to use the A-Mem system in building your agents, please refer to our official implementation at: [A-mem-sys](https://github.com/WujiangXu/A-mem-sys)

For more details, please refer to our paper: [A-Mem: Agentic Memory for LLM Agents](https://arxiv.org/pdf/2502.12110)

> 🔥 **News (Oct 2026):** We provide a stronger version of A-Mem, **FlexA-Mem: Budget-Adaptive Agentic Memory for LLM Agents** ([paper](https://arxiv.org/html/2610.33274v1), [code](https://github.com/WujiangXu/Flex-Amem)). FlexA-Mem keeps A-Mem's linked notes but uses them as the compressed history itself: at every step, the agent sees the whole history as a timeline of notes, and the notes most relevant to the current step are expanded in place into their raw content or excerpts, so the prompt fits any token budget. It is evaluated on LongMemEval-S, LoCoMo, SWE-bench Verified and AppWorld.

## Reproduction study: A-Mem on LoCoMo, with and without memory evolution

This fork (branch `reproduction`) reproduces A-Mem's LoCoMo results (F1 / BLEU-1 per question category, paper
Table 1) with **Qwen2.5-3B-Instruct**, and makes one modification: **disabling memory evolution**
(`--no_evolution`), the step in which adding a note rewrites the context and tags of related existing notes. Note
construction, linking, retrieval and answering are unchanged. **DeepSeek (`deepseek-chat`)** stands in for the
paper's GPT-4o-mini, whose Table 3 has the authors' own "w/o memory evolution" ablation. Every condition is run with
3 seeds and reported as mean ± std. Everything uses `test_advanced_robust.py`, the evaluation the upstream README
recommends; `test_advanced.py` is unchanged.

Findings, deviations from the authors' setup and the debug diary are in
[REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md).

### Setup (Linux or WSL2 Ubuntu, NVIDIA GPU with ≥ 12 GB)

Tested on Windows 11 + WSL2 Ubuntu (Python 3.12), RTX 4080 Super 16 GB, vLLM 0.33.0.

```bash
# uv (no sudo needed) and a Python build that ships Python.h (vLLM's compiler steps need it;
# Ubuntu's system python3 has no headers without `sudo apt install python3.12-dev`)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv ~/venvs/amem --python 3.12 --managed-python

# vLLM + the repo requirements. 'pathlib' is skipped: on PyPI it is an old Python 2 backport
tr -d '\r' < requirements.txt | grep -vxE 'pathlib|#.*|' > /tmp/amem_reqs.txt
uv pip install --python ~/venvs/amem/bin/python vllm -r /tmp/amem_reqs.txt

# API key for DeepSeek (only needed for the DeepSeek runs); .env is gitignored
cp .env.example .env    # then put your key in OPENAI_API_KEY
```

The first vLLM start downloads Qwen2.5-3B-Instruct from Hugging Face (about 6 GB).

Ollama also works as a backend (`ollama pull qwen2.5:3b-instruct-fp16`, then `--backend ollama --model
qwen2.5:3b-instruct-fp16 --num_ctx 8192`), but the results here use vLLM as the authors did.

### Running the experiments

```bash
# terminal 1: the model server (authors' settings: fp16, --max-model-len 8192)
bash scripts/start_vllm.sh

# terminal 2, from the repository root, one step at a time
bash scripts/run_experiments.sh ksweep          # Qwen, evolution on, seed 1, k = 10..50 -> choose K
K=10 bash scripts/run_experiments.sh qwen       # Qwen: evolution on / off x seeds 1 2 3
K=10 bash scripts/run_experiments.sh deepseek   # DeepSeek: the same six runs (no GPU server needed)
K=10 bash scripts/run_experiments.sh full       # optional: Qwen baseline on the whole dataset
bash scripts/run_experiments.sh aggregate       # results/summary.md + CSVs
```

Runs use `--ratio 0.3`: the first 3 of the 10 LoCoMo conversations (1,451 turns, 497 questions). Measured on the
GPU above with three runs at once: building memories takes about 3.3 s per turn for Qwen with evolution, 1.7 s
without, and 3.8 s for DeepSeek (API latency); answering about 5 s per question. One trial is therefore roughly
2 h (Qwen, evolution on), 1.3 h (Qwen, off) and 2.3 h (DeepSeek).

`PARALLEL` runs that many trials at once. Each evaluation process also puts its embedding and BERTScore models on
the GPU (about 1.8 GB); with the vLLM server (`GPU_UTIL=0.55`) three processes peaked at 15.7 of 16 GB, so use
**`PARALLEL=2`** on a 16 GB card while vLLM runs, and run DeepSeek after the Qwen steps (or stop vLLM first).
Memories are cached in `cached_memories_robust/` per backend, model, evolution setting, seed and run tag, so the
k-sweep and later steps reuse them.

On the DeepSeek API, `deepseek-chat` is an alias: responses report `deepseek-flash`. Each results file records the
served model in `settings.served_models`.

A single run, for reference:

```bash
~/venvs/amem/bin/python test_advanced_robust.py --backend vllm --model Qwen/Qwen2.5-3B-Instruct \
    --sglang_port 30000 --ratio 0.3 --retrieve_k 10 --seed 1 --run_tag evo_s1            # baseline
~/venvs/amem/bin/python test_advanced_robust.py --backend vllm --model Qwen/Qwen2.5-3B-Instruct \
    --sglang_port 30000 --ratio 0.3 --retrieve_k 10 --seed 1 --run_tag noevo_s1 --no_evolution
~/venvs/amem/bin/python test_advanced_robust.py --backend openai --model deepseek-chat \
    --base_url https://api.deepseek.com --ratio 0.3 --retrieve_k 10 --seed 1 --run_tag evo_s1
```

### Aggregating

`python scripts/aggregate.py results/main/*.json --out results/summary` groups the result files by model ×
evolution on/off × k × ratio and writes `results/summary.md` and CSVs: F1 and BLEU-1 (%) per category and overall as
mean ± std over the trials, once with category 5 scored as in the paper and once corrected, the paper's numbers
for comparison, and cost and reliability (LLM calls, tokens, minutes, failed answers and context overflows, parse
fallbacks, neighbour notes rewritten).

Each results JSON holds `settings` (model, backend, base URL, temperatures, max tokens, k, seed, context length,
evolution), `summary` and `aggregate_metrics` (per category), `aggregate_metrics_by_c5_scoring`, `cost`
(construction vs QA), `llm_failures`, `parse_fallbacks`, `evolution_stats` and per-question `individual_results`.

### Changes from upstream

All changes are in `test_advanced_robust.py`, `memory_layer_robust.py`, `llm_text_parsers.py`,
`load_dataset.py` and new files; one commit each.

| Change | Why |
|---|---|
| `.gitignore` | The repo had none: logs, caches, results and `.env` (API keys) could be committed. |
| `--categories`, category names in results | Categories were hard-coded; results only had numbers. Mapping (1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial) checked against the data and the paper's Table 1. |
| Both category-5 scorings in every run | Upstream now scores category 5 against "Not mentioned in the conversation"; the paper scored it against the adversarial answer. Saving both makes runs comparable with the paper and with the corrected metric. |
| `--run_tag`, per-configuration memory cache, `--rebuild_memory` | The cache was named only by backend and model, so a run with evolution off or another seed would silently load memories built for a different configuration. Model names are made path-safe (`qwen2.5:3b` crashed on Windows); logs are UTF-8; results are saved by default. |
| `--seed` seeds `random`, `numpy` and each LLM request | It only fixed the category-5 option order; all calls sample at temperature 0.7. |
| Cost, time and settings in results | No record of LLM calls, tokens, time (memory construction vs answering) or of the settings behind a result. |
| `--num_ctx` for Ollama (default 8192) | Ollama's default context is smaller and it truncates longer prompts silently. |
| `--retrieve_k_per_category` (optional) | The paper tunes k per category (Appendix Table 8); the script had one k. |
| `--base_url` (or `OPENAI_BASE_URL`), `.env` loading | To use an OpenAI-compatible API (DeepSeek). The robust pipeline sends no `response_format`, so no JSON-mode change was needed. |
| Parse-fallback counters | When a model's plain-text output cannot be parsed, the parsers silently use defaults or heuristics; now counted (`parse_fallbacks`). |
| **`--no_evolution`** (the modification) | Skips the neighbour-update call, the only step that changes existing notes; the decision call, links, the new note's tags and consolidation timing are unchanged. Evolution counts are saved with each cache (`evolution_stats`). |
| Context-overflow counting | An answer prompt longer than the context window (vLLM `--max-model-len 8192`) is scored as an empty prediction; now counted separately (`answer_context_overflow`) and marked per question. |
| `scripts/aggregate.py`, `scripts/start_vllm.sh`, `scripts/run_experiments.sh`, `.gitattributes` | Mean ± std tables over trials; serving and running the experiments; LF line endings for shell scripts on Windows checkouts. |

## Introduction 🌟

Large Language Model (LLM) agents have demonstrated remarkable capabilities in handling complex real-world tasks through external tool usage. However, to effectively leverage historical experiences, they require sophisticated memory systems. Traditional memory systems, while providing basic storage and retrieval functionality, often lack advanced memory organization capabilities.

Our project introduces an innovative **Agentic Memory** system that revolutionizes how LLM agents manage and utilize their memories:

<div align="center">
  <img src="Figure/intro-a.jpg" alt="Traditional Memory System" width="600"/>
  <img src="Figure/intro-b.jpg" alt="Our Proposed Agentic Memory" width="600"/>
  <br>
  <em>Comparison between traditional memory system (top) and our proposed agentic memory (bottom). Our system enables dynamic memory operations and flexible agent-memory interactions.</em>
</div>

## Key Features ✨

- 🔄 Dynamic memory organization based on Zettelkasten principles
- 🔍 Intelligent indexing and linking of memories
- 📝 Comprehensive note generation with structured attributes
- 🌐 Interconnected knowledge networks
- 🔄 Continuous memory evolution and refinement
- 🤖 Agent-driven decision making for adaptive memory management

## Framework 🏗️

<div align="center">
  <img src="Figure/framework.jpg" alt="Agentic Memory Framework" width="800"/>
  <br>
  <em>The framework of our Agentic Memory system showing the dynamic interaction between LLM agents and memory components.</em>
</div>

## How It Works 🛠️

When a new memory is added to the system:
1. Generates comprehensive notes with structured attributes
2. Creates contextual descriptions and tags
3. Analyzes historical memories for relevant connections
4. Establishes meaningful links based on similarities
5. Enables dynamic memory evolution and updates

## Results 📊

Empirical experiments conducted on six foundation models demonstrate superior performance compared to existing SOTA baselines.

## Getting Started 🚀

1. Clone the repository:
```bash
git clone https://github.com/WujiangXu/AgenticMemory.git
cd AgenticMemory
```

2. Install dependencies:
Option 1: Using venv (Python virtual environment)
```bash
# Create and activate virtual environment
python -m venv a-mem
source a-mem/bin/activate  # Linux/Mac
a-mem\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

Option 2: Using Conda
```bash
# Create and activate conda environment
conda create -n myenv python=3.9
conda activate myenv

# Install dependencies
pip install -r requirements.txt
```

3. Run the experiments in LoCoMo dataset:

**Option A — Original evaluation (requires OpenAI JSON schema support):**
```bash
python test_advanced.py
```

**Option B — Robust evaluation (recommended, works with any LLM backend):**

The robust evaluation (`test_advanced_robust.py`) removes the JSON schema dependency and supports OpenAI API, vLLM, and Ollama backends.

```bash
# OpenAI models
python test_advanced_robust.py --backend openai --model gpt-4o-mini \
    --dataset data/locomo10.json --output results_robust_gpt-4o-mini.json

# vLLM-served open-source models (start vLLM server first)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct --port 30000 \
    --dtype float16 --enforce-eager --max-model-len 8192
python test_advanced_robust.py --backend vllm --model Qwen/Qwen2.5-3B-Instruct \
    --dataset data/locomo10.json --output results_robust_qwen3b.json \
    --sglang_port 30000

# Ollama models
python test_advanced_robust.py --backend ollama --model qwen2.5:3b \
    --dataset data/locomo10.json --output results_robust_ollama_qwen3b.json
```

Key arguments:
- `--retrieve_k`: Number of memories to retrieve per query (default: 10). Tune this per model for best results.
- `--ratio`: Fraction of dataset to evaluate (e.g., `--ratio 0.1` for 10% quick test).
- `--backend`: One of `openai`, `vllm`, `ollama`.
- `--sglang_port`: Port for vLLM/SGLang server (default: 30000).

4. Run the full k-sweep to find optimal retrieval k per model:
```bash
bash run_k_sweep.sh
```

**Note:** To achieve the optimal performance reported in our paper, please adjust the hyperparameter `k` value accordingly. Memories are cached after the first run, so subsequent k-sweep evaluations only re-run the QA answering step.

**Categories Information:** The LoCoMo dataset contains the following categories:
* Category 1: Multi-hop
* Category 2: Temporal
* Category 3: Open-domain
* Category 4: Single-hop
* Category 5: Adversarial

For more details about the categories, please refer to [this GitHub issue](https://github.com/snap-research/locomo/issues/6). 

## Citation 📚

If you use this code in your research, please cite our series of works on agentic memory:

```bibtex
@inproceedings{xu2025amem,
  title={A-Mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}

@article{xu2026flexamem,
  title={FlexA-Mem: Budget-Adaptive Agentic Memory for LLM Agents},
  author={Xu, Wujiang and Han, Jiaojiao and Zhu, Xi and Zhang, Qizheng and Liang, Kaiqu and Yin, Yutong and Mei, Kai and Metaxas, Dimitris N.},
  journal={arXiv preprint arXiv:2610.33274},
  year={2026}
}
```

## License 📄

This project is licensed under the MIT License. See LICENSE for details.


