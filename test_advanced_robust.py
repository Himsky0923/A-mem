"""
Evaluation harness using the robust memory layer (no JSON schema dependency).
Drop-in replacement for test_advanced.py.

Usage:
    python test_advanced_robust.py --backend openai --model gpt-4o-mini --dataset data/locomo10.json
    python test_advanced_robust.py --backend ollama --model qwen2.5:3b --dataset data/locomo10.json
"""

from memory_layer_robust import RobustLLMController, RobustAgenticMemorySystem, USAGE, set_usage_phase
from memory_layer import FAILURES, valid_link, cache_is_current, mark_cache
from llm_text_parsers import (
    parse_plain_text_answer,
    parse_relevant_parts,
    parse_keywords_response,
)
import os
import re
import json
import time
import argparse
import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation, CATEGORY_NAMES
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import copy
import pickle
import random
from tqdm import tqdm
from utils import calculate_metrics, aggregate_metrics, score_prediction
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

logger = logging.getLogger("amem_robust")


class RobustAdvancedMemAgent:
    """Agent using the robust memory system with plain-text LLM calls."""

    def __init__(self, model, backend, retrieve_k, temperature_c5,
                 sglang_host="http://localhost", sglang_port=30000, refresh_index=False, seed=None):
        self.memory_system = RobustAgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
            refresh_index=refresh_index,
            seed=seed,
        )
        self.retriever_llm = RobustLLMController(
            backend=backend,
            model=model,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
            seed=seed,
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(content, k=k)

    def retrieve_memory_llm(self, memories_text, query):
        """Select relevant parts of conversation memories — plain text, no JSON schema."""
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts of the conversation that would help answer this specific question.
If no parts are relevant, return the input unchanged."""

        response = self.retriever_llm.llm.get_completion(prompt)
        return parse_relevant_parts(response)

    def generate_query_llm(self, question):
        """Generate query keywords — plain text, no JSON schema."""
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

        response = self.retriever_llm.llm.get_completion(prompt)
        result = parse_keywords_response(response)
        logger.debug("generate_query_llm response: %s", result)
        return result

    def answer_question(self, question: str, category: int, answer: str,
                        rng: Optional[random.Random] = None) -> tuple:
        """Generate answer for a question — plain text, no JSON schema."""
        keywords = self.generate_query_llm(question)
        raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)
        context = raw_context

        assert category in [1, 2, 3, 4, 5]

        if category == 5:
            answer_tmp = list()
            if (rng or random).random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:"""
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
            temperature = 0.7
        elif category == 3:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7

        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt, temperature=temperature,
            )
        except Exception as e:
            logger.warning("answer_question failed: %s — returning empty", e)
            FAILURES["answer"] += 1  # scored as an empty prediction; reported at the end
            response = ""
        return response, user_prompt, raw_context


def pool_memories(agents: List[RobustAdvancedMemAgent]) -> RobustAdvancedMemAgent:
    """One agent whose memory is the union of the given agents' memories (--memory_scope pooled).
    Notes and their retriever rows (the same texts and embeddings) are appended conversation by
    conversation, and each note's links are offset to its position in the pooled list."""
    memories, corpus, embeddings = {}, [], []
    for agent in agents:
        notes = list(agent.memory_system.memories.values())
        retriever = agent.memory_system.retriever
        assert len(notes) == len(retriever.corpus) == len(retriever.embeddings), "memory and retriever out of sync"
        offset = len(memories)
        for note in notes:
            note = copy.copy(note)
            note.links = [link + offset for link in note.links if valid_link(link, len(notes))]
            memories[note.id] = note
        corpus.extend(retriever.corpus)
        embeddings.append(retriever.embeddings)
    embeddings = np.vstack(embeddings)
    assert len(memories) == len(corpus) == len(embeddings), "duplicate note ids across conversations"
    pooled = agents[0]
    pooled.memory_system.memories = memories
    pooled.memory_system.retriever.corpus = corpus
    pooled.memory_system.retriever.embeddings = embeddings
    pooled.memory_system.retriever.document_ids = {doc: i for i, doc in enumerate(corpus)}
    return pooled


def category_label(split_name: str) -> str:
    """'category_1' -> 'category_1 (multi_hop)'; other split names unchanged."""
    if split_name.startswith("category_"):
        cat = int(split_name.split("_")[1])
        return f"{split_name} ({CATEGORY_NAMES.get(cat, 'unknown')})"
    return split_name


def summarize(aggregate: Dict, metrics=("f1", "bleu1")) -> Dict:
    """Mean of the headline metrics per category name (and overall), plus question counts."""
    summary = {}
    for split_name, split in aggregate.items():
        name = "overall" if split_name == "overall" else CATEGORY_NAMES.get(int(split_name.split("_")[1]), split_name)
        summary[name] = {m: split[m]["mean"] for m in metrics if m in split}
        summary[name]["count"] = split[metrics[0]]["count"] if metrics[0] in split else 0
    return summary


REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def safe_name(text: str) -> str:
    """text usable as one path component on every OS: 'Qwen/Qwen2.5-3B' -> 'Qwen-Qwen2.5-3B', 'qwen2.5:3b' -> 'qwen2.5-3b'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-") or "none"


def memory_cache_dir(backend: str, model: str, run_tag: str, seed: int,
                     refresh_index: bool = False, evolution: bool = True) -> str:
    """The memory cache of one configuration. Everything that changes how memory is built is in the name, so a run
    with other settings never loads these memories by accident."""
    name = "{}_{}_{}{}_seed{}_{}".format(backend, safe_name(model), "evo" if evolution else "noevo",
                                         "_refresh" if refresh_index else "", seed, safe_name(run_tag))
    return os.path.join(REPO_DIR, "cached_memories_robust", name)


def default_output_path(model: str, backend: str, run_tag: str) -> str:
    return os.path.join(REPO_DIR, "results", f"{safe_name(model)}_{backend}_{safe_name(run_tag)}.json")


def backend_settings(agent: "RobustAdvancedMemAgent", backend: str) -> Dict:
    """Where requests go and the server's context length, if it reports one."""
    llm = agent.memory_system.llm_controller.llm
    info = {"base_url": None, "context_length": None}
    if hasattr(llm, "client"):  # openai-compatible API
        info["base_url"] = str(llm.client.base_url)
    elif backend == "ollama":
        info["base_url"] = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    elif hasattr(llm, "base_url"):
        info["base_url"] = llm.base_url
    if backend == "vllm":
        try:
            import requests
            models = requests.get(f"{llm.base_url}/v1/models", timeout=10).json().get("data", [])
            info["context_length"] = next((m.get("max_model_len") for m in models if m.get("id") == llm.model),
                                          models[0].get("max_model_len") if models else None)
        except Exception as e:  # the run does not depend on it
            logger.warning("Could not read max_model_len from the vLLM server: %s", e)
    return info


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    eval_logger = logging.getLogger('locomo_eval_robust')
    eval_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    eval_logger.addHandler(console_handler)

    if log_file:
        # utf-8: with the locale encoding (e.g. cp950 on Windows) lines holding other characters were dropped
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        eval_logger.addHandler(file_handler)

    return eval_logger


def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None,
                     ratio: float = 1.0, backend: str = "sglang",
                     temperature_c5: float = 0.5, retrieve_k: int = 10,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000,
                     legacy_c5_reference: bool = False, seed: int = 42,
                     refresh_index: bool = False, memory_scope: str = "conversation",
                     categories=(1, 2, 3, 4, 5), run_tag: str = "default", rebuild_memory: bool = False):
    """Evaluate the robust agent on the LoComo dataset."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = (f"eval_robust_{safe_name(model)}_{backend}_{safe_name(run_tag)}_ratio{ratio}_{memory_scope}"
                    f"_{timestamp}.log")
    log_path = os.path.join(REPO_DIR, "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if output_path is None:
        output_path = default_output_path(model, backend, run_tag)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    eval_logger = setup_logger(log_path)
    eval_logger.info(f"Run tag: {run_tag}; results -> {output_path}; log -> {log_path}")
    eval_logger.info(f"Loading dataset from {dataset_path}")
    eval_logger.info(f"Using ROBUST memory layer (no JSON schema dependency)")
    random.seed(seed)
    np.random.seed(seed)
    eval_logger.info(f"Seed {seed}: Python random, numpy, the category-5 option order and every LLM request "
                     f"(where the backend supports a seed)")
    FAILURES.clear()
    USAGE.clear()
    # wall-clock seconds per phase; construction only counts memories built in this run (not loaded from cache)
    phase_seconds = {"construction": 0.0, "qa": 0.0}

    samples = load_locomo_dataset(dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")

    results = []
    all_metrics = []
    # the same predictions with category 5 scored the other way (see --legacy_c5_reference), so every run
    # reports both the paper's scoring and the corrected one
    all_metrics_other_c5 = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)

    i = 0
    error_num = 0
    memories_dir = memory_cache_dir(backend, model, run_tag, seed, refresh_index)
    os.makedirs(memories_dir, exist_ok=True)
    eval_logger.info(f"Memory cache: {memories_dir}" + (" (--rebuild_memory: existing cache ignored and overwritten)"
                                                        if rebuild_memory else ""))
    built_samples = []
    allow_categories = sorted(set(int(c) for c in categories))
    eval_logger.info("Evaluating categories: " + ", ".join(f"{c} ({CATEGORY_NAMES[c]})" for c in allow_categories))
    cached_samples, stale_cache_samples = [], []

    def load_or_build(sample_idx, sample):
        """An agent holding this conversation's memory, loaded from the cache or built and cached."""
        agent = RobustAdvancedMemAgent(model, backend, retrieve_k, temperature_c5,
                                       sglang_host, sglang_port, refresh_index, seed=seed)

        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(
            memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )

        if os.path.exists(memory_cache_file) and not rebuild_memory:
            eval_logger.info(f"Loading cached memories for sample {sample_idx}")
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            if os.path.exists(retriever_cache_file):
                eval_logger.info(f"Found retriever cache files")
                agent.memory_system.retriever = agent.memory_system.retriever.load(
                    retriever_cache_file, retriever_cache_embeddings_file
                )
            else:
                eval_logger.info(f"No retriever cache found, loading from memory")
                agent.memory_system.consolidate_memories()  # the document format of this memory system
            eval_logger.info(f"Successfully loaded {len(cached_memories)} memories")
            cached_samples.append(sample_idx)
            if not cache_is_current(memory_cache_file):
                stale_cache_samples.append(sample_idx)
                eval_logger.warning(f"Memories of sample {sample_idx} were cached by another version of the memory "
                                    f"layer; delete {memories_dir} to rebuild them")
        else:
            eval_logger.info(f"Building memories for sample {sample_idx} (no cache or --rebuild_memory).")
            built_samples.append(sample_idx)
            set_usage_phase("construction")
            started = time.perf_counter()

            for _, turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    turn_datatime = turns.date_time
                    conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
                    agent.add_memory(conversation_tmp, time=turn_datatime)

            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            mark_cache(memory_cache_file)
            phase_seconds["construction"] += time.perf_counter() - started
            eval_logger.info(f"Successfully cached {len(memories_to_cache)} memories")
        return agent

    if memory_scope == "pooled":
        # every evaluated conversation's memory in one store: retrieval must also reject the
        # other conversations' notes (issue #27); built from the same per-conversation caches
        pooled_agent = pool_memories([load_or_build(i, s) for i, s in enumerate(samples)])
        eval_logger.info(f"Pooled memory: {len(pooled_agent.memory_system.memories)} notes "
                         f"from {len(samples)} conversations")

    for sample_idx, sample in enumerate(samples):
        agent = pooled_agent if memory_scope == "pooled" else load_or_build(sample_idx, sample)

        eval_logger.info(f"Processing sample {sample_idx + 1}/{len(samples)}")
        set_usage_phase("qa")
        started = time.perf_counter()

        for qa_idx, qa in enumerate(sample.qa):
            if int(qa.category) in allow_categories:
                total_questions += 1
                category_counts[qa.category] += 1

                # the category-5 option order comes from the question's own generator
                rng = random.Random(f"{seed}:{sample_idx}:{qa_idx}")
                prediction, user_prompt, raw_context = agent.answer_question(
                    qa.question, qa.category, qa.final_answer, rng
                )

                # Parse the prediction (handles both JSON and plain text)
                prediction = parse_plain_text_answer(prediction)

                reference, metrics = score_prediction(prediction, qa, legacy_c5_reference)
                other_c5 = None
                if qa.category == 5:
                    other_reference, other_metrics = score_prediction(prediction, qa, not legacy_c5_reference)
                    other_c5 = {"reference": other_reference, "metrics": other_metrics}
                all_metrics_other_c5.append(other_c5["metrics"] if other_c5 else metrics)

                eval_logger.info(f"Question {total_questions}: {qa.question}")
                eval_logger.info(f"Prediction: {prediction}")
                eval_logger.info(f"Reference: {reference}")
                eval_logger.info(f"User Prompt: {user_prompt}")
                eval_logger.info(f"Category: {qa.category}")
                eval_logger.info(f"Raw Context: {raw_context}")

                all_metrics.append(metrics)
                all_categories.append(qa.category)

                result = {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": reference,
                    "category": qa.category,
                    "adversarial_answer": qa.adversarial_answer,
                    "metrics": metrics,
                }
                if other_c5:
                    result["other_c5_scoring"] = other_c5
                results.append(result)

                if total_questions % 10 == 0:
                    eval_logger.info(f"Processed {total_questions} questions")
        phase_seconds["qa"] += time.perf_counter() - started
        settings_agent = agent

    aggregate_results = aggregate_metrics(all_metrics, all_categories)
    cost = {}
    for phase in ("construction", "qa"):
        usage = USAGE.get(phase, {})
        cost[phase] = {key: int(usage.get(key, 0)) for key in
                       ("calls", "attempts", "prompt_tokens", "completion_tokens", "max_prompt_tokens", "usage_missing")}
        cost[phase]["seconds"] = round(phase_seconds[phase], 1)
    cost["qa"]["questions"] = total_questions
    cost["construction"]["samples_built"] = len(built_samples)
    settings = {
        "model": model, "backend": backend, **backend_settings(settings_agent, backend),
        # temperatures and limits hard-coded in memory_layer_robust / answer_question
        "temperature": 0.7, "temperature_c5": temperature_c5, "temperature_keyword_retry": 0.3,
        "max_tokens": 1000, "retrieve_k": retrieve_k, "seed": seed, "ratio": ratio,
        "categories": allow_categories, "legacy_c5_reference": legacy_c5_reference,
        "refresh_index": refresh_index, "memory_scope": memory_scope, "run_tag": run_tag,
    }
    main_c5, other_c5_name = ("legacy", "not_mentioned") if legacy_c5_reference else ("not_mentioned", "legacy")
    aggregate_by_c5 = {main_c5: aggregate_results,
                       other_c5_name: aggregate_metrics(all_metrics_other_c5, all_categories)}

    final_results = {
        "model": model,
        "dataset": dataset_path,
        "memory_layer": "robust",
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "categories": allow_categories,
        "category_names": {str(c): n for c, n in CATEGORY_NAMES.items()},
        # mean F1 / BLEU-1 per category name; the full statistics are in aggregate_metrics
        "summary": summarize(aggregate_results),
        "aggregate_metrics": aggregate_results,
        # category 5 scored against the adversarial answer as in the paper ("legacy") and against
        # "Not mentioned in the conversation" ("not_mentioned"); other categories are identical in both
        "summary_by_c5_scoring": {k: summarize(v) for k, v in aggregate_by_c5.items()},
        "aggregate_metrics_by_c5_scoring": aggregate_by_c5,
        "ratio": ratio,
        "seed": seed,
        "legacy_c5_reference": legacy_c5_reference,
        "refresh_index": refresh_index,
        "memory_scope": memory_scope,
        "run_tag": run_tag,
        "settings": settings,
        # LLM calls, tokens and wall-clock time; construction is empty for memories loaded from the cache
        "cost": cost,
        "memory_cache_dir": memories_dir,
        "rebuild_memory": rebuild_memory,
        "built_samples": built_samples,
        "pooled_notes": len(pooled_agent.memory_system.memories) if memory_scope == "pooled" else None,
        "cached_samples": cached_samples,
        "stale_cache_samples": stale_cache_samples,
        # failures absorbed in this run (memories loaded from the cache were built earlier);
        # "answer" counts questions scored as empty predictions because the call failed
        "llm_failures": dict(FAILURES),
        "individual_results": results,
    }
    eval_logger.info(f"Error number: {error_num}")
    eval_logger.info(f"LLM failures absorbed in this run: {dict(FAILURES) or 'none'}")
    eval_logger.info(f"Settings: {settings}")
    for phase, c in cost.items():
        eval_logger.info(f"Cost ({phase}): {c}")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        eval_logger.info(f"Results saved to {output_path}")

    eval_logger.info("Evaluation Summary:")
    eval_logger.info(f"Total questions evaluated: {total_questions}")
    eval_logger.info("Category Distribution:")
    for category, count in sorted(category_counts.items()):
        eval_logger.info(f"Category {category} ({CATEGORY_NAMES.get(category, 'unknown')}): "
                         f"{count} questions ({count/total_questions*100:.1f}%)")

    eval_logger.info("Aggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        eval_logger.info(f"{category_label(split_name)}:")
        for metric_name, stats in metrics.items():
            eval_logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                eval_logger.info(f"    {stat_name}: {value:.4f}")

    for scoring, summary in final_results["summary_by_c5_scoring"].items():
        eval_logger.info(f"Mean F1 / BLEU-1 with category 5 scored '{scoring}': " + "; ".join(
            f"{name} {s.get('f1', 0):.4f} / {s.get('bleu1', 0):.4f} (n={s['count']})" for name, s in summary.items()))

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate robust text-only agent on LoComo dataset (no JSON schema dependency)"
    )
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="openai",
                        help="Backend to use (openai, ollama, sglang, or vllm)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                        help="Temperature for category 5 questions")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang server port (for sglang backend)")
    parser.add_argument("--legacy_c5_reference", action="store_true",
                        help="Score category 5 against the adversarial answer, as the paper's evaluation did "
                             "(default: against 'Not mentioned in the conversation', the correct response)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed: Python random, numpy, the order of the two category-5 answer options, "
                             "the seed sent with each LLM request (vLLM, Ollama, OpenAI-compatible APIs) and the "
                             "memory cache name, so each seed builds its own memories")
    parser.add_argument("--refresh_index", action="store_true",
                        help="Re-embed neighbour memories as soon as evolution updates them (default: their "
                             "embeddings refresh only every evo_threshold evolutions, as in the paper). Builds "
                             "its own memory cache")
    parser.add_argument("--memory_scope", choices=["conversation", "pooled"], default="conversation",
                        help="conversation: each question is answered from its own conversation's memory "
                             "(LoCoMo's protocol, as in the paper). pooled: from one memory holding all "
                             "evaluated conversations, so retrieval must also reject other conversations' "
                             "notes (uses the same memory caches)")
    parser.add_argument("--categories", type=str, default="1,2,3,4,5",
                        help="Comma-separated LoCoMo categories to evaluate: 1 multi-hop, 2 temporal, "
                             "3 open-domain, 4 single-hop, 5 adversarial (default: all)")
    parser.add_argument("--run_tag", type=str, default="default",
                        help="Name of this run. Part of the memory cache directory (with backend, model, evolution "
                             "setting and seed) and of the default output results/{model}_{backend}_{run_tag}.json. "
                             "Runs that should share memories (e.g. a k-sweep) use the same tag and seed")
    parser.add_argument("--rebuild_memory", action="store_true",
                        help="Build the memories again and overwrite this configuration's cache")
    args = parser.parse_args()

    categories = [int(c) for c in args.categories.split(",") if c.strip()]
    if not categories or any(c not in CATEGORY_NAMES for c in categories):
        raise ValueError(f"--categories must be a comma-separated subset of {sorted(CATEGORY_NAMES)}")

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = os.path.join(REPO_DIR, args.output) if args.output else None  # None: default_output_path

    evaluate_dataset(
        dataset_path, args.model, output_path, ratio=args.ratio,
        backend=args.backend, temperature_c5=args.temperature_c5, retrieve_k=args.retrieve_k,
        sglang_host=args.sglang_host, sglang_port=args.sglang_port,
        legacy_c5_reference=args.legacy_c5_reference, seed=args.seed,
        refresh_index=args.refresh_index, memory_scope=args.memory_scope,
        categories=categories, run_tag=args.run_tag, rebuild_memory=args.rebuild_memory,
    )


if __name__ == "__main__":
    main()
