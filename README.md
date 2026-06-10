# learnagent

Building an agentic framework from scratch — no LangChain, no LangGraph — to develop first-principles intuition about how agents actually work. The deliverable is being able to explain **every design decision and its alternatives**, not just a working agent.

## Approach: Build → Break → Read → Implement

Build the dumbest version that works, break it intentionally to expose its failure mode, read the paper written to solve that failure, implement the fix, and **measure the delta**. The measured delta is something you experienced, not a fact you memorized.

## Task & Dataset

**Multi-hop QA** on [HotpotQA](https://hotpotqa.github.io/) — questions that can't be answered in one lookup. Exercises reasoning, multiple tool calls, retrieval, cross-step memory, and verifiable evaluation in a single task. Every paper on the roadmap benchmarks on it, so results here are directly comparable.

## Papers Roadmap

| Problem encountered | Paper |
|---------------------|-------|
| No way to "look things up" mid-reasoning | **ReAct** (Yao et al., 2022) |
| Hallucinated tool names / arguments | **Toolformer** (Schick et al., 2023) |
| Context window overflows on long runs | **RAG** (Lewis et al., 2020) |
| Query embeds far from answer documents | **HyDE** (Gao et al., 2022) |
| Agent loops, can't tell when it's wrong | **Reflexion** (Shinn et al., 2023) |
| Memory doesn't survive long trajectories | **MemGPT** (Packer et al., 2023) |
| Reasoning across whole docs, not chunks | **RAPTOR** (Sarthi et al., 2024) |
| Context bloat is expensive | **LLMLingua** (Jiang et al., 2023) |
| Info buried mid-context gets ignored | **Lost in the Middle** (Liu et al., 2023) |
| Not every query needs the biggest model | **LLM Cascade** (Yue et al., 2023) |
| Evaluating open-ended agent tasks | **GAIA** (Mialon et al., 2023) |

## Progress

### Event-driven Tracer (observability first)

Observability is laid down *before* agent logic, not bolted on later. The tracer is **event-driven**: the agent only calls `self._emit(event, data)` and has zero knowledge of the tracer.

- `callbacks/base.py` — abstract `CallbackHandler` event contract
- `callbacks/tracer.py` — `EventTracer` listens and accumulates a structured trace (session → queries → LLM calls, with tokens + latency), saved to JSON

Event-driven over imperative keeps the agent decoupled from instrumentation — adding another handler is just appending to a list, no agent changes.

### Baseline agent + LLM-as-Judge eval

`main.py` — `AnthropicAgent` with a clean session lifecycle (`start_session` → `query` → `close`), one bare LLM call per query. The deliberate "dumbest version that works."

`eval/hotpotqa_eval.py` — runs the agent over N HotpotQA examples and scores each answer with a separate Claude call as judge, reporting accuracy + session metrics.

### Experiment: Baseline (single LLM call, no tools)

The bare LLM answers from memory alone — no retrieval, no reasoning loop. Run on 10 HotpotQA validation examples:

```
Evaluation Results:
  Correct: 1/10
  Accuracy: 10.0%

Session Metrics:
  Total Duration: 39675ms (39.7s)
  Avg Duration per Query: 3967ms
  Total Tokens: 1102 (in: 262, out: 840)
  Avg Tokens per Query: 110
```

**~10% is the number every future component has to beat.** A single call falls apart on multi-hop questions — it answers "I don't have enough information" or confidently guesses wrong. That gap is exactly what tools, retrieval, and reasoning loops exist to close.

> **Eval bug worth recording:** the first judge checked `if "CORRECT" in response`, which matched substrings inside the judge's *reasoning* and rubber-stamped everything — a fake **100%**. Answers that literally contradicted the gold answer passed. Switching to a constrained `"yes"/"no"` output with strict prefix matching gave the real **10%**. A too-lenient judge fails silently: it tells you you're winning when you're not.

## Architecture

```
src/
├── main.py                 # AnthropicAgent — session lifecycle, emits events
├── callbacks/
│   ├── base.py             # CallbackHandler — abstract event contract
│   └── tracer.py           # EventTracer — listens, accumulates, saves trace
├── eval/
│   └── hotpotqa_eval.py    # LLM-as-Judge evaluation on HotpotQA
└── react.py                # ReAct loop (next component)
data/hotpotqa.py            # HotpotQA loader
runs/                       # JSON traces, one per session
```

Dependency direction is strict: **agent → tracer, never the reverse.**

## Setup & Running

```bash
uv add anthropic datasets
export ANTHROPIC_API_KEY="sk-ant-..."   # billed per token, separate from Claude.ai Pro

python src/main.py                    # single baseline query → writes trace to runs/
python src/eval/hotpotqa_eval.py 10   # eval on 10 HotpotQA examples
```

## Next

Wire up the **ReAct loop** (`react.py`) — Thought → Action → Observation — with `wikipedia_search` and `finish` tools so the agent can look things up, then measure the delta against the 10% baseline.
