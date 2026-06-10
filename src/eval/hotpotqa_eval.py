"""Evaluation using LLM as Judge on HotpotQA"""
import sys
import datasets
import anthropic
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import AnthropicAgent
from callbacks.tracer import EventTracer


JUDGE_PROMPT = """You are a strict evaluator. Determine if the model's prediction correctly answers the question.

The prediction is correct only if it provides the same or equivalent answer to the ground truth.

Question: {question}
Ground Truth Answer: {ground_truth}
Model's Prediction: {prediction}

Respond with ONLY "yes" or "no". Nothing else."""


def load_data(num_examples: int = 10):
    """Load HotpotQA validation examples"""
    ds = datasets.load_dataset('hotpotqa/hotpot_qa', 'distractor')
    val_data = ds['validation'].select(range(num_examples))
    return val_data


def judge_answer(client: anthropic.Anthropic, question: str, ground_truth: str, prediction: str) -> bool:
    """Use LLM as Judge to evaluate if prediction is correct"""
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction
    )

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.content[0].text.strip().lower()
    return answer.startswith("yes")


def eval_agent(agent: AnthropicAgent, dataset, judge_client: anthropic.Anthropic, tracer: EventTracer):
    """Run agent on dataset and use LLM as Judge for evaluation"""
    correct = 0
    results = []

    for i, example in enumerate(dataset):
        question = example["question"]
        ground_truth = example["answer"]

        try:
            print(f"\n[{i+1}/{len(dataset)}] Q: {question[:60]}...")
            answer = agent.query(question)

            # Use LLM as Judge
            print(f"  Judging answer...")
            is_correct = judge_answer(judge_client, question, ground_truth, answer)
            correct += is_correct

            results.append({
                "question": question,
                "predicted": answer,
                "ground_truth": ground_truth,
                "correct": is_correct
            })

            print(f"  Predicted: {answer[:80]}...")
            print(f"  Ground truth: {ground_truth}")
            print(f"  Judgment: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "question": question,
                "predicted": f"Error: {str(e)[:50]}",
                "ground_truth": ground_truth,
                "correct": False
            })

    return results, correct


if __name__ == "__main__":
    num_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print(f"Loading HotpotQA dataset ({num_examples} examples)...")
    val_data = load_data(num_examples)

    tracer = EventTracer()
    agent = AnthropicAgent("claude-haiku-4-5", 1024, callbacks=[tracer])
    judge_client = anthropic.Anthropic()

    agent.start_session()

    print(f"\nRunning evaluation on {num_examples} examples...\n")
    results, correct = eval_agent(agent, val_data, judge_client, tracer)

    agent.close()

    # Calculate session metrics
    total_input_tokens = sum(q["tokens"]["input"] for q in tracer.queries)
    total_output_tokens = sum(q["tokens"]["output"] for q in tracer.queries)
    total_tokens = total_input_tokens + total_output_tokens
    total_duration_ms = (tracer.session_end - tracer.session_start) * 1000 if tracer.session_end else 0
    avg_duration_per_query = total_duration_ms / num_examples if num_examples > 0 else 0

    # Print summary
    accuracy = (correct / num_examples * 100) if num_examples > 0 else 0
    print(f"\n{'='*70}")
    print(f"Evaluation Results:")
    print(f"  Correct: {correct}/{num_examples}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"\nSession Metrics:")
    print(f"  Total Duration: {total_duration_ms:.0f}ms ({total_duration_ms/1000:.1f}s)")
    print(f"  Avg Duration per Query: {avg_duration_per_query:.0f}ms")
    print(f"  Total Tokens: {total_tokens} (in: {total_input_tokens}, out: {total_output_tokens})")
    print(f"  Avg Tokens per Query: {total_tokens/num_examples:.0f}")
    print(f"{'='*70}")

    tracer.save("runs/eval_trace.json")
