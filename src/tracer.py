import uuid
import json
import time

class Tracer:

    def __init__(self, question) -> None:
        self.start_time = time.time()
        self.question = question
        self.steps = []
        self.llm_steps = []
        self.run_id = uuid.uuid4()

    def start_step(self):
        self.step_start = time.time()
        step_id = len(self.steps)
        return step_id

    def stop_step(self, step_id):
        time_spent = (time.time() - self.step_start) * 1000
        step_metadata = {
            "step_id": step_id,
            "start_ms": self.step_start,
            "stop_ms": time.time(),
            "duration_ms": time_spent,
        }
        self.steps.append(step_metadata)

    def record_llm_call(self, step_id, input_tokens, output_tokens, duration_ms, model):
        llm_call_id = len(self.llm_steps)
        llm_call_metadata = {
            "call_id": llm_call_id,
            "step_id": step_id,
            "ninput_tokens": input_tokens,
            "noutput_tokens": output_tokens,
            "duration_ms": duration_ms,
            "model": model,
        }

        self.llm_steps.append(llm_call_metadata)

    def save(self, final_answer):
        self.end_time = time.time()
        self.final_answer = final_answer
        metadata = {
            "run_id": str(self.run_id),
            "question": self.question,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_steps": len(self.steps),
            "final_answer": self.final_answer,
            "steps": self.steps,
            "llm_calls": self.llm_steps,
        }

        import os
        os.makedirs('runs', exist_ok=True)
        with open(f'runs/{str(self.run_id)}.json', 'w') as f:
            json.dump(metadata, f, indent=2)

