"""Event tracer for the agent"""
from callbacks import base
import time
import json
import os
import uuid


class EventTracer(base.CallbackHandler):

    def __init__(self) -> None:
        self.session_start = None
        self.session_end = None
        self.queries = []
        self.current_query = None

    def on_session_start(self, data: dict) -> None:
        self.session_start = data["timestamp"]

    def on_session_end(self, data: dict) -> None:
        self.session_end = data["timestamp"]

    def on_query_start(self, data: dict) -> None:
        self.current_query = {
            "query_id": data["query_id"],
            "user_input": data["user_input"],
            "start_time": data["timestamp"],
            "llm_calls": []
        }

    def on_llm_start(self, data: dict) -> None:
        pass

    def on_llm_end(self, data: dict) -> None:
        self.current_query["llm_calls"].append({
            "model": data["model"],
            "input_tokens": data["input_tokens"],
            "output_tokens": data["output_tokens"],
            "duration_ms": data["duration_ms"]
        })

    def on_query_end(self, data: dict) -> None:
        self.current_query["answer"] = data["answer"]
        self.current_query["duration_ms"] = data["duration_ms"]
        self.current_query["tokens"] = {
            "input": data["total_tokens_in"],
            "output": data["total_tokens_out"]
        }
        self.queries.append(self.current_query)
        self.current_query = None

    def save(self, filepath="runs/trace.json"):
        os.makedirs('runs', exist_ok=True)

        trace = {
            "session_id": str(uuid.uuid4()),
            "start_time": self.session_start,
            "end_time": self.session_end,
            "total_duration_ms": (self.session_end - self.session_start) * 1000 if self.session_end else 0,
            "total_queries": len(self.queries),
            "total_tokens": {
                "input": sum(q["tokens"]["input"] for q in self.queries),
                "output": sum(q["tokens"]["output"] for q in self.queries)
            },
            "queries": self.queries
        }
        
        filepath = f"runs/trace_{trace['session_id']}"
        with open(filepath, 'w') as f:
            json.dump(trace, f, indent=2)
