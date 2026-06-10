import anthropic
import time
from callbacks.tracer import EventTracer


class AnthropicAgent:

    def __init__(self, model_name, max_tokens, callbacks=None):
        self.tools = {}
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()
        self.callbacks = callbacks or []
        self.session_started = False
        self.query_count = 0

    def _emit(self, event_name: str, data: dict):
        for callback in self.callbacks:
            method = getattr(callback, f"on_{event_name}", None)
            if method:
                method(data)

    def start_session(self):
        if not self.session_started:
            self._emit("session_start", {"timestamp": time.time()})
            self.session_started = True

    def close(self):
        self._emit("session_end", {"timestamp": time.time()})

    def query(self, question: str) -> str:
        if not self.session_started:
            self.start_session()

        self.query_count += 1
        query_id = self.query_count

        query_start_time = time.time()
        self._emit("query_start", {
            "query_id": query_id,
            "user_input": question,
            "timestamp": query_start_time
        })

        llm_start_time = time.time()
        self._emit("llm_start", {"llm_call_id": 0, "model": self.model_name})

        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": question}]
        )

        llm_end_time = time.time()
        self._emit("llm_end", {
            "llm_call_id": 0,
            "model": self.model_name,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "duration_ms": (llm_end_time - llm_start_time) * 1000
        })

        answer = message.content[0].text
        query_end_time = time.time()

        self._emit("query_end", {
            "query_id": query_id,
            "answer": answer,
            "duration_ms": (query_end_time - query_start_time) * 1000,
            "total_tokens_in": message.usage.input_tokens,
            "total_tokens_out": message.usage.output_tokens
        })

        return answer


def main():
    tracer = EventTracer()
    agent = AnthropicAgent("claude-haiku-4-5", 1024, callbacks=[tracer])

    agent.start_session()

    question = 'Were Scott Derrickson and Ed Wood of the same nationality?'
    answer = agent.query(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")

    agent.close()
    tracer.save()


if __name__ == "__main__":
    main()
