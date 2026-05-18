import anthropic
import tracer
import time

# simple loop that takes in user question 
# and returns answer from llm
# class AnthropicLLM:

#     def __init__(self, model_name: str, max_tokens=1024) -> None:
#         self.model_name = model_name
#         self.client = anthropic.Anthropic()
#         self.max_tokens = max_tokens

#     def invoke_client(self, question:str) -> str:
#         message = self.client.messages.create(
#             model=self.model_name,
#             max_tokens=self.max_tokens,
#             messages=[{
#                 "role": "user",
#                 "content": question,
#             }]
#         )
#         return message.content[0].text

#     def __call__(self, question: str, answer_format: str = 'JSON') -> dict[str, str]:
#         llm_response = self.invoke_client(question=question)
#         return {"answer": llm_response}


class AnthropicAgent:

    def __init__(self, model_name, max_tokens) -> None:
        self.tools = {}
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()

    def invoke_client(self, question, tracer):
        step_id = tracer.start_step()

        t_start = time.time()
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": question,
            }]
        )
        t_end = time.time()
        tracer.record_llm_call(step_id, message.usage.input_tokens, message.usage.output_tokens, (t_end - t_start) * 1000, self.model_name)
        tracer.stop_step(step_id)
        return message.content[0].text

    def _generate_tool_prompt(self) -> str:
        return f"You have access to the following tools, use them wisely: {','.join(self.tools.keys())}"

    def _generate_system_prompt(self) -> str:
        system_prompt = None
        return ""

    def __call__(self, question: str, tracer: tracer.Tracer) -> dict[str, str]:
        return {"answer": self.invoke_client(question, tracer)}



def main():
    #print("Hello from learnagent!"
    agent = AnthropicAgent("claude-haiku-4-5", 1024)
    # print(agent("Hello claude"))
    hoptopQ = 'Were Scott Derrickson and Ed Wood of the same nationality?'
    agent_tracer = tracer.Tracer(question=hoptopQ)
    final_answer = agent(hoptopQ, agent_tracer)
    print(final_answer)
    agent_tracer.save(final_answer=final_answer["answer"])

if __name__ == "__main__":
    main()
