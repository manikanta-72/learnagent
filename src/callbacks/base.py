"""Callback Handler"""
import abc

class CallbackHandler(abc.ABC):

    @abc.abstractmethod
    def on_session_start(self, data: dict) -> None:
        ...

    @abc.abstractmethod
    def on_session_end(self, data: dict) -> None:
        ...

    @abc.abstractmethod
    def on_query_start(self, data: dict) -> None:
        ...

    @abc.abstractmethod
    def on_query_end(self, data: dict) -> None:
        ...

    @abc.abstractmethod
    def on_llm_start(self, data: dict) -> None:
        ...

    @abc.abstractmethod
    def on_llm_end(self, data: dict) -> None:
        ...

