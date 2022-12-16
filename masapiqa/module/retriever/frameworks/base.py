from abc import ABC, abstractmethod


class BaseRetrieverFramework(ABC):

    @abstractmethod
    def predict(self, question: str, config: dict) -> dict:
        ...

    @abstractmethod
    def to(self, device):
        ...
