from abc import ABC, abstractmethod


class BaseValType(ABC):

    @abstractmethod
    def check_value(self, model_value: str, correct_value: str) -> float | int:
        pass