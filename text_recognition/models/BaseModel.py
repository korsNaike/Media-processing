from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def image_to_string(self, image, **params):
        pass