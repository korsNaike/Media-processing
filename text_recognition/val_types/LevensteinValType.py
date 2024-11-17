from text_recognition.val_types.BaseValType import BaseValType
from fuzzywuzzy import fuzz


class LevensteinValType(BaseValType):

    def check_value(self, model_value: str, correct_value: str) -> float | int:
        similarity = fuzz.ratio(model_value, correct_value) / 100
        return similarity