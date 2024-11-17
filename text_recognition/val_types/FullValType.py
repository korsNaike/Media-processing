from text_recognition.val_types.BaseValType import BaseValType


class FullValType(BaseValType):


    def check_value(self, model_value: str, correct_value: str) -> float | int:
        return 1 if model_value == correct_value else 0