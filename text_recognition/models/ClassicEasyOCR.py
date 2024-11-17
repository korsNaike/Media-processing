import logging

from text_recognition.models.BaseModel import BaseModel
import easyocr


class ClassicEasyOCR(BaseModel):

    def __init__(self, model: easyocr.easyocr.Reader = None):
        self.model = model or easyocr.Reader(['en'])

    def image_to_string(self, image, **params):
        results = self.model.readtext(image, allowlist='0123456789', **params)

        if len(results) < 1:
            return None

        for (bbox, text, prob) in results:
            return text