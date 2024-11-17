from text_recognition.models.BaseModel import BaseModel
import pytesseract

class ClassicPytesseract(BaseModel):

    def image_to_string(self, image, **params):
        return pytesseract.image_to_string(image)