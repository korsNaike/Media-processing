from text_recognition.models.BaseModel import BaseModel
import pytesseract

from text_recognition.playground import clear_from_chars


class TesseractWithPostProcessing(BaseModel):

    def image_to_string(self, image, **params):
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        return clear_from_chars(pytesseract.image_to_string(image, config=custom_config))