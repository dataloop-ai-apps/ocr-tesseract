import os
import unittest
import dtlpy as dl
from model_adapter import TesseractAdapter


class TestRunner(unittest.TestCase):
    def setUp(self):
        dl.setenv("rc")
        print(f'env -> {dl.environment()}')
        email = os.environ.get('LOGIN_EMAIL', None)
        password = os.environ.get('LOGIN_EMAIL_PASSWORD', None)
        print(f'email {email}')
        if email and password:
            is_logged_in = dl.login_m2m(email, password)
            print(f'is_logged_in {is_logged_in}')
        else:
            dl.login()
        self.item = dl.items.get(item_id='656f18f8acee2c047f321021')
        model = dl.models.get(model_id='65a8f2c8364dccc205861b3f')
        self.tesseract_adapter = TesseractAdapter(model_entity=model)

    def test_prediction(self):

        np_image = self.tesseract_adapter.prepare_item_func(item=self.item)
        image_annotations = self.tesseract_adapter.predict(batch=[np_image])
        self.assertIsInstance(image_annotations, list)
        self.assertIsInstance(image_annotations[0], dl.AnnotationCollection)


if __name__ == "__main__":
    unittest.main()
