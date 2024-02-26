import json
import os
import unittest
import dtlpy as dl
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from model_adapter import TesseractAdapter

class MockDataloopItem:
    def download(self, save_locally):
        return "tests/test_image.png"


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.item = MockDataloopItem()
        with open(r'dataloop.json', 'r') as f:
            config = json.load(f)

        model = dl.Model.from_json(_json=config.get('components', dict()).get('models', list())[0], client_api=dl.ApiClient(),
                                   project=None, package=dl.Package())

        self.adapter = TesseractAdapter(model_entity=model)

    def test_prediction(self):
        np_image = self.adapter.prepare_item_func(item=self.item)
        image_annotations = self.adapter.predict(batch=[np_image])
        self.assertIsInstance(image_annotations, list)
        self.assertIsInstance(image_annotations[0], dl.AnnotationCollection)


if __name__ == "__main__":
    unittest.main()
