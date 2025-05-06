import pytesseract
from PIL import Image
import pandas as pd
import dtlpy as dl
import os
import logging

logger = logging.getLogger('tesseract-adapter')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for Tesseract OCR',
                              init_inputs={'model_entity': dl.Model})
class TesseractAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        logger.info('model loaded')

    def prepare_item_func(self, item: dl.Item):
        logger.info('preparing item')
        buffer = item.download(save_locally=False)
        image = Image.open(buffer)
        return image

    def predict(self, batch, **kwargs):
        logger.info('Predicting batch of size: {}'.format(len(batch)))
        batch_annotations = list()

        for image in batch:
            image_annotations = dl.AnnotationCollection()
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
            logger.info('Text extracted')

            # Check if OCR data is valid and contains necessary columns
            if data is not None and 'block_num' in data.columns:
                # Proceed with processing if data is valid
                grouped_blocks = data.groupby('block_num')
                logger.info('Number of blocks found: {}'.format(len(grouped_blocks)))

                for block_num, block in grouped_blocks:
                    ser = block['text'].dropna()

                    # Check if the text series for the block is not empty
                    if ser.empty == False:
                        # Concatenate the text if there is any
                        block_text = ser.str.cat(sep=' ').strip()
                        conf_ser = block['conf']
                        # filter only cong > 0
                        conf_ser = conf_ser[conf_ser > 0]
                        logger.info("Block text:' {}'".format(block_text))
                        # Check if block_text is not an empty string
                        if block_text != "":
                            # Extract bounding box dimensions
                            x, y, w, h = block.iloc[0]['left'], block.iloc[0]['top'], block.iloc[0]['width'], block.iloc[0]['height']
                            box_annotation = dl.Box(left=x, top=y, right=x + w, bottom=y + h, label='Text', description=block_text)
                            image_annotations.add(box_annotation, model_info={'name': self.model_entity.name,
                                                                              'model_id': self.model_entity.id,
                                                                              'confidence': float(conf_ser.mean()) / 100})
                        else:
                            logger.info('No text, skipping this block.')

                    else:
                        logger.info('Empty series, skipping this block.')

            else:
                # Log or handle the case when OCR data is not valid or missing 'block_num'
                logger.info('Invalid OCR data or missing block numbers, skipping this image.')

            batch_annotations.append(image_annotations)

        return batch_annotations
