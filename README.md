# OCR pyteseract Model Adapter

## Introduction

This repo is a model integration between [Google's Tesseract OCR Engine](https://github.com/madmaze/pytesseract)
and [Dataloop](https://dataloop.ai/).

Python-tesseract is an optical character recognition (OCR) tool for python.
That is, it will recognize and "read" the text embedded in images. In this repo we implement the integration between Google's Tesseract OCR Engine
model with our Dataloop platform.

## Requirements

- dtlpy
- pytesseract==0.3.10
- Pillow==10.1.0
- pandas==2.1.4
- numpy==1.26.2
- An account in the [Dataloop platform](https://console.dataloop.ai/)

## Installation

To install the package and create the Google's Tesseract OCR Engine model adapter, you will need a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project).

## Cloning

For instruction how to clone the pretrained model for prediction
click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#predicting).

### Editing the configuration

To edit configurations via the platform, go to the pytesseract page in the Model Management and edit the json file displayed there or, via the SDK, by editing the model configuration.
Click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#model-configuration) for more information.

## Deployment

After installing the pretrained model, it is necessary to deploy it, so it can be used
for prediction.

## Sources and Further Reading

- [Tesseract documentation](https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file)
