FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.11_opencv

USER root
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev 

RUN pip install --no-cache-dir pytesseract dtlpy
USER 1000




