# Use an official Python runtime as a parent image
FROM dataloopai/dtlpy-agent:cpu.py3.8.opencv4.7

# Install any needed packages specified in requirements.txt
USER root
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev 

# Install any Python packages
RUN pip install --no-cache-dir pytesseract pandas numpy pillow dtlpy

RUN useradd -m myuser
USER myuser




