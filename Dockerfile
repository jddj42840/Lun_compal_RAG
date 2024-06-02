# FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
FROM python:3.10.12-slim-bookworm

WORKDIR /compal_rag

COPY ./icons /compal_rag/icons
COPY ./utils /compal_rag/utils
COPY ./config.json /compal_rag/config.json
COPY ./main.py /compal_rag/main.py
COPY ./requirements.txt /compal_rag/requirements.txt

RUN python3 -m pip install -r requirements.txt

EXPOSE 7861

CMD [ "python3", "main.py" ]