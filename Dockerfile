FROM python:3.10-slim-buster

ENV PORT=8000
ENV DEBUG=1
ENV WORKERS=1
ENV MODEL_PATH=model/ner_pos_model.keras
ENV DATA_PATH=dataset/data.tsv
ENV MAX_LEN=50
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive


WORKDIR /api

RUN pip3 install --upgrade pip
RUN pip3 install poetry
COPY ./pyproject.toml ./poetry.lock ./
RUN poetry export --only main --output requirements.txt
RUN pip3 uninstall poetry -y
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE ${PORT}
CMD ["/bin/sh", "-c", "uvicorn application:app --workers ${WORKERS} --host 0.0.0.0 --port ${PORT}"]