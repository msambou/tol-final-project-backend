FROM python:3.10-slim as requirements-stage
WORKDIR /tmp
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* /tmp/

RUN poetry self add poetry-plugin-export

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
FROM python:3.10
WORKDIR /code
COPY --from=requirements-stage /tmp/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt && wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf
COPY ./fastapi_p4 /code/app
CMD ["uvicorn", "app.llmserver:app", "--host", "0.0.0.0", "--port", "8080"]
