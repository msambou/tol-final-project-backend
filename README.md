# tol-final-project-backend
LLM Inference server for the Tools for Online Learning Project

## Dev Setup
1. Ensure that the `tinyllama-1.1b-chat-v1.0.Q2_K.gguf` file is downloaded into the worker/src directory

2. To do that, run the command below:

    wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf

## Run the Inference Server Locally

1. Activate the virtual environment

    cd worker
    source .venv/bin/activate

2. Run the app
    cd src
    uvicorn worker.misconceptions:app


## Local Testing

    curl http://localhost:8000/healthcheck


