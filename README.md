# tol-final-project-backend
LLM Inference server for the Tools for Online Learning Project


## Run the Application Locally

1. Activate the virtual environment

    cd worker
    source .venv/bin/activate

2. Run the app
    cd src
    uvicorn worker.misconceptions:app


## Local Testing

    curl http://localhost:8000/healthcheck


