"""LLM Server Definition."""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Iterator, List, Optional, Union

from fastapi import Depends, FastAPI, Query
from fastapi.exceptions import RequestValidationError
from llama_cpp import CreateCompletionResponse, Llama
from pydantic import BaseModel, validator
from typing_extensions import Literal
from collections.abc import AsyncGenerator

CONTEXT_SIZE = 512
LLM_MODEL = None


class LLMPrompt(BaseModel):
    """LLM Input Validation."""

    message: str = Query("message")

    @validator("message")
    def check_token_count(cls, v: str) -> str:
        """Checks token count against context limit."""
        token_count = len(LLM_MODEL.tokenize(v.encode("utf-8")))
        if token_count > CONTEXT_SIZE:
            msg = f"Token count exceeds the maximum allowed limit of {CONTEXT_SIZE}."
            raise RequestValidationError(msg)
        return v


class Choice(BaseModel):
    """Choice Output Validator."""

    text: str = "Beep boop"
    index: int = 0
    logprobs: Optional[str] = "null"
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Output Validator."""

    prompt_tokens: int = 198
    completion_tokens: int = 10
    total_tokens: int = 208


class BaseLlamaResponse(BaseModel):
    """Simple API response validator."""

    id: str = "cmpl-7fc1be4c-8f5b-4b2f-805f-f8c5086a9fb4"
    object: str = "text_completion"
    created: int = 1708459650
    model: str = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    choices: List[Choice]
    usage: Usage


async def get_llm_response(
    message: LLMPrompt,
    llm: Llama,
) -> Union[CreateCompletionResponse, Iterator[CreateCompletionResponse]]:
    """Generates structured feedback on student code."""

    system_message = "You are an expert Python tutor. Your job is to identify misconceptions students have in the code, explain them clearly, and suggest correct alternatives."

    prompt = """A student wrote the following function:

def fibonacci(n):
    a, b = 0, 1
    while a < n:
        a, b = b, a + b
    return b

Please analyze this function and explain:
1. What the student likely intended.
2. Any misconceptions in the code.
3. A correct version of the code if the goal is to return the nth Fibonacci number.
4. Example usage and output.
"""

    template = f"""<|system|>
{system_message}</s>
<|user|>
{prompt}</s>
<|assistant|>"""

    return llm(template, temperature=0.0, max_tokens=512)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan handler for FastAPI startup/shutdown."""
    global LLM_MODEL
    LLM_MODEL = Llama(
        model_path="tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        n_ctx=CONTEXT_SIZE,
        n_batch=1,
    )
    yield
    del LLM_MODEL


app = FastAPI(lifespan=lifespan)


@app.get("/api", response_model=BaseLlamaResponse)
async def send_llm_response(
    message: LLMPrompt = Depends(LLMPrompt),
) -> Union[CreateCompletionResponse, Iterator[CreateCompletionResponse]]:
    """Send model output as API response."""
    message = """
        def fibonacci(n):
            a, b = 0, 1
            while a < n:
                a, b = b, a + b
            return b
"""
    msgObj = LLMPrompt(message=message)
    return await get_llm_response(msgObj, LLM_MODEL)


@app.get("/healthcheck")
def healthcheck() -> Literal["OK"]:
    """Simple healthcheck."""
    return "OK"

@app.post("/analyze")
def analyzeMisconceptions():
    # I am using fastAPI
    pass
