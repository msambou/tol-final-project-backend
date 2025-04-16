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
        """Checks token count.

        Args:
            v (str): The string we want to check.

        Raises:
            RequestValidationError: If the number of tokens is greater than our context-size, then fail.

        Returns:
            str: Returns string as-is if validated.
        """
        token_count = len(LLM_MODEL.tokenize(v.encode("utf-8")))
        if token_count > CONTEXT_SIZE:
            msg = f"Token count exceeds the maximum allowed limit of {CONTEXT_SIZE}."
            raise RequestValidationError(
                msg,
            )
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
    """Simple LLM response function.

    Args:
        message (LLMPrompt): Prompt as input.
        llm (Llama): Pre-loaded model.

    Returns:
        Union[CreateCompletionResponse, Iterator[CreateCompletionResponse]]: Tokens of output.
    """
    system_message = "You are a helpful assistant."
    prompt = message
    template = f"""<|system|>
    {system_message}</s>
    <|user|>
    {prompt}</s>
    <|assistant|>"""
    return llm(template, temperature=0.0, max_tokens=128)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # pragma: no cover
    """Gets the lifespan of an app.

    Args:
        app (FastAPI): app instance.

    Yields:
        None: Allows FastAPI to handle startup and shutdown events.
    """
    # Load the ML model
    global LLM_MODEL
    LLM_MODEL = Llama(
        model_path="tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        n_ctx=CONTEXT_SIZE,
        n_batch=1,
    )

    yield
    # Clean up the ML models and release the resources
    del LLM_MODEL


app = FastAPI(lifespan=lifespan)


@app.get("/api", response_model=BaseLlamaResponse)
async def send_llm_response(
    message: LLMPrompt = Depends(LLMPrompt),
) -> Union[CreateCompletionResponse, Iterator[CreateCompletionResponse]]:
    """Send model output as API response.

    Args:
        message (LLMPrompt, optional): API input.. Defaults to Depends(LLMPrompt).

    Returns:
        Union[CreateCompletionResponse, Iterator[CreateCompletionResponse]]: API output.
    """
    return await get_llm_response(message, LLM_MODEL)


@app.get("/healthcheck")
def healthcheck() -> Literal["OK"]:
    """Simple healthcheck.

    Returns:
        str: "OK"
    """
    return "OK"
