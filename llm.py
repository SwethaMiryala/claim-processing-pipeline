"""Small helper around the LLM so every agent calls it the same way."""
import os
from functools import lru_cache
from typing import Type, TypeVar

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    # Reads GOOGLE_API_KEY from the environment automatically.
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0,
    )


def extract(schema: Type[T], system: str, user: str) -> T:
    """Ask the LLM and force the answer into a Pydantic schema."""
    structured = get_llm().with_structured_output(schema)
    return structured.invoke([("system", system), ("human", user)])
