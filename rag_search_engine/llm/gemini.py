from __future__ import annotations

import logging, os, time
from typing import Optional

from rag_search_engine.config import GEMINI_API_KEY
from google import genai

from rag_search_engine.llm.prompt import gemini_method

logger = logging.getLogger(__name__)


class Gemini:
    """
    Small wrapper around a Gemini model for spell-checking movie queries.

    Best practices:
    - API key is read from an environment variable by default (GEMINI_API_KEY).
    - Query text is *optionally* logged; consider masking/redacting if queries can contain PII.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
    ) -> None:
        if api_key is None:
            api_key = GEMINI_API_KEY

        if not api_key:
            # Fail fast if key is missing; don't attempt network calls with invalid config.
            raise ValueError("GEMINI_API_KEY is not set and no api_key was provided")

        self.model_name = model
        self.client = genai.Client(api_key=api_key)

    def enhance(self, method: str, query: str) -> str:
        """
        Use Gemini to correct obvious spelling errors in a movie search query.

        Logging:
        - Duration of the call.
        - Prompt / response / total token usage (if available).
        - Model name and success/failure.
        - Optionally a truncated version of the query and response.

        Security / privacy:
        - You may want to truncate or hash the query in logs in production.
        """
        if not query.strip():
            # Short-circuit empty input
            return query

        start_time = time.perf_counter()
        response_text: str = query  # default to original query on failure

        try:
            content = gemini_method(method, query)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content,
            )

            # Some SDKs expose `response.text`; adjust if your version differs.
            response_text = getattr(response, "text", "") or query

            duration_s = time.perf_counter() - start_time

            # Usage metadata (token counts)
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            response_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)

            # Log a concise, structured line
            # NOTE: consider truncating query/response if logging to shared systems
            logger.info(
                "Gemini spell_check completed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    # Be cautious with these in production if PII is possible:
                    "query_preview": query[:100],
                    "response_preview": response_text[:100],
                    "success": True,
                },
            )

        except Exception as e:
            duration_s = time.perf_counter() - start_time

            # Log the error but do NOT leak secrets or full response.
            logger.exception(
                "Gemini spell_check failed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    # Again, consider truncation / masking in real deployments:
                    "query_preview": query[:100],
                    "success": False,
                },
            )

            # Decide how you want to fail:
            # - return original query (graceful degradation)
            # - or raise the exception
            return query
        return response_text
