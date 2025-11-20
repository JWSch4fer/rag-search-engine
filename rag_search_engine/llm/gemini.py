from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Callable

from rag_search_engine.config import GEMINI_API_KEY
from google import genai
from google.genai import errors as genai_errors  # type: ignore[import]

from rag_search_engine.llm.prompt import gemini_method, gemini_reranking

logger = logging.getLogger(__name__)


class Gemini:
    """
    Small wrapper around a Gemini model for query enhancement and reranking.

    Best practices:
    - API key is read from an environment variable by default (GEMINI_API_KEY).
    - Query text is *optionally* logged; consider masking/redacting if queries can contain PII.
    """

    # How long to sleep (seconds) when we hit a RESOURCE_EXHAUSTED / 429 quota error
    RETRY_SLEEP_SECONDS: int = 30

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

    # ------------------------------------------------------------------
    # Internal helper for rate-limited calls
    # ------------------------------------------------------------------
    @classmethod
    def _generate_with_retry(
        cls,
        func: Callable[[], Any],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Call `func` (typically a `generate_content` call), retrying once on
        RESOURCE_EXHAUSTED / 429 with a 10s backoff.

        Args:
            func: Zero-argument callable performing the Gemini API call.
            context: Optional context dict included in warning logs.

        Returns:
            Whatever `func` returns (usually a `GenerateContentResponse`).

        Raises:
            Any non-quota `ClientError`, or quota errors after the second attempt.
        """
        last_exc: Optional[Exception] = None

        for attempt in (1, 2):
            try:
                return func()
            except genai_errors.ClientError as exc:  # type: ignore[attr-defined]
                last_exc = exc
                is_quota = getattr(
                    exc, "code", None
                ) == 429 or "RESOURCE_EXHAUSTED" in str(exc)
                if attempt == 1 and is_quota:
                    logger.warning(
                        "Gemini quota exceeded; sleeping %s seconds before retry",
                        cls.RETRY_SLEEP_SECONDS,
                        extra={"context": context or {}},
                    )
                    time.sleep(cls.RETRY_SLEEP_SECONDS)
                    continue
                # Not a quota error or already retried once → bubble up
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Gemini call failed without a response")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enhance(self, method: str, query: str) -> str:
        """
        Use Gemini to enhance a movie search query.

        Methods:
            - "spell":   fix obvious spelling errors.
            - "rewrite": rewrite the query to be clearer / more natural.
            - "expand":  expand the query with related terms.

        Logging:
        - Duration of the call.
        - Prompt / response / total token usage (if available).
        - Model name and success/failure.
        - Optionally a truncated version of the query and response.

        Security / privacy:
        - You may want to truncate or hash the query in logs in production.
        """
        if not query.strip() or not method:
            # Short-circuit empty input
            return query

        start_time = time.perf_counter()
        response_text: str = query  # default to original query on failure

        try:
            contents = gemini_method(method, query)

            response = self._generate_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                ),
                context={
                    "kind": "enhance",
                    "method": method,
                    "model": self.model_name,
                },
            )

            # Some SDKs expose `response.text`; adjust if your version differs.
            response_text = getattr(response, "text", "") or query

            duration_s = time.perf_counter() - start_time

            # Usage metadata (token counts)
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            response_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)

            logger.info(
                "Gemini enhance completed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    # Be cautious with these in production if PII is possible:
                    "method": method,
                    "query_preview": query[:100],
                    "response_preview": response_text[:100],
                    "success": True,
                },
            )

        except Exception:
            duration_s = time.perf_counter() - start_time

            # Log the error but do NOT leak secrets or full response.
            logger.exception(
                "Gemini enhance failed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "method": method,
                    # Again, consider truncation / masking in real deployments:
                    "query_preview": query[:100],
                    "success": False,
                },
            )

            # Graceful degradation: return original query
            return query

        return response_text

    def rerank_document(self, query: str, doc: Dict[str, Any], method: str) -> float:
        """
        Ask Gemini to rate how well a single movie matches the search query.

        Intended for "individual" rerank mode. The prompt text is built by
        `gemini_reranking(method, query, doc)`.

        Returns:
            A float score in [0, 10]. If parsing fails or the call errors, returns 0.0.
        """
        title = doc.get("title", "")

        prompt = gemini_reranking(query, doc, method)
        start_time = time.perf_counter()
        score: float = 0.0

        try:
            response = self._generate_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                ),
                context={
                    "kind": "rerank_document",
                    "model": self.model_name,
                    "title_preview": title[:80],
                },
            )

            raw_text = getattr(response, "text", "").strip()

            # Try to parse the first number in the response
            # (handles cases like "8", "8.5", "8.5/10" etc.)
            parsed: Optional[float] = None
            for token in raw_text.replace("/10", " ").split():
                try:
                    parsed = float(token)
                    break
                except ValueError:
                    continue

            if parsed is None:
                score = 0.0
            else:
                # Clamp to [0, 10] just in case
                score = max(0.0, min(10.0, parsed))

            duration_s = time.perf_counter() - start_time

            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            response_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)

            logger.info(
                "Gemini rerank_document completed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    "query_preview": query[:100],
                    "title_preview": title[:80],
                    "raw_llm_score": raw_text[:50],
                    "parsed_score": score,
                    "success": True,
                },
            )

        except Exception:
            duration_s = time.perf_counter() - start_time
            logger.exception(
                "Gemini rerank_document failed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "query_preview": query[:100],
                    "title_preview": title[:80],
                    "success": False,
                },
            )
            score = 0.0

        return score

    def rerank_batch(
        self, query: str, docs: List[Dict[str, Any]], method: str
    ) -> List[int]:
        """
        Batch rerank: pass the query and a list of candidate docs, and receive
        a JSON list of IDs in order of relevance (best first).

        Prompt shape (built by `gemini_reranking(method, query, docs)`):

            Rank these movies by relevance to the search query.

            Query: "{query}"

            Movies:
            ID: 1
            Title: ...
            Description: ...

            ID: 2
            ...

            Return ONLY the IDs in order of relevance (best match first).
            Return a valid JSON list, nothing else. For example:

            [75, 12, 34, 2, 1]

        Returns:
            List of doc IDs in ranked order. If parsing fails, returns [].
        """

        if not docs:
            return []

        prompt = gemini_reranking(query, docs, method)
        start_time = time.perf_counter()
        ranked_ids: List[int] = []

        try:
            response = self._generate_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                ),
                context={
                    "kind": "rerank_batch",
                    "model": self.model_name,
                    "num_docs": len(docs),
                },
            )

            raw_text = getattr(response, "text", "").strip()

            # Try direct JSON parse
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                # Fallback: extract the first [...] region
                start = raw_text.find("[")
                end = raw_text.rfind("]")
                if start != -1 and end != -1 and end > start:
                    snippet = raw_text[start : end + 1]
                    parsed = json.loads(snippet)
                else:
                    raise

            if not isinstance(parsed, list):
                raise ValueError("LLM returned non-list JSON for batch rerank")

            for item in parsed:
                try:
                    ranked_ids.append(int(item))
                except (TypeError, ValueError):
                    continue

            duration_s = time.perf_counter() - start_time

            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            response_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)

            logger.info(
                "Gemini rerank_batch completed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    "query_preview": query[:100],
                    "num_docs": len(docs),
                    "num_ranked_ids": len(ranked_ids),
                    "success": True,
                },
            )

        except Exception:
            duration_s = time.perf_counter() - start_time
            logger.exception(
                "Gemini rerank_batch failed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "query_preview": query[:100],
                    "num_docs": len(docs),
                    "success": False,
                },
            )
            return []

        return ranked_ids
