from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Callable

import mimetypes
from rag_search_engine.config import GEMINI_API_KEY
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors  # type: ignore[import]

from rag_search_engine.llm.prompt import (
    gemini_evaluation,
    gemini_method,
    gemini_rag_answer,
    gemini_rag_basic,
    gemini_rag_citations,
    gemini_rag_summarize,
    gemini_reranking,
)

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

    def evaluate_results(
        self,
        query: str,
        docs: List[Dict[str, Any]],
    ) -> List[int]:
        """
        Use the LLM to rate each result from 0–3 for relevance to the query.

        Scale:
        - 3: Highly relevant
        - 2: Relevant
        - 1: Marginally relevant
        - 0: Not relevant

        Returns:
            List of integer scores (0–3), same length as `docs`. Any parsing
            or API failure results in default score 0.
        """
        if not docs:
            return []

        # Format each result as a single line:
        # "1. Title - Description"
        formatted_results = []
        for idx, d in enumerate(docs, start=1):
            title = str(d.get("title", "")).strip()
            desc = str(d.get("description", d.get("document", ""))).strip()
            formatted_results.append(f"{idx}. {title} - {desc}")

        prompt: str = gemini_evaluation(query, formatted_results)

        start_time = time.perf_counter()
        scores: List[int] = [0] * len(docs)

        try:
            response = self._generate_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                ),
                context={
                    "kind": "evaluate_results",
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
                raise ValueError("LLM returned non-list JSON for evaluate_results")

            # Map parsed values to int scores 0–3, align with docs length
            for i in range(len(docs)):
                try:
                    raw_val = parsed[i]
                except IndexError:
                    break
                try:
                    val = int(raw_val)
                except (TypeError, ValueError):
                    val = 0
                scores[i] = max(0, min(3, val))

            duration_s = time.perf_counter() - start_time
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            response_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)

            logger.info(
                "Gemini evaluate_results completed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    "query_preview": query[:100],
                    "num_docs": len(docs),
                    "success": True,
                },
            )

        except Exception:
            duration_s = time.perf_counter() - start_time
            logger.exception(
                "Gemini evaluate_results failed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "query_preview": query[:100],
                    "num_docs": len(docs),
                    "success": False,
                },
            )
            # scores stays as all zeros

        return scores

    def rag_answer(self, query: str, docs: str) -> str:
        """
        Use Gemini to generate an answer based on the query and retrieved documents.

        The answer should be tailored to Hoopla users (a movie streaming service).
        """
        # Build the prompt in your preferred incremental style
        prompt = gemini_rag_basic(query, docs)
        # Log at the start
        logger.info(
            "Gemini rag_answer starting",
            extra={
                "model": self.model_name,
                "query_preview": query[:100],
            },
        )
        logger.debug(
            "rag_answer prompt preview: %s",
            prompt[:2000],  # avoid spamming logs with giant prompts
        )
        logger.debug(
            "rag_answer docs preview: %s",
            docs[:2000],
        )

        start_time = time.perf_counter()
        answer: str = ""

        try:
            response = self._generate_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                ),
                context={
                    "kind": "rag_answer",
                    "model": self.model_name,
                },
            )

            answer = getattr(response, "text", "").strip()

            duration_s = time.perf_counter() - start_time
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            response_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)

            logger.debug(
                "rag_answer answer preview: %s",
                answer[:500],
            )

            logger.info(
                "Gemini rag_answer completed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    "query_preview": query[:100],
                    "success": True,
                },
            )

        except Exception:
            duration_s = time.perf_counter() - start_time
            logger.exception(
                "Gemini rag_answer failed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "query_preview": query[:100],
                    "success": False,
                },
            )
            # fall back to a simple message if needed
            answer = "Sorry, I couldn't generate an answer at this time."

        return answer

    def summarize_results(self, query: str, docs: str) -> str:
        """
        Use Gemini to summarize multiple search results for a given query.

        The summary should be:
        - useful to Hoopla users (a movie streaming service),
        - information-dense and concise,
        - 3–4 sentences combining information from multiple movies.
        """
        # Build the prompt in incremental style
        prompt = gemini_rag_summarize(query, docs)
        # Logging at start
        logger.info(
            "Gemini summarize_results starting",
            extra={
                "model": self.model_name,
                "query_preview": query[:100],
            },
        )
        logger.debug(
            "summarize_results prompt preview: %s",
            prompt[:2000],
        )
        logger.debug(
            "summarize_results results preview: %s",
            docs[:2000],
        )

        start_time = time.perf_counter()
        summary: str = ""

        try:
            response = self._generate_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                ),
                context={
                    "kind": "summarize_results",
                    "model": self.model_name,
                },
            )

            summary = getattr(response, "text", "").strip()

            duration_s = time.perf_counter() - start_time
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            response_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)

            logger.debug(
                "summarize_results answer preview: %s",
                summary[:500],
            )

            logger.info(
                "Gemini summarize_results completed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    "query_preview": query[:100],
                    "success": True,
                },
            )

        except Exception:
            duration_s = time.perf_counter() - start_time
            logger.exception(
                "Gemini summarize_results failed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "query_preview": query[:100],
                    "success": False,
                },
            )
            summary = (
                "Sorry, I couldn't generate a summary at this time. "
                "Please try modifying your query or trying again later."
            )

        return summary

    def answer_with_citations(self, query: str, docs: str) -> str:
        """
        Use Gemini to answer a query based on retrieved documents and include
        inline citations like [1], [2], etc.
        """
        # Build the prompt incrementally in your preferred style
        prompt = gemini_rag_citations(query, docs)
        # Log start
        logger.info(
            "Gemini answer_with_citations starting",
            extra={
                "model": self.model_name,
                "query_preview": query[:100],
            },
        )
        logger.debug(
            "answer_with_citations prompt preview: %s",
            prompt[:2000],
        )
        logger.debug(
            "answer_with_citations documents preview: %s",
            docs[:2000],
        )

        start_time = time.perf_counter()
        answer: str = ""

        try:
            response = self._generate_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                ),
                context={
                    "kind": "answer_with_citations",
                    "model": self.model_name,
                },
            )

            answer = getattr(response, "text", "").strip()

            duration_s = time.perf_counter() - start_time
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            response_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)

            logger.debug(
                "answer_with_citations answer preview: %s",
                answer[:500],
            )

            logger.info(
                "Gemini answer_with_citations completed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    "query_preview": query[:100],
                    "success": True,
                },
            )

        except Exception:
            duration_s = time.perf_counter() - start_time
            logger.exception(
                "Gemini answer_with_citations failed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "query_preview": query[:100],
                    "success": False,
                },
            )
            answer = (
                "Sorry, I couldn't generate a citation-backed answer at this time. "
                "Please try again later."
            )

        return answer

    def answer_question(self, question: str, context: str) -> str:
        """
        Use Gemini to answer a user's question based on movie search results.

        The answer should be:
        - direct and concise
        - casual and conversational
        - not cringe or hype-y
        - like a normal chat reply for Hoopla users.
        """
        # Build the prompt incrementally
        prompt = gemini_rag_answer(question, context)
        # Log start
        logger.info(
            "Gemini answer_question starting",
            extra={
                "model": self.model_name,
                "question_preview": question[:100],
            },
        )
        logger.debug(
            "answer_question prompt preview: %s",
            prompt[:2000],
        )
        logger.debug(
            "answer_question context preview: %s",
            context[:2000],
        )

        start_time = time.perf_counter()
        answer: str = ""

        try:
            response = self._generate_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                ),
                context={
                    "kind": "answer_question",
                    "model": self.model_name,
                },
            )

            answer = getattr(response, "text", "").strip()

            duration_s = time.perf_counter() - start_time
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            response_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)

            logger.debug(
                "answer_question answer preview: %s",
                answer[:500],
            )

            logger.info(
                "Gemini answer_question completed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    "question_preview": question[:100],
                    "success": True,
                },
            )

        except Exception:
            duration_s = time.perf_counter() - start_time
            logger.exception(
                "Gemini answer_question failed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "question_preview": question[:100],
                    "success": False,
                },
            )
            answer = (
                "Sorry, I couldn't generate an answer right now. "
                "Please try again in a bit."
            )

        return answer

    def rewrite_multimodal_query(
        self,
        image_path: str,
        query: str,
    ) -> tuple[str, Optional[int]]:
        """
        Use Gemini multimodal (image + text) to rewrite a search query.

        Args:
            image_path: path to the image file.
            query: original text query to rewrite.

        Returns:
            (rewritten_query, total_token_count_or_None)
        """
        query = query.strip()

        # Guess MIME type from file extension, default to image/jpeg
        mime, _ = mimetypes.guess_type(image_path)
        mime = mime or "image/jpeg"

        logger.info(
            "Gemini rewrite_multimodal_query starting",
            extra={
                "model": self.model_name,
                "image_path": image_path,
                "mime": mime,
                "query_preview": query[:100],
            },
        )

        # Read image bytes
        try:
            with open(image_path, "rb") as f:
                img = f.read()
        except OSError as exc:
            logger.exception(
                "Failed to read image file for rewrite_multimodal_query",
                extra={"image_path": image_path},
            )
            # Fall back: no rewrite, return original query
            return query, None

        # System prompt describing task
        system_prompt = (
            "Given the included image and text query, rewrite the text query to improve search "
            "results from a movie database. Make sure to:\n"
            "- Synthesize visual and textual information\n"
            "- Focus on movie-specific details (actors, scenes, style, etc.)\n"
            "- Return only the rewritten query, without any additional commentary"
        )

        # Build parts: system prompt, image bytes, and original query
        parts = [
            system_prompt,
            genai_types.Part.from_bytes(data=img, mime_type=mime),
            query,
        ]

        logger.debug(
            "rewrite_multimodal_query: sending multimodal request with mime=%s and query_preview=%s",
            mime,
            query[:100],
        )

        start_time = time.perf_counter()
        rewritten: str = query
        total_tokens: Optional[int] = None

        try:
            response = self._generate_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=parts,
                ),
                context={
                    "kind": "rewrite_multimodal_query",
                    "model": self.model_name,
                    "mime": mime,
                },
            )

            rewritten = (getattr(response, "text", "") or query).strip()

            usage = getattr(response, "usage_metadata", None)
            total_tokens = getattr(usage, "total_token_count", None)
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            response_tokens = getattr(usage, "candidates_token_count", None)

            duration_s = time.perf_counter() - start_time

            logger.debug(
                "rewrite_multimodal_query: rewritten_preview=%s",
                rewritten[:200],
            )

            logger.info(
                "Gemini rewrite_multimodal_query completed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    "query_preview": query[:100],
                    "rewritten_preview": rewritten[:100],
                    "success": True,
                },
            )

        except Exception:
            duration_s = time.perf_counter() - start_time
            logger.exception(
                "Gemini rewrite_multimodal_query failed",
                extra={
                    "model": self.model_name,
                    "duration_s": duration_s,
                    "query_preview": query[:100],
                    "image_path": image_path,
                    "success": False,
                },
            )
            # On failure, just return original query and no token info
            rewritten = query
            total_tokens = None

        return rewritten, total_tokens
