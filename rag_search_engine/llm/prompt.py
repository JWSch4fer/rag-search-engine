from typing import Dict, Any, List


def gemini_method(method: str, query: str) -> str:
    match method:
        case "spell":
            SPELL_CHECK = "Fix any spelling errors in this movie search query.\n\n"
            SPELL_CHECK += (
                "Only correct obvious typos. Don't change correctly spelled words.\n\n"
            )
            SPELL_CHECK += f'Query: "{query}"\n\n'
            SPELL_CHECK += "Examples:\n\n"
            SPELL_CHECK += " - l33t 5p3ak -> leet speak\n"
            SPELL_CHECK += " - nrml eror -> normal errors\n"
            SPELL_CHECK += " - tooo maany charraccters -> too many characters\n\n"
            SPELL_CHECK += "If no errors, return the original query.\n"
            SPELL_CHECK += "Corrected:"
            return SPELL_CHECK

        case "rewrite":
            REWRITE = "Rewrite this movie search query to be more specific and searchable.\n\n"
            REWRITE += f'Original: "{query}"\n\n'
            REWRITE += "Consider:\n"
            REWRITE += " - Common movie knowledge (famous actors, popular films)\n"
            REWRITE += " - Genre conventions (horror = scary, animation = cartoon)\n"
            REWRITE += " - Keep it concise (under 10 words)\n"
            REWRITE += (
                " - It should be a google style search query that's very specific\n"
            )
            REWRITE += " - Don't use boolean logic\n\n"
            REWRITE += "Examples:\n"
            REWRITE += ' - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"\n'
            REWRITE += ' - "movie about bear in london with marmalade" -> "Paddington London marmalade"\n'
            REWRITE += ' - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"\n\n'
            REWRITE += "Rewritten query:"
            return REWRITE

        case "expand":
            EXPAND = f"Expand this movie search query with related terms.\n\n"
            EXPAND += "Add synonyms and related concepts that might appear in movie descriptions.\n"
            EXPAND += "Keep expansions relevant and focused.\n"
            EXPAND += "This will be appended to the original query.\n"
            EXPAND += "Examples:\n\n"
            EXPAND += '- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"\n'
            EXPAND += '- "action movie with bear" -> "action thriller bear chase fight adventure"\n'
            EXPAND += (
                '- "comedy with bear" -> "comedy funny bear humor lighthearted"\n\n'
            )
            EXPAND += f'Query: "{query}"\n\n'
            EXPAND += "Expanded query:"
            return EXPAND

        case _:
            return ""


def gemini_reranking(
    query: str, docs: Dict[str, Any] | List[Dict[str, Any]], method: str
) -> str:
    match method:
        case "individual":
            INDIVIDUAL = "Rate how well this movie matches the search query.\n"
            INDIVIDUAL += "NOTE: you may not ask follow up questions the response should be a score.\n\n"
            INDIVIDUAL += f'Query: "{query}"\n\n'
            INDIVIDUAL += f"\"Movie: {docs['title']} - {docs['description']}\n\n\""
            INDIVIDUAL += "Consider:\n"
            INDIVIDUAL += " - Direct relevance to query\n"
            INDIVIDUAL += " - User intent (what they're looking for)\n"
            INDIVIDUAL += " - Content appropriateness\n\n"
            INDIVIDUAL += "Rate 0-10 (10 = perfect match).\n"
            INDIVIDUAL += "Give me ONLY the number in your response, no other text or explanation.\n"
            INDIVIDUAL += "Score:"
            return INDIVIDUAL
        case "batch":
            # Build the Movies: section
            lines = []
            for doc in docs:
                doc_id = doc.get("id")
                title = doc.get("title", "")
                desc = doc.get("description", "")
                desc = desc.replace("\n", " ").strip()
                lines.append(f"ID: {doc_id}\nTitle: {title}\nDescription: {desc}")

            BATCH = "Rank these movies by relevance to the search query.\n\n"
            BATCH += f'Query: "{query}"\n\n'
            BATCH += "Movies:\n"
            BATCH += "{:}".format("\n\n".join(lines))
            BATCH += "-------------------------------------------------------------\n"
            BATCH += "Return ONLY the IDs in order of relevance (best match first).\n"
            BATCH += "Return a valid JSON list, nothing else. For example:\n"
            BATCH += "[75, 12, 34, 2, 1]\n"
            return BATCH
        case _:
            return ""


def gemini_evaluation(query: str, formatted_results: List[str]) -> str:
    EVALUATION = "Rate how relevant each result is to this query on a 0-3 scale:\n\n"
    EVALUATION += f"Query: {query}\n\n"
    EVALUATION += "Results:\n"
    EVALUATION += "\n".join(formatted_results) + "\n\n"
    EVALUATION += "Scale:\n"
    EVALUATION += "- 3: Highly relevant\n"
    EVALUATION += "- 2: Relevant\n"
    EVALUATION += "- 1: Marginally relevant\n"
    EVALUATION += "- 0: Not relevant\n\n"
    EVALUATION += "Do NOT give any numbers out than 0, 1, 2, or 3.\n\n"
    EVALUATION += (
        "Return ONLY the scores in the same order you were given the documents. "
    )
    EVALUATION += "Return a valid JSON list, nothing else. For example:\n\n"
    EVALUATION += "[2, 0, 3, 2, 0, 1]"
    return EVALUATION
