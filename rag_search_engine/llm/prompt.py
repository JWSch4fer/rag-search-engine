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
