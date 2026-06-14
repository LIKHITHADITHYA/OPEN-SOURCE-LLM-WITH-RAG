"""
Recursive text splitter implementation.
Splits large text blocks on structural boundaries (paragraphs, lines, spaces)
to keep semantic content intact without cutting words in half.
"""

def split_text_recursively(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """
    Recursively splits a text string into chunks, attempting to keep paragraphs,
    sentences, or words intact by looking for separators in the overlap window.

    Args:
        text: The raw string of text to split.
        chunk_size: The target maximum length of each chunk.
        chunk_overlap: The overlap size between adjacent chunks.

    Returns:
        A list of chunked strings.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # Determine the maximum end index of this chunk
        end = min(start + chunk_size, text_len)

        # If we aren't at the end of the text, try to find a natural boundary
        if end < text_len:
            # We look for separators inside the overlap window
            search_start = max(start + chunk_size - chunk_overlap, start)
            best_sep_idx = -1

            # Prioritize paragraph breaks, then newlines, then spaces
            for sep in ["\n\n", "\n", " "]:
                idx = text.rfind(sep, search_start, end)
                if idx != -1:
                    best_sep_idx = idx + len(sep)
                    break

            # If a clean separator was found in the overlap window, use it
            if best_sep_idx != -1:
                end = best_sep_idx

        # Slice and record the chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start index forward for next step, accounting for overlap
        start = end - chunk_overlap if end < text_len else end

        # Safety break if no forward progress is made or text is exhausted
        if start >= text_len or end == text_len:
            break

    return chunks
