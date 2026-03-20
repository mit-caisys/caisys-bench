"""
Self-RAG utility functions.

Helper functions for text formatting and output display.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def split_sentence(sentence: str, max_len: int = 80):
    words = sentence.split()
    chunks = []
    current_chunk = ""

    for word in words:
        test_chunk = current_chunk + ("" if current_chunk == "" else " ") + word
        encoded_len = len(test_chunk.encode("unicode_escape").decode("ascii"))

        if encoded_len > max_len:
            chunks.append(current_chunk)
            current_chunk = word
        else:
            current_chunk = test_chunk

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def opt_print(opt: bool, *args, **kwargs):
    if opt:
        print(*args, **kwargs)


def opt_print_factory(opt: bool):
    if opt:
        return print
    return lambda *args, **kwargs: None
