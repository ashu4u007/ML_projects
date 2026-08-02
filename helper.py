from typing import Any, Dict, List, Optional, TypeVar
import json
import os

T = TypeVar("T")

__all__ = [
    "load_json",
    "save_json",
    "ensure_dir",
    "read_lines",
    "write_lines",
    "chunk_list",
]


def load_json(path: str) -> Dict[str, Any]:
    """Load and return JSON content from a file.

    Parameters
    ----------
    path : str
        Filesystem path to a JSON file.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON as a Python dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file content is not valid JSON.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(obj: Any, path: str, indent: Optional[int] = 2) -> None:
    """Serialize an object as JSON and write it to a file.

    Creates parent directories if they do not exist.

    Parameters
    ----------
    obj : Any
        JSON-serializable Python object to write.
    path : str
        Destination file path.
    indent : Optional[int]
        Indentation level for pretty printing. If None, the most compact
        representation is written.
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)


def ensure_dir(path: str) -> None:
    """Ensure that a directory exists, creating it (and parents) if needed.

    Parameters
    ----------
    path : str
        Directory path to ensure exists. When an empty string or "." is
        provided, no action is taken.
    """
    if not path or path == ".":
        return
    os.makedirs(path, exist_ok=True)


def read_lines(path: str, encoding: str = "utf-8") -> List[str]:
    """Read a text file and return a list of lines (without trailing newlines).

    Parameters
    ----------
    path : str
        Path to the text file.
    encoding : str
        File encoding to use.

    Returns
    -------
    List[str]
        Lines from the file with trailing newlines stripped.
    """
    with open(path, "r", encoding=encoding) as fh:
        return [line.rstrip("\n") for line in fh]


def write_lines(lines: List[str], path: str, encoding: str = "utf-8") -> None:
    """Write an iterable of lines to a text file, adding newlines.

    Parameters
    ----------
    lines : List[str]
        Sequence of lines to write. Newline characters will be appended.
    path : str
        Destination file path.
    encoding : str
        File encoding to use.
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding=encoding) as fh:
        for line in lines:
            fh.write(f"{line}\n")


def chunk_list(lst: List[T], n: int) -> List[List[T]]:
    """Split a list into consecutive chunks of size `n`.

    The last chunk may be smaller if the list length is not divisible by
    `n`.

    Parameters
    ----------
    lst : List[T]
        Input list to split.
    n : int
        Desired chunk size (must be >= 1).

    Returns
    -------
    List[List[T]]
        A list of list chunks.

    Raises
    ------
    ValueError
        If `n` is less than 1.
    """
    if n < 1:
        raise ValueError("chunk size n must be >= 1")
    return [lst[i : i + n] for i in range(0, len(lst), n)]
