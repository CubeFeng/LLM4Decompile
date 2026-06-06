"""Preprocess Ghidra output into function records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .storage import TaskPaths


FUNCTION_MARKER = "// Function:"
FUNCTION_NAME_RE = re.compile(r"// Function:\s*([^\n{]+)")


@dataclass(frozen=True)
class FunctionRecord:
    index: int
    name: str
    code: str
    char_count: int
    send_to_llm: bool


@dataclass(frozen=True)
class PreprocessResult:
    functions_path: Path
    functions: list[FunctionRecord]


class Preprocessor:
    def __init__(self, min_function_chars: int = 20):
        self.min_function_chars = min_function_chars

    def process(self, raw_code_path: Path, safe_name: str, paths: TaskPaths) -> PreprocessResult:
        raw_code = raw_code_path.read_text(encoding="utf-8", errors="ignore")
        chunks = self._split_functions(raw_code)
        records: list[FunctionRecord] = []
        for index, code in enumerate(chunks, start=1):
            stripped = code.strip()
            name = self._extract_name(stripped, index)
            send_to_llm = self._is_valid_function(stripped)
            records.append(
                FunctionRecord(
                    index=index,
                    name=name,
                    code=stripped,
                    char_count=len(stripped),
                    send_to_llm=send_to_llm,
                )
            )

        functions_path = paths.functions_dir / f"{safe_name}.functions.json"
        with functions_path.open("w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "index": record.index,
                        "name": record.name,
                        "code": record.code,
                        "char_count": record.char_count,
                        "send_to_llm": record.send_to_llm,
                    }
                    for record in records
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )

        return PreprocessResult(functions_path=functions_path, functions=records)

    def _split_functions(self, raw_code: str) -> list[str]:
        if FUNCTION_MARKER not in raw_code:
            return [raw_code] if raw_code.strip() else []

        parts = raw_code.split(FUNCTION_MARKER)
        functions = []
        for part in parts:
            if not part.strip():
                continue
            functions.append(FUNCTION_MARKER + part)
        return functions

    def _extract_name(self, code: str, index: int) -> str:
        match = FUNCTION_NAME_RE.search(code)
        if not match:
            return f"function_{index}"
        name = match.group(1).strip()
        return name or f"function_{index}"

    def _is_valid_function(self, code: str) -> bool:
        if len(code) < self.min_function_chars:
            return False
        return "{" in code and "}" in code
