"""vLLM OpenAI-compatible inference client."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import Settings
from .preprocessor import FunctionRecord


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FunctionInference:
    index: int
    name: str
    success: bool
    output: str
    error: str | None = None


class VllmInferenceClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def health(self) -> bool:
        request = urllib.request.Request(
            f"{self.settings.vllm_base_url}/models",
            headers=self._headers(),
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    def infer_functions(
        self,
        functions: list[FunctionRecord],
        *,
        task_id: str | None = None,
        deadline: float | None = None,
        task_timeout_seconds: int | None = None,
    ) -> list[FunctionInference]:
        tag = f"task={task_id} " if task_id else ""
        llm_functions = [function for function in functions if function.send_to_llm]
        skipped = len(functions) - len(llm_functions)
        started_at = time.monotonic()
        logger.info(
            "%sLLM inference started: total=%d refine=%d skipped=%d",
            tag,
            len(functions),
            len(llm_functions),
            skipped,
        )

        results: list[FunctionInference] = []
        refined_index = 0
        for function in functions:
            if not function.send_to_llm:
                results.append(
                    FunctionInference(
                        index=function.index,
                        name=function.name,
                        success=False,
                        output=function.code,
                        error="Function filtered out before inference",
                    )
                )
                continue

            refined_index += 1
            if deadline is not None and time.monotonic() >= deadline:
                timeout_label = task_timeout_seconds or int(deadline - started_at)
                raise RuntimeError(f"Decompilation timed out after {timeout_label}s")

            function_started_at = time.monotonic()
            try:
                request_timeout = self.settings.vllm_timeout_seconds
                if deadline is not None:
                    request_timeout = max(
                        1,
                        min(
                            int(deadline - time.monotonic()),
                            self.settings.vllm_timeout_seconds,
                        ),
                    )
                output = self._infer_one(function.code, timeout_seconds=request_timeout)
                duration = time.monotonic() - function_started_at
                logger.info(
                    "%sLLM inference [%d/%d] finished: %s (%.1fs, ok)",
                    tag,
                    refined_index,
                    len(llm_functions),
                    function.name,
                    duration,
                )
                results.append(
                    FunctionInference(
                        index=function.index,
                        name=function.name,
                        success=True,
                        output=output,
                    )
                )
            except Exception as exc:
                duration = time.monotonic() - function_started_at
                logger.warning(
                    "%sLLM inference [%d/%d] finished: %s (%.1fs, failed: %s)",
                    tag,
                    refined_index,
                    len(llm_functions),
                    function.name,
                    duration,
                    exc,
                )
                results.append(
                    FunctionInference(
                        index=function.index,
                        name=function.name,
                        success=False,
                        output=function.code,
                        error=str(exc),
                    )
                )

        success_count = sum(1 for item in results if item.success)
        failed_count = sum(1 for item in results if not item.success and item.error != "Function filtered out before inference")
        logger.info(
            "%sLLM inference finished: refined=%d success=%d failed=%d duration=%.1fs",
            tag,
            len(llm_functions),
            success_count,
            failed_count,
            time.monotonic() - started_at,
        )
        return results

    def _infer_one(self, ghidra_code: str, *, timeout_seconds: int | None = None) -> str:
        payload = {
            "model": self.settings.vllm_model,
            "prompt": self._prompt(ghidra_code),
            "temperature": 0,
            "max_tokens": self.settings.vllm_max_tokens,
        }
        request = urllib.request.Request(
            f"{self.settings.vllm_base_url}/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={**self._headers(), "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds or self.settings.vllm_timeout_seconds,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {exc.code}: {body}") from exc

        try:
            return data["choices"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("vLLM returned an unexpected response") from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.vllm_api_key}"
        return headers

    def _prompt(self, ghidra_code: str) -> str:
        return f"# This is the assembly code:\n{ghidra_code.strip()}\n# What is the source code?\n"
