from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from PIL import Image

from .types import AiDecision, AiSettings, BBox, DocumentPage, MaterialSpec


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class CandidateRegion:
    candidate_id: str
    label: str
    bbox: BBox
    metadata: dict[str, Any]


class HybridAiResolver:
    def __init__(self, settings: AiSettings | None = None):
        self.settings = settings or AiSettings()
        self.decisions: list[AiDecision] = []
        self._provider_cache: tuple[str | None, str | None, str | None] | None = None

    def _record_fallback(
        self,
        decision_type: str,
        fallback_reason: str,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        self.decisions.append(
            AiDecision(
                decision_type=decision_type,
                used=False,
                fallback_used=True,
                fallback_reason=fallback_reason,
                rationale=fallback_reason,
                evidence=evidence or {},
            )
        )

    def _record_used(
        self,
        decision_type: str,
        provider: str,
        model: str,
        selected: Any,
        confidence: float | None,
        rationale: str,
        evidence: dict[str, Any] | None,
        raw_response: dict[str, Any] | None,
    ) -> None:
        self.decisions.append(
            AiDecision(
                decision_type=decision_type,
                used=True,
                provider=provider,
                model=model,
                selected=selected,
                confidence=confidence,
                rationale=rationale,
                fallback_used=False,
                evidence=evidence or {},
                raw_response=raw_response or {},
            )
        )

    def _http_json(
        self,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        body = None
        method = "GET"
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            method = "POST"
        req = request.Request(url, data=body, method=method)
        req.add_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def _detect_provider(self) -> tuple[str | None, str | None, str | None]:
        if self._provider_cache is not None:
            return self._provider_cache

        if self.settings.mode == "off":
            self._provider_cache = (None, None, None)
            return self._provider_cache

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            openai_base_url = self.settings.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            openai_model = self.settings.model or os.getenv("OPENAI_MODEL") or "gpt-4o"
            self._provider_cache = ("openai_compat", openai_base_url.rstrip("/"), openai_model)
            return self._provider_cache

        explicit_base_url = self.settings.base_url or os.getenv("ESTIMATOR_AI_BASE_URL")
        explicit_model = self.settings.model or os.getenv("ESTIMATOR_AI_MODEL")

        ollama_url = os.getenv("ESTIMATOR_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        ollama_model = self.settings.model or os.getenv("ESTIMATOR_OLLAMA_MODEL") or "llava"
        try:
            self._http_json(f"{ollama_url}/api/tags", timeout=1.5)
            self._provider_cache = ("ollama", ollama_url, ollama_model)
            return self._provider_cache
        except Exception:
            pass

        if explicit_base_url and explicit_model:
            self._provider_cache = ("openai_compat", explicit_base_url.rstrip("/"), explicit_model)
            return self._provider_cache

        self._provider_cache = (None, None, None)
        return self._provider_cache

    def _image_base64(self, image_path: Path, max_dim: int = 1600) -> str:
        image = Image.open(image_path).convert("RGB")
        if max(image.size) > max_dim:
            scale = max_dim / max(image.size)
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _parse_json_content(self, content: Any) -> dict[str, Any]:
        if isinstance(content, dict):
            return content
        if isinstance(content, list):
            content = "\n".join(str(item.get("text", item)) for item in content)
        text = str(content).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = JSON_BLOCK_RE.search(text)
            if not match:
                raise
            return json.loads(match.group(0))

    def _chat_json(
        self,
        decision_type: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        provider, base_url, model = self._detect_provider()
        if provider is None or base_url is None or model is None:
            if self.settings.mode == "require":
                raise RuntimeError("AI resolution was required but no local or configured model endpoint was available.")
            self._record_fallback(
                decision_type,
                "AI endpoint unavailable; deterministic fallback used.",
                evidence=evidence,
            )
            return None

        try:
            if provider == "ollama":
                images = [self._image_base64(path) for path in (image_paths or [])]
                payload = {
                    "model": model,
                    "stream": False,
                    "format": "json",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt, "images": images},
                    ],
                }
                response_payload = self._http_json(f"{base_url}/api/chat", payload=payload, timeout=35.0)
                parsed = self._parse_json_content(response_payload.get("message", {}).get("content", ""))
                parsed["_provider"] = provider
                parsed["_model"] = model
                parsed["_raw"] = response_payload
                return parsed

            if provider == "openai_compat":
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ESTIMATOR_AI_API_KEY")
                content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
                for path in image_paths or []:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{self._image_base64(path)}"},
                        }
                    )
                payload = {
                    "model": model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                }
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                response_payload = self._http_json(f"{base_url}/chat/completions", payload=payload, headers=headers, timeout=35.0)
                parsed = self._parse_json_content(response_payload.get("choices", [{}])[0].get("message", {}).get("content", ""))
                parsed["_provider"] = provider
                parsed["_model"] = model
                parsed["_raw"] = response_payload
                return parsed
        except (error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError, OSError, KeyError, IndexError, ValueError) as exc:
            if self.settings.mode == "require":
                raise RuntimeError(f"AI resolution failed during {decision_type}: {exc}") from exc
            self._record_fallback(
                decision_type,
                f"AI resolution failed ({exc}); deterministic fallback used.",
                evidence=evidence,
            )
            return None

        if self.settings.mode == "require":
            raise RuntimeError(f"Unsupported AI provider: {provider}")
        self._record_fallback(decision_type, "Unsupported AI provider; deterministic fallback used.", evidence=evidence)
        return None

    def choose_plan_region(
        self,
        page: DocumentPage,
        candidates: list[CandidateRegion],
    ) -> str | None:
        if self.settings.mode == "off" or len(candidates) < 2:
            return None

        prompt = {
            "task": "Choose which candidate region should drive exterior footprint measurement.",
            "rule": "Prefer the main house ground-floor plan. Exclude loft, attic, upper floor, or 'PARVEN POHJA' regions.",
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "label": candidate.label,
                    "bbox": candidate.bbox.model_dump(mode="json"),
                    "metadata": candidate.metadata,
                }
                for candidate in candidates
            ],
            "required_response_schema": {
                "selected_candidate_id": "candidate id from the list",
                "confidence": "0..1",
                "rationale": "short explanation",
            },
        }
        response = self._chat_json(
            "plan_region",
            "You are helping a deterministic estimator choose the correct plan subview. Return strict JSON only.",
            json.dumps(prompt, ensure_ascii=False),
            image_paths=[page.render_path],
            evidence={
                "page_id": page.page_id,
                "candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "label": candidate.label,
                        "bbox": candidate.bbox.model_dump(mode="json"),
                        "metadata": candidate.metadata,
                    }
                    for candidate in candidates
                ],
            },
        )
        if not response:
            return None
        selected = str(response.get("selected_candidate_id", "")).strip()
        if selected not in {candidate.candidate_id for candidate in candidates}:
            self._record_fallback(
                "plan_region",
                "AI returned an unknown candidate id; deterministic fallback used.",
                evidence={"selected": selected},
            )
            return None
        self._record_used(
            "plan_region",
            provider=str(response.get("_provider")),
            model=str(response.get("_model")),
            selected=selected,
            confidence=float(response.get("confidence", 0.0) or 0.0),
            rationale=str(response.get("rationale", "")),
            evidence={"page_id": page.page_id},
            raw_response=response.get("_raw", {}),
        )
        return selected

    def choose_wall_height(
        self,
        candidate_values_m: list[float],
        evidence: dict[str, Any],
    ) -> float | None:
        unique_values = sorted({round(value, 3) for value in candidate_values_m})
        if self.settings.mode == "off" or len(unique_values) < 2:
            return None

        prompt = {
            "task": "Choose the best exterior wall height used for facade gross wall area.",
            "rule": "Pick the repeated eave or wall-top level, not the ridge or roof peak.",
            "candidates_m": unique_values,
            "evidence": evidence,
            "required_response_schema": {
                "selected_value_m": "one numeric candidate value",
                "confidence": "0..1",
                "rationale": "short explanation",
            },
        }
        response = self._chat_json(
            "wall_height",
            "You are helping a deterministic estimator choose the wall-height marker. Return strict JSON only.",
            json.dumps(prompt, ensure_ascii=False),
            evidence={"candidate_values_m": unique_values, **evidence},
        )
        if not response:
            return None
        try:
            selected = round(float(response.get("selected_value_m")), 3)
        except (TypeError, ValueError):
            self._record_fallback("wall_height", "AI returned a non-numeric wall height; deterministic fallback used.", evidence=evidence)
            return None
        if selected not in unique_values:
            self._record_fallback("wall_height", "AI wall height was not one of the candidates; deterministic fallback used.", evidence={"selected": selected, **evidence})
            return None
        self._record_used(
            "wall_height",
            provider=str(response.get("_provider")),
            model=str(response.get("_model")),
            selected=selected,
            confidence=float(response.get("confidence", 0.0) or 0.0),
            rationale=str(response.get("rationale", "")),
            evidence=evidence,
            raw_response=response.get("_raw", {}),
        )
        return selected

    def choose_primary_material(
        self,
        specs: dict[str, MaterialSpec],
        section_material_lines: list[str],
        local_label_counts: dict[str, int],
    ) -> str | None:
        if self.settings.mode == "off" or len(specs) < 2:
            return None

        prompt = {
            "task": "Choose the primary exterior cladding code for procurement when elevation segmentation is ambiguous.",
            "rule": "Prefer the section wall-build-up note if it clearly identifies the dominant cladding.",
            "specs": {code: spec.description for code, spec in specs.items()},
            "section_material_lines": section_material_lines[:8],
            "local_label_counts": local_label_counts,
            "required_response_schema": {
                "selected_code": "one code from specs",
                "confidence": "0..1",
                "rationale": "short explanation",
            },
        }
        response = self._chat_json(
            "primary_material",
            "You are helping a deterministic estimator choose the dominant exterior cladding. Return strict JSON only.",
            json.dumps(prompt, ensure_ascii=False),
            evidence={"spec_codes": list(specs), "local_label_counts": local_label_counts},
        )
        if not response:
            return None
        selected = str(response.get("selected_code", "")).strip().lower()
        if selected not in specs:
            self._record_fallback("primary_material", "AI returned an unknown material code; deterministic fallback used.", evidence={"selected": selected})
            return None
        self._record_used(
            "primary_material",
            provider=str(response.get("_provider")),
            model=str(response.get("_model")),
            selected=selected,
            confidence=float(response.get("confidence", 0.0) or 0.0),
            rationale=str(response.get("rationale", "")),
            evidence={"local_label_counts": local_label_counts},
            raw_response=response.get("_raw", {}),
        )
        return selected
