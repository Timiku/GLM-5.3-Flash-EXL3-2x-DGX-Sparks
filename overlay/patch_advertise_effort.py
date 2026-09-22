#!/usr/bin/env python3
"""Advertise reasoning-effort support on /v1/models.

The served chat template consumes reasoning_effort (low/high/max; unset
falls back to max) and the launcher pins GLM53_DEFAULT_REASONING_EFFORT,
but /v1/models carries no capability surface, so OpenAI clients cannot
discover the field or its accepted levels.

Adds two optional fields to ModelCard:
- reasoning_effort: levels this serving stack accepts
- default_reasoning_effort: what an omitted request resolves to
  (GLM53_DEFAULT_REASONING_EFFORT when set, else the template fallback)

Additive and optional: stock clients ignore unknown fields. Targets are
inside the image at /usr/local/lib/python3.12/dist-packages/vllm:
  entrypoints/openai/engine/protocol.py   (ModelCard)
  entrypoints/openai/models/serving.py    (show_available_models)
Validated before write; idempotent; fail-closed on drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
PROTOCOL_PATH = Path("entrypoints/openai/engine/protocol.py")
SERVING_PATH = Path("entrypoints/openai/models/serving.py")
MARK = "# [glm53-advertise-reasoning-effort]"

LEVELS = '["none", "minimal", "low", "medium", "high", "xhigh", "max"]'

PROTOCOL_OLD = """class ModelCard(OpenAIBaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm"
    root: str | None = None
    parent: str | None = None
    max_model_len: int | None = None
    permission: list[ModelPermission] = Field(default_factory=list)"""
PROTOCOL_NEW = """class ModelCard(OpenAIBaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm"
    root: str | None = None
    parent: str | None = None
    max_model_len: int | None = None
    permission: list[ModelPermission] = Field(default_factory=list)
    # [glm53-advertise-reasoning-effort] Reasoning-effort capability surface.
    # Optional and None by default: omitted from responses unless the serving
    # layer fills them in (OpenAI clients ignore unknown fields).
    reasoning_effort: list[str] | None = None
    default_reasoning_effort: str | None = None"""

SERVING_OLD = '''    async def show_available_models(self) -> ModelList:
        """Show available models (base models only)."""
        max_model_len = self.model_config.max_model_len
        return ModelList(
            data=[
                ModelCard(
                    id=base_model.name,
                    max_model_len=max_model_len,
                    root=base_model.model_path,
                    permission=[ModelPermission()],
                )
                for base_model in self.base_model_paths
            ]
        )'''

# Built with placeholder markers; backticks/braces kept out of the source.
SERVING_NEW_TMPL = '''    async def show_available_models(self) -> ModelList:
        """Show available models (base models only)."""
        max_model_len = self.model_config.max_model_len
        # [glm53-advertise-reasoning-effort] Announce the reasoning-effort
        # surface the served chat template implements (none..max; template
        # maps unset to max). default_reasoning_effort reports the launcher
        # pin GLM53_DEFAULT_REASONING_EFFORT when set, else the template
        # fallback.
        _effort_levels = @@LEVELS@@
        _default_effort = os.environ.get("GLM53_DEFAULT_REASONING_EFFORT") or "max"
        return ModelList(
            data=[
                ModelCard(
                    id=base_model.name,
                    max_model_len=max_model_len,
                    root=base_model.model_path,
                    permission=[ModelPermission()],
                    reasoning_effort=_effort_levels,
                    default_reasoning_effort=_default_effort,
                )
                for base_model in self.base_model_paths
            ]
        )'''
SERVING_NEW = SERVING_NEW_TMPL.replace("@@LEVELS@@", LEVELS)


def _replace_region(src: str, old: str, new: str) -> tuple[str, str]:
    if MARK in src:
        if src.count(MARK) != 1:
            return src, "drifted:patched-region"
        compile(src, "advertise-effort-target.py", "exec")
        return src, "skipped"
    if src.count(old) != 1:
        return src, "missing:unique-target"
    updated = src.replace(old, new, 1)
    compile(updated, "advertise-effort-target.py", "exec")
    return updated, "applied"


def apply_protocol(src: str) -> tuple[str, str]:
    return _replace_region(src, PROTOCOL_OLD, PROTOCOL_NEW)


def apply_serving(src: str) -> tuple[str, str]:
    updated, status = _replace_region(src, SERVING_OLD, SERVING_NEW)
    if status != "applied":
        return updated, status
    if "import os\n" not in updated:
        anchor = "from collections import defaultdict\n"
        if updated.count(anchor) != 1:
            return src, "drifted:os-import"
        updated = updated.replace(anchor, anchor + "import os\n", 1)
    compile(updated, "advertise-effort-serving.py", "exec")
    return updated, "applied"


def main(argv: list[str]) -> int:
    status_only = len(argv) > 1 and argv[1] == "--status"
    root_arg = 2 if status_only else 1
    root = Path(argv[root_arg]) if len(argv) > root_arg else ROOT
    changes = []
    for rel, transform in (
        (PROTOCOL_PATH, apply_protocol),
        (SERVING_PATH, apply_serving),
    ):
        target = root / rel
        if not target.is_file():
            print(f"[glm53-advertise-reasoning-effort] missing {target}",
                  file=sys.stderr)
            return 1
        original = target.read_text(encoding="utf-8")
        updated, status = transform(original)
        if status not in ("applied", "skipped"):
            print(f"[glm53-advertise-reasoning-effort] {status}: {target}",
                  file=sys.stderr)
            return 1
        changes.append((target, updated, status))
    if status_only:
        complete = all(s == "skipped" for _, _, s in changes)
        print("advertise-reasoning-effort:",
              "APPLIED" if complete else "NOT APPLIED")
        return 0
    for target, updated, status in changes:
        if status == "applied":
            target.write_text(updated, encoding="utf-8")
        print(f"[glm53-advertise-reasoning-effort] {status}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
