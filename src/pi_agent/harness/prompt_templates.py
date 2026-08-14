"""Prompt template loading and argument substitution.

Python port of `packages/agent/src/harness/prompt-templates.ts`.

TypeScript threads an injectable `ExecutionEnv` through every filesystem call
so the harness can run against non-Node backends. This port instead reads
directly from `pathlib.Path` (see `pi_agent.harness.skills` for the same
choice, and the module-level docstring convention noted in the porting
task): the directory or file to load from is a plain path parameter, which
keeps this module trivially testable against `tmp_path` without needing a
ported `ExecutionEnv` implementation (the concrete Node backend,
`harness/env/nodejs.ts`, is out of scope for this port). `Path.is_dir()`/
`Path.is_file()` already resolve symlinks to their target kind, so the
`resolveKind`/`canonicalPath` fallback dance in the TypeScript source has no
counterpart here.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import yaml

from .types import PromptTemplate

PromptTemplateDiagnosticCode = Literal["file_info_failed", "list_failed", "read_failed", "parse_failed"]


@dataclass(kw_only=True)
class PromptTemplateDiagnostic:
    """Warning produced while loading prompt templates."""

    code: PromptTemplateDiagnosticCode
    """Stable diagnostic code."""
    message: str
    """Human-readable diagnostic message."""
    path: str
    """Path associated with the diagnostic."""
    type: Literal["warning"] = "warning"
    """Diagnostic severity. Currently only warnings are emitted."""


TSource = TypeVar("TSource")


@dataclass(kw_only=True)
class SourcedPromptTemplate(Generic[TSource]):
    prompt_template: PromptTemplate
    source: TSource


@dataclass(kw_only=True)
class SourcedPromptTemplateDiagnostic(Generic[TSource]):
    diagnostic: PromptTemplateDiagnostic
    source: TSource


def _normalize_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


async def load_prompt_templates(
    paths: str | Path | Sequence[str | Path],
) -> tuple[list[PromptTemplate], list[PromptTemplateDiagnostic]]:
    """Load prompt templates from one or more paths.

    Directory inputs load direct `.md` children non-recursively. File inputs
    load explicit `.md` files. Missing paths and non-markdown files are
    skipped. Read and parse failures are returned as diagnostics.
    """
    prompt_templates: list[PromptTemplate] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    for path in _normalize_paths(paths):
        try:
            exists = path.exists()
        except OSError as error:
            diagnostics.append(PromptTemplateDiagnostic(code="file_info_failed", message=str(error), path=str(path)))
            continue
        if not exists:
            continue
        try:
            is_dir = path.is_dir()
        except OSError as error:
            diagnostics.append(PromptTemplateDiagnostic(code="file_info_failed", message=str(error), path=str(path)))
            continue
        if is_dir:
            dir_templates, dir_diagnostics = _load_templates_from_dir(path)
            prompt_templates.extend(dir_templates)
            diagnostics.extend(dir_diagnostics)
        elif path.is_file() and path.name.endswith(".md"):
            template, file_diagnostics = _load_template_from_file(path)
            if template is not None:
                prompt_templates.append(template)
            diagnostics.extend(file_diagnostics)
    return prompt_templates, diagnostics


async def load_sourced_prompt_templates(
    inputs: Sequence[tuple[str | Path, TSource]],
    map_prompt_template: Callable[[PromptTemplate, TSource], PromptTemplate] | None = None,
) -> tuple[list[SourcedPromptTemplate[TSource]], list[SourcedPromptTemplateDiagnostic[TSource]]]:
    """Load prompt templates from source-tagged paths.

    Source values are preserved exactly and attached to every loaded prompt
    template and diagnostic. The agent package does not interpret source
    values; applications define their own provenance shape.
    """
    prompt_templates: list[SourcedPromptTemplate[TSource]] = []
    diagnostics: list[SourcedPromptTemplateDiagnostic[TSource]] = []
    for path, source in inputs:
        result_templates, result_diagnostics = await load_prompt_templates(path)
        for template in result_templates:
            mapped = map_prompt_template(template, source) if map_prompt_template else template
            prompt_templates.append(SourcedPromptTemplate(prompt_template=mapped, source=source))
        for diagnostic in result_diagnostics:
            diagnostics.append(SourcedPromptTemplateDiagnostic(diagnostic=diagnostic, source=source))
    return prompt_templates, diagnostics


def _load_templates_from_dir(directory: Path) -> tuple[list[PromptTemplate], list[PromptTemplateDiagnostic]]:
    prompt_templates: list[PromptTemplate] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    try:
        entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
    except OSError as error:
        diagnostics.append(PromptTemplateDiagnostic(code="list_failed", message=str(error), path=str(directory)))
        return prompt_templates, diagnostics

    for entry in entries:
        if not entry.is_file() or not entry.name.endswith(".md"):
            continue
        template, file_diagnostics = _load_template_from_file(entry)
        if template is not None:
            prompt_templates.append(template)
        diagnostics.extend(file_diagnostics)
    return prompt_templates, diagnostics


def _load_template_from_file(path: Path) -> tuple[PromptTemplate | None, list[PromptTemplateDiagnostic]]:
    diagnostics: list[PromptTemplateDiagnostic] = []
    try:
        raw_content = path.read_text(encoding="utf-8")
    except OSError as error:
        diagnostics.append(PromptTemplateDiagnostic(code="read_failed", message=str(error), path=str(path)))
        return None, diagnostics

    try:
        frontmatter, body = _parse_frontmatter(raw_content)
    except Exception as error:
        diagnostics.append(PromptTemplateDiagnostic(code="parse_failed", message=str(error), path=str(path)))
        return None, diagnostics

    first_line = next((line for line in body.split("\n") if line.strip()), None)
    description = frontmatter.get("description") if isinstance(frontmatter.get("description"), str) else ""
    if not description and first_line:
        description = first_line[:60]
        if len(first_line) > 60:
            description += "..."

    return (
        PromptTemplate(
            name=re.sub(r"\.md$", "", path.name, flags=re.IGNORECASE), description=description, content=body
        ),
        diagnostics,
    )


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---"):
        return {}, normalized
    end_index = normalized.find("\n---", 3)
    if end_index == -1:
        return {}, normalized
    yaml_string = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()
    return (yaml.safe_load(yaml_string) or {}), body


def parse_command_args(args_string: str) -> list[str]:
    """Parse an argument string using simple shell-style single and double quotes."""
    args: list[str] = []
    current = ""
    in_quote: str | None = None

    for char in args_string:
        if in_quote:
            if char == in_quote:
                in_quote = None
            else:
                current += char
        elif char in ('"', "'"):
            in_quote = char
        elif char in (" ", "\t"):
            if current:
                args.append(current)
                current = ""
        else:
            current += char
    if current:
        args.append(current)
    return args


def substitute_args(content: str, args: list[str]) -> str:
    """Substitute prompt template placeholders (`$1`, `$@`, `$ARGUMENTS`, `${@:N}`, `${@:N:L}`) with command arguments."""
    result = content

    def replace_positional(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        return args[index] if 0 <= index < len(args) else ""

    result = re.sub(r"\$(\d+)", replace_positional, result)

    def replace_slice(match: re.Match[str]) -> str:
        start = int(match.group(1)) - 1
        if start < 0:
            start = 0
        length = match.group(2)
        if length:
            return " ".join(args[start : start + int(length)])
        return " ".join(args[start:])

    result = re.sub(r"\$\{@:(\d+)(?::(\d+))?\}", replace_slice, result)
    all_args = " ".join(args)
    result = result.replace("$ARGUMENTS", all_args)
    result = result.replace("$@", all_args)
    return result


def format_prompt_template_invocation(template: PromptTemplate, args: list[str] | None = None) -> str:
    """Format a prompt template invocation with positional arguments."""
    return substitute_args(template.content, args or [])


__all__ = [
    "PromptTemplateDiagnostic",
    "PromptTemplateDiagnosticCode",
    "SourcedPromptTemplate",
    "SourcedPromptTemplateDiagnostic",
    "format_prompt_template_invocation",
    "load_prompt_templates",
    "load_sourced_prompt_templates",
    "parse_command_args",
    "substitute_args",
]
