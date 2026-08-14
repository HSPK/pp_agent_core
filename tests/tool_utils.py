"""Python port of `packages/agent/test/utils/calculate.ts` and
`packages/agent/test/utils/get-current-time.ts`.

TypeScript evaluates the expression with `new Function("return <expr>")`.
Python has no equivalent that is safe to call on test input, so `calculate`
walks a parsed `ast` of arithmetic operators instead. The observable contract
is the same: it returns `"<expression> = <result>"` and raises on anything it
cannot evaluate.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from datetime import datetime
from typing import Any

from pi_ai import TextContent, Usage

from pi_agent.types import AgentTool, AgentToolResult

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _format_number(value: float | int) -> str:
    """Render like JavaScript: whole floats lose their `.0` suffix."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def calculate(expression: str) -> AgentToolResult:
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _evaluate(parsed)
    except Exception as error:
        raise ValueError(str(error) or repr(error)) from error
    return AgentToolResult(
        content=[TextContent(text=f"{expression} = {_format_number(result)}")],
        details=None,
    )


CALCULATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {"type": "string", "description": "The mathematical expression to evaluate"},
    },
    "required": ["expression"],
}


async def _execute_calculate(tool_call_id: str, args: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
    return calculate(args["expression"])


calculate_tool = AgentTool(
    label="Calculator",
    name="calculate",
    description="Evaluate mathematical expressions",
    parameters=CALCULATE_SCHEMA,
    execute=_execute_calculate,
)


def create_calculate_tool_with_usage(usage: Usage) -> AgentTool:
    async def execute(tool_call_id: str, args: dict[str, Any], signal=None, on_update=None) -> AgentToolResult:
        result = calculate(args["expression"])
        result.usage = usage
        return result

    return AgentTool(
        label=calculate_tool.label,
        name=calculate_tool.name,
        description=calculate_tool.description,
        parameters=calculate_tool.parameters,
        execute=execute,
    )


async def get_current_time(timezone: str | None = None) -> AgentToolResult:
    date = datetime.now().astimezone()
    if timezone:
        try:
            from zoneinfo import ZoneInfo

            localized = date.astimezone(ZoneInfo(timezone))
        except Exception as error:
            raise ValueError(f"Invalid timezone: {timezone}. Current UTC time: {date.isoformat()}") from error
        return AgentToolResult(
            content=[TextContent(text=localized.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z"))],
            details={"utcTimestamp": int(date.timestamp() * 1000)},
        )
    return AgentToolResult(
        content=[TextContent(text=date.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z"))],
        details={"utcTimestamp": int(date.timestamp() * 1000)},
    )


GET_CURRENT_TIME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "timezone": {
            "type": "string",
            "description": "Optional timezone (e.g., 'America/New_York', 'Europe/London')",
        },
    },
}


async def _execute_get_current_time(
    tool_call_id: str, args: dict[str, Any], signal=None, on_update=None
) -> AgentToolResult:
    return await get_current_time(args.get("timezone"))


get_current_time_tool = AgentTool(
    label="Current Time",
    name="get_current_time",
    description="Get the current date and time",
    parameters=GET_CURRENT_TIME_SCHEMA,
    execute=_execute_get_current_time,
)
