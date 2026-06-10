"""Dataclass-backed configuration with automatic CLI binding.

Configs are plain ``@dataclass`` objects. :func:`build_parser` introspects a
(possibly nested) dataclass and exposes every field as a command-line flag,
using dotted names for nested dataclasses (e.g. ``--data.dataset ml-1m``).
:func:`from_namespace` reconstructs the dataclass tree from parsed args. No YAML.

Supported field types: int, float, str, bool, Optional[...] of those, and
list[int]/list[str]/list[float]. Booleans become ``--flag/--no-flag`` pairs.
"""

from __future__ import annotations

import argparse
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints

__all__ = ["build_parser", "from_namespace", "parse_args"]

_NONE_TYPE = type(None)


def _resolve_hints(config_cls: type) -> dict[str, Any]:
    """Resolve field annotations to real types.

    ``from __future__ import annotations`` turns annotations into strings, so
    ``field.type`` is unreliable; resolve them against the class's module.
    """
    return get_type_hints(config_cls)


def _is_optional(tp: Any) -> bool:
    return get_origin(tp) is Union and _NONE_TYPE in get_args(tp)


def _unwrap_optional(tp: Any) -> Any:
    if _is_optional(tp):
        return next(a for a in get_args(tp) if a is not _NONE_TYPE)
    return tp


def _field_default(f) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return None


def _add_field(parser: argparse.ArgumentParser, f, tp: Any, prefix: str) -> None:
    tp = _unwrap_optional(tp)
    flag = f"--{prefix}{f.name}".replace("_", "-")
    dest = f"{prefix}{f.name}"
    default = _field_default(f)
    help_txt = f.metadata.get("help", "") if f.metadata else ""

    origin = get_origin(tp)
    if tp is bool:
        parser.add_argument(
            flag, dest=dest, default=default,
            action=argparse.BooleanOptionalAction, help=help_txt,
        )
    elif origin in (list, tuple):
        (elem_tp,) = get_args(tp) or (str,)
        parser.add_argument(
            flag, dest=dest, default=default, nargs="*",
            type=elem_tp, help=help_txt,
        )
    elif tp in (int, float, str):
        parser.add_argument(flag, dest=dest, default=default, type=tp, help=help_txt)
    else:
        # Fallback: treat as string; reconstruction passes it through unchanged.
        parser.add_argument(flag, dest=dest, default=default, type=str, help=help_txt)


def build_parser(
    config_cls: type,
    parser: argparse.ArgumentParser | None = None,
    prefix: str = "",
) -> argparse.ArgumentParser:
    """Build (or extend) an argparse parser from a dataclass type."""
    if not is_dataclass(config_cls):
        raise TypeError(f"{config_cls!r} is not a dataclass")
    if parser is None:
        parser = argparse.ArgumentParser(
            description=config_cls.__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    hints = _resolve_hints(config_cls)
    for f in fields(config_cls):
        inner = _unwrap_optional(hints[f.name])
        if isinstance(inner, type) and is_dataclass(inner):
            build_parser(inner, parser, prefix=f"{prefix}{f.name}.")
        else:
            _add_field(parser, f, hints[f.name], prefix)
    return parser


def from_namespace(config_cls: type, ns: argparse.Namespace, prefix: str = "") -> Any:
    """Reconstruct a (nested) dataclass instance from a parsed namespace."""
    values = vars(ns)
    hints = _resolve_hints(config_cls)
    kwargs: dict[str, Any] = {}
    for f in fields(config_cls):
        inner = _unwrap_optional(hints[f.name])
        if isinstance(inner, type) and is_dataclass(inner):
            kwargs[f.name] = from_namespace(inner, ns, prefix=f"{prefix}{f.name}.")
        else:
            kwargs[f.name] = values[f"{prefix}{f.name}"]
    return config_cls(**kwargs)


def parse_args(config_cls: type, argv: list[str] | None = None) -> Any:
    """Convenience: build a parser, parse ``argv``, and return a config instance."""
    parser = build_parser(config_cls)
    ns = parser.parse_args(argv)
    return from_namespace(config_cls, ns)
