from __future__ import annotations

from . import cli_commands as _commands

# Preserve the historical monkeypatch/import surface while the implementation
# lives behind the command application module.
_PATCHABLE = tuple(
    name
    for name in dir(_commands)
    if not name.startswith("_") and name not in {"main", "build_parser"}
)
globals().update({name: getattr(_commands, name) for name in _PATCHABLE})

build_parser = _commands.build_parser


def main(argv: list[str] | None = None) -> int:
    for name in _PATCHABLE:
        setattr(_commands, name, globals()[name])
    return _commands.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
