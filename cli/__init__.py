"""Front end over the modules above: `python -m cli <subcommand>`.

`cli.app` holds the subcommands and the parser. This module exports `main` so
`python -m cli` and an `import cli` both reach the same entry point.
"""
from cli.app import main

__all__ = ["main"]
