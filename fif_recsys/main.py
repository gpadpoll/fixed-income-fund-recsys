"""
Entrypoint for CLI.

This is the main entry point for the fif_recsys CLI application.
It provides configuration management capabilities through the config command.
"""
import typer

from fif_recsys.commands import config

app = typer.Typer(
    no_args_is_help=True,
    help="Ranking Brazilian fixed income funds",
    rich_markup_mode="rich",
)
app.add_typer(config.app, name="config")


def fif():
    app()


if __name__ == "__main__":
    app()
