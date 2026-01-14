"""
Entrypoint for CLI.

This is the main entry point for the fif_recsys CLI application.
It provides configuration management capabilities through the config command.
"""
import typer

from fif_recsys.commands import config, data, feature, model, policy

app = typer.Typer(
    no_args_is_help=True,
    help="Ranking Brazilian fixed income funds",
    rich_markup_mode="rich",
)
app.add_typer(config.app, name="config", help="manage configuration settings")
app.add_typer(data.app, name="data")
app.add_typer(feature.app, name="feature")
app.add_typer(model.app, name="model")
app.add_typer(policy.app, name="policy")


def fif():
    app()


if __name__ == "__main__":
    app()
