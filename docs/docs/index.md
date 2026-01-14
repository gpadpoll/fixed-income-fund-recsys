# Welcome to fif_recsys

Ranking Brazilian fixed income funds.

## Start Here!

See the `Notebooks > E2E Example` tab in the docs on a full minimal recsys example on brazilian fixed income funds.

To automate this as a scheduled job, we have a Dockerfile that runs the full pipeline based on a config YAML. In a production scenario, we would have it as an Airflow job and evalute which steps should be splitted into specific airflow tasks. But for now, a cron job that executes the full pipeline as a whole is enough, refer to the `manifest.yaml` for the config YAML.


To run it as a docker container, run the commands:

> make docker-image

> docker run --rm -v "$(pwd)/manifest.yaml:/manifest.yaml" -v "/tmp/fif_data:/data" acli pipeline /manifest.yaml /data

## Installation

```bash
pip install fif_recsys
```

## Usage

The CLI provides a set of commands:

```bash
fif data --help
fif feature --help
fif model --help
fif policy --help
```

## Notebooks

Interactive examples are available as Jupyter notebooks and are converted to Markdown for inclusion in the site. To regenerate the converted docs locally run:

```bash
make convert-notebooks
```

- **Example notebook**: `notebooks/example.md`


## Development

For developers who want to extend this CLI or use it as a template:

### Testing

Run the test suite:

```bash
pytest tests/ -v
```

### Building Documentation

Build the documentation locally:

```bash
mkdocs serve -f docs/mkdocs.yml
```

The CLI is built with:
- [Typer](https://typer.tiangolo.com/) - Modern CLI framework
- [Rich](https://rich.readthedocs.io/) - Rich text and beautiful formatting
- [Pytest](https://pytest.org/) - Testing framework

