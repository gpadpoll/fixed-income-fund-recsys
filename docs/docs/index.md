# Welcome to fif_recsys

Ranking Brazilian fixed income funds

## Installation

```bash
pip install fif_recsys
```

## Usage

The CLI provides a `config` command for managing configuration settings:

```bash
fif config --help
```

## Configuration Management

### Set Configuration Values

Set a configuration value directly:

```bash
fif config set theme dark
fif config set debug true
fif config set timeout 30
```

Set a configuration value interactively (you'll be prompted for the value):

```bash
fif config set custom_key
```

Values are automatically converted to appropriate types:
- `true`/`false` → boolean
- Numbers → integer or float
- Everything else → string

### Get Configuration Values

Retrieve a specific configuration value:

```bash
fif config get theme
fif config get debug
```

### List All Configuration

Display all configuration values in a formatted table:

```bash
fif config list
```

### Reset Configuration

Reset all configuration to default values (with confirmation prompt):

```bash
fif config reset
```

## Configuration Storage

Configuration is stored in a JSON file:
- Default location: `~/.acli_config.json`
- Custom location: Set `ACLI_CONFIG_PATH` environment variable

```bash
export ACLI_CONFIG_PATH=/path/to/my/config.json
fif config set theme dark
```

## Default Configuration

The CLI comes with the following default configuration:

| Key | Value | Type | Description |
|-----|--------|------|-------------|
| `theme` | `default` | string | UI theme setting |
| `output_format` | `table` | string | Default output format |
| `auto_save` | `true` | boolean | Auto-save settings |
| `debug` | `false` | boolean | Debug mode |

## Examples

### Basic Configuration Setup

```bash
# View current configuration
fif config list

# Set your preferred theme
fif config set theme dark

# Enable debug mode
fif config set debug true

# View updated configuration
fif config list

# Get a specific value
fif config get theme
```

### Using Custom Configuration Path

```bash
# Set custom config location
export ACLI_CONFIG_PATH="$HOME/my-project/.acli_config.json"

# Now all config commands use the custom location
fif config set project_name "My Project"
fif config list
```

### Resetting Configuration

```bash
# Reset to defaults (with confirmation)
fif config reset

# Verify reset worked
fif config list
```

## Notebooks

Interactive examples are available as Jupyter notebooks and are converted to Markdown for inclusion in the site. To regenerate the converted docs locally run:

```bash
make convert-notebooks
```

- **Example notebook**: `notebooks/example.md`


```bash
# Reset to defaults (with confirmation)
fif config reset

# Verify reset worked
fif config list
```

## Error Handling

The CLI provides helpful error messages:

```bash
# Trying to get a non-existent key
$ fif config get nonexistent
Error: Configuration key 'nonexistent' not found.

# Corrupted config file
$ fif config get theme
Error: Configuration file /path/to/config.json is corrupted.
```

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

