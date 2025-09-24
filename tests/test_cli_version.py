from typer.testing import CliRunner

from douglas import __version__
from douglas.cli import app


def test_cli_supports_version_flag():
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"Douglas {__version__}"


def test_cli_supports_short_version_flag():
    runner = CliRunner()
    result = runner.invoke(app, ["-V"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"Douglas {__version__}"
