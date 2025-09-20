import typer
from douglas.core import Douglas

app = typer.Typer()

@app.command()
def init(project: str = typer.Option(..., help="Name of the new project"), non_interactive: bool = False):
    """Initialize a new project with Douglas configuration and templates."""
    d = Douglas()
    d.init_project(project_name=project, non_interactive=non_interactive)

@app.command()
def run():
    """Run the Douglas AI development loop."""
    d = Douglas()
    d.run_loop()

@app.command()
def check():
    """Perform checks on configuration and environment."""
    d = Douglas()
    d.check()

@app.command()
def doctor():
    """Diagnose the local environment and tools."""
    d = Douglas()
    d.doctor()

if __name__ == '__main__':
    app()
