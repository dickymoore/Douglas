import typer

app = typer.Typer()


@app.command()
def hello():
    print("Douglas CLI is ready.")


if __name__ == "__main__":
    app()
