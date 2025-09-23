from src import main


def test_greet_returns_custom_message() -> None:
    assert main.greet("$project_name") == "Hello, $project_name!"
