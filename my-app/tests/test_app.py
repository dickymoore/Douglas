from app import get_welcome_message


def test_get_welcome_message_includes_project_name():
    message = get_welcome_message()
    assert "my-app" in message
