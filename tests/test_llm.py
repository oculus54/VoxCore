from unittest.mock import patch, MagicMock

from src.llm import generate


@patch("src.llm.get_client")
def test_generate_returns_text(mock_get_client):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "mocked summary"
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    result = generate("test prompt")

    assert result == "mocked summary"
    mock_client.chat.completions.create.assert_called_once()
