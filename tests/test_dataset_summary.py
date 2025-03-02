from unittest.mock import mock_open, patch

import pytest

from miexp.util.data import generate_data_summary


@pytest.fixture
def mock_os_walk():
    return [
        (".", ("subdir1",), ("bulk_conf.yaml",)),
        ("./subdir1", (), ("bulk_conf.yaml",)),
        ("./subdir2", (), ("bulk_conf.yaml",)),
    ]


@pytest.fixture
def mock_yaml_load():
    return {"key1": "value1", "key2": "value2"}


@pytest.fixture
def mock_readme_preamble():
    return "This is the preamble."


@patch("os.getcwd", return_value="/test_dir")
@patch("os.walk")
@patch("builtins.open", new_callable=mock_open, read_data="This is the preamble.")
@patch("yaml.safe_load")
def test_generate_data_summary(
    mock_yaml,
    mock_open,
    mock_walk,
    mock_getcwd,
    mock_os_walk,
    mock_yaml_load,
    mock_readme_preamble,
):
    mock_walk.return_value = mock_os_walk
    mock_yaml.return_value = mock_yaml_load

    generate_data_summary()

    # Check if the correct files were opened
    mock_open.assert_any_call("./subdir1/bulk_conf.yaml")
    mock_open.assert_any_call("./subdir2/bulk_conf.yaml")
    mock_open.assert_any_call("./README_preamble.md")
    mock_open.assert_any_call("./README.md", "w")

    # Check if the correct data was written to README.md
    handle = mock_open()
    handle.write.assert_called_with(
        mock_readme_preamble
        + "\n"
        + "|                      | key1   | key2   |\n|:---------------------|:-------|:-------|\n| [subdir1](./subdir1) | value1 | value2 |\n| [subdir2](./subdir2) | value1 | value2 |"
    )
