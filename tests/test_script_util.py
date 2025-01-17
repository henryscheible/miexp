import sys

import pytest
from pydantic import BaseModel

from miexp.script_util import parse_args_from_conf


class TestConfig(BaseModel):
    name: str
    age: int


def test_parse_args_from_conf():
    test_args = ["--name", "John", "--age", "30"]
    sys.argv = [sys.argv[0]] + test_args
    config = parse_args_from_conf(TestConfig)
    assert config.name == "John"
    assert config.age == 30


def test_parse_args_from_conf_invalid_arg():
    test_args = ["--name", "John", "--age", "thirty"]
    sys.argv = [sys.argv[0]] + test_args
    with pytest.raises(SystemExit):
        parse_args_from_conf(TestConfig)
