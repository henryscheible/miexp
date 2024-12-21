from pydantic import BaseModel

from miexp.script_util import parse_args_from_conf


class Configuration(BaseModel):
    arg: list[str]


def main(args: Configuration):
    return args


if __name__ == "__main__":
    args = parse_args_from_conf(Configuration)
