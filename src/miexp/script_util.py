from argparse import ArgumentParser

from pydantic import BaseModel


def parse_args_from_conf[CONF_CLASS: BaseModel](
    configuration_class: type[CONF_CLASS],
) -> CONF_CLASS:
    """Parses command line arguments according to a Pydantic BaseModel and returns an instantiation of that model.

    Args:
        configuration_class (type[BaseModel]): Configuration class
    """
    parser = ArgumentParser()
    for field_name, field_info in configuration_class.model_fields.items():
        if field_info.description is not None:
            description = field_info.description
        elif field_info.annotation is not None:
            description = str(field_info.annotation)
        else:
            description = None
        parser.add_argument(
            f"--{field_name}",
            type=field_info.annotation if field_info.annotation is not None else str,
            help=description,
        )
    args = parser.parse_args()
    return configuration_class(**args.vars())
