import os

import pandas as pd
import yaml
from tabulate import tabulate


def generate_data_summary() -> None:
    """Generates a summary of data from YAML configuration files and writes it to a README file.

    This function walks through the current directory and its subdirectories to find YAML files named
    "bulk_conf.yaml". It parses these files and creates a summary dictionary where the keys are the
    subdirectory names and the values are the parsed YAML content. This summary is then converted to
    a pandas DataFrame and subsequently to a markdown string using the tabulate library. The resulting
    markdown string is appended to the content of a pre-existing "README_preamble.md" file and written
    to a new "README.md" file.

    Raises:
        FileNotFoundError: If "bulk_conf.yaml" or "README_preamble.md" is not found in the expected locations.
        yaml.YAMLError: If there is an error parsing the YAML files.
        pd.errors.EmptyDataError: If the DataFrame creation fails due to empty data.
    """
    data_summary = {}
    current_directory = os.getcwd()

    for subdir, _, files in os.walk(current_directory):
        if "bulk_conf.yaml" in files and subdir != ".":
            subdir_name = os.path.basename(subdir)
            yaml_path = os.path.join(subdir, "bulk_conf.yaml")
            with open(yaml_path) as yaml_file:
                parsed_yaml = yaml.safe_load(yaml_file)
                data_summary[f"[{subdir_name}](./{subdir_name})"] = parsed_yaml

    # Convert the dictionary to a pandas DataFrame
    data_summary_df = pd.DataFrame.from_dict(data_summary, orient="index")
    data_summary_df = data_summary_df.sort_index()

    # Convert the DataFrame to a markdown string using tabulate
    markdown_string = tabulate(data_summary_df, headers="keys", tablefmt="pipe")  # type: ignore

    with open("./README_preamble.md") as f:
        preamble = f.read()

    with open("./README.md", "w") as f:
        f.write(preamble + "\n" + markdown_string)
