import json
import os
from typing import Any, Dict, List

from jinja2 import Environment
from loguru import logger


def to_json(value):
    """
    Converts a Python object to a JSON string.

    Args:
        value (Any):
            The Python object to convert.

    Returns:
        str:
            The JSON string representation of the object.
    """
    try:
        return json.dumps(value)
    except Exception as e:
        logger.error(e)
        raise e


def render_eda_output(
    figures: Dict[str, str],
    unique_labels: List[str],
    results: Dict[str, Any],
    output_file: str,
) -> None:
    """
    Renders an EDA report template with the provided figures, labels, and metrics.

    Args:
        figures (Dict[str, str]):
            A dictionary of figure names and paths to the corresponding image files.
        unique_labels (List[str]):
            A list of unique labels in the dataset.
        results (Dict[str, Any]):
            A dictionary of EDA results.
        output_file (str):
            The path to the output file to save the rendered report to.

    Returns:
        None
    """
    template_file = os.path.join(os.path.dirname(__file__), "template.md")
    # Load the Markdown template from a file

    with open(template_file) as f:
        template_str = f.read()

    # Create a Jinja2 template object
    env = Environment()
    env.filters["to_json"] = to_json
    template = env.from_string(template_str)

    # Render the template with the necessary variables
    rendered_template = template.render(figures=figures, unique_labels=unique_labels, results=results)

    # Save the rendered Markdown to a file
    with open(output_file, "w") as f:
        f.write(rendered_template)
