import logging
import os

import click
import matplotlib as mpl
from loguru import logger

from tmlc.exploratory import preprocess_data, visualize_multi_label_data
from tmlc.exploratory.configclasses import EDAClassifiersEvaluationConfig
from tmlc.exploratory.pretraining import train_and_evaluate_classifiers
from tmlc.exploratory.render import render_eda_output
from tmlc.utils import load_yaml_config

logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
mpl.rcParams["figure.max_open_warning"] = 0


def create_eda(config_path):
    """
    Generates an exploratory data analysis (EDA) report for multi-label text data. This function
    reads a YAML configuration file containing the paths to the input CSV file, the name of the
    column containing the text data, the list of column names containing the labels, the path to the
    output Markdown file, and the path to the folder where the output images will be saved. Then, it
    preprocesses the data, visualizes the label distributions and co- occurrences, generates
    embeddings using a pre-trained language model, trains and evaluates multiple classifiers using
    cross-validation, and renders a Markdown report with the results.

    Args:
        config_path (str): Path to the YAML configuration file.

    Example:
        To generate an EDA report for the "toxic_multi_class.csv" dataset with custom options,
        run the following command in the terminal:

        ```
        python tmlc/scripts/eda.py --config_path "eda.yaml"
        ```
    """

    logger.info("Loading configuration from YAML file...")
    config: EDAClassifiersEvaluationConfig = load_yaml_config(config_path, EDAClassifiersEvaluationConfig)

    logger.info("Preprocessing data...")
    data = config.get_data.partial()
    (
        data,
        unique_labels,
        label_freq,
        co_occurrence_matrix,
        correlations,
        similarity_matrix,
        reduced_embeddings,
    ) = preprocess_data(data, config.message_column, config.labels_columns)

    logger.info("Creating data visualizations...")
    figs = visualize_multi_label_data(
        data,
        unique_labels,
        label_freq,
        co_occurrence_matrix,
        correlations,
        similarity_matrix,
        reduced_embeddings,
    )

    figures = {}

    if not os.path.exists("docs/imgs"):
        os.makedirs("docs/imgs")

    # Save each figure to a file with a unique name based on the dictionary key
    for fig_name, fig in figs.items():
        fig_file = f"docs/imgs/{fig_name}.png"
        fig.savefig(fig_file, dpi=300, bbox_inches="tight")
        figures[fig_name] = f"../imgs/{fig_name}.png"

    logger.info("Training and evaluating classifiers...")
    results = train_and_evaluate_classifiers(config, data)

    logger.info("Rendering EDA output...")

    render_eda_output(
        figures=figures, unique_labels=unique_labels, results=results, output_file=config.output_file
    )
    return results


@click.command()
@click.option("--config_path", default="config.yaml", help="Path to the YAML configuration file")
def ccreate_eda(config_path):
    return create_eda(config_path)


if __name__ == "__main__":
    ccreate_eda()
