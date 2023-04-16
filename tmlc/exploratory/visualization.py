import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


def plot_label_correlations(correlations: pd.DataFrame, label: str) -> plt.Figure:
    """
    Generates a heatmap showing the correlations between variables.

    Args:
        correlations (pd.DataFrame): The DataFrame containing correlation data.
        label (str): The label to be analyzed.

    Returns:
        matplotlib.figure.Figure: The heatmap showing the correlations.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(correlations[label], annot=True, cmap="coolwarm", ax=ax)
    ax.set_title(f"Correlations between Label '{label}' Properties")
    ax.set_xlabel("Properties")
    ax.set_ylabel("Properties")
    return fig


def plot_label_distribution(label_freq: dict) -> plt.Figure:
    """
    Genera te a barplot showing the distribution of labels in the dataset.

    Args:
        label_freq (dict): A dictionary containing the frequency of each label.

    Returns:
        matplotlib.figure.Figure: The barplot showing the label distribution.
    """
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x=list(label_freq.keys()), y=list(label_freq.values()))
    plt.title("Label Distribution")
    plt.xlabel("Labels")
    plt.ylabel("Frequency")
    return fig


def plot_label_co_occurrence_heatmap(co_occurrence_matrix: np.ndarray, unique_labels: list) -> plt.Figure:
    """
    Generate a heatmapshowing theco-occurrence of labels in the dataset.

    Args:
        co_occurrence_matrix (np.ndarray): The matrix containing co-occurrence data.
        unique_labels (list): The list of unique labels.

    Returns:
        matplotlib.figure.Figure: The heatmap showing the label co-occurrence.
    """
    fig = plt.figure(figsize=(12, 6))
    co_occurrence_matrix = co_occurrence_matrix.fillna(0).astype(float)
    sns.heatmap(
        co_occurrence_matrix,
        annot=True,
        cmap="coolwarm",
        xticklabels=unique_labels,
        yticklabels=unique_labels,
    )
    plt.title("Label Co-occurrence Matrix")
    plt.xlabel("Labels")
    plt.ylabel("Labels")
    return fig


def plot_message_length_distribution(data: pd.DataFrame, label: Optional[str] = None) -> plt.Figure:
    """
    Generate a histogram showing the distribution of message lengths in the dataset.

    Args:
        data (pd.DataFrame): The input DataFrame with message length data.
        label (str, optional): The label to be analyzed. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The histogram showing the message length distribution.
    """
    if label:
        data = data[data["labels"].apply(lambda x: label in x)]
        plt_title = f"Message Length Distribution for Label '{label}'"
    else:
        plt_title = "Message Length Distribution"

    fig = plt.figure(figsize=(12, 6))
    sns.histplot(data["message_length"], kde=True, bins=50)
    plt.title(plt_title)
    plt.xlabel("Message Length")
    plt.ylabel("Frequency")
    return fig


def plot_reduced_embeddings(
    reduced_embeddings: np.ndarray, sample_labels: list, function_name: str
) -> plt.Figure:
    """
    Generate a scatter plot showing the reduced BERT embeddings for a given function.

    Args:
        reduced_embeddings (np.ndarray): The reduced BERT embeddings.
        sample_labels (list): The labels for each sample.
        function_name (str): The name of the function being analyzed.

    Returns:
        matplotlib.figure.Figure: The scatterplot showing the reduced BERT embeddings.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    label_colors = {}
    unique_labels = sorted(set([label for labels in sample_labels for label in labels]))

    for i, label in enumerate(unique_labels):
        label_colors[label] = plt.colormaps.get_cmap("tab10")(i)

    for i, (x, y) in enumerate(reduced_embeddings):
        for label in sample_labels[i]:
            ax.scatter(
                x,
                y,
                color=label_colors[label],
                label=label if label not in ax.get_legend_handles_labels()[1] else None,
            )

    ax.set_title(f"Visualizing BERT Embeddings {function_name}")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend()
    return fig


def plot_pos_tag_distribution(pos_tag_freq: dict) -> plt.Figure:
    """
    Generate a barplot showing the distribution of POS tags in the dataset.

    Parameters:
        pos_tag_freq (dict): A dictionary containing the frequency of each POS tag.

    Returns:
        matplotlib.figure.Figure: The barplot showing the POS tag distribution.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(pos_tag_freq.keys()), y=list(pos_tag_freq.values()), ax=ax)
    ax.set_title("POS Tag Distribution")
    ax.set_xlabel("POS Tags")
    ax.set_ylabel("Frequency")
    return fig


def plot_avg_sentence_length_distribution(data: pd.DataFrame, label: Optional[str] = None) -> plt.Figure:
    """
    Generate a histogram showing the distribution of average sentence lengths in the dataset.

    Parameters:
        data (pd.DataFrame): The input DataFrame with average sentence length data.
        label (str, optional): The label to be analyzed. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The histogram showing the average sentence length distribution.
    """
    if label:
        data = data[data["labels"].apply(lambda x: label in x)]
        plt_title = f"Average Sentence Length Distribution for Label '{label}'"
    else:
        plt_title = "Average Sentence Length Distribution"

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data["avg_sent_length"], kde=True, bins=50, ax=ax)
    ax.set_title(plt_title)
    ax.set_xlabel("Average Sentence Length")
    ax.set_ylabel("Frequency")
    return fig


def plot_ngram_distribution(
    corpus: list, label: Optional[str] = None, ngram_range: tuple = (1, 1), top_n: int = 20
) -> plt.Figure:
    """
    Plot the frequency distribution of n-grams for the given corpus.

    Parameters:
        corpus (list): A list of strings representing the corpus.
        label (str): Optional. The label associated with the corpus.
        ngram_range (tuple): Optional. A tuple containing the range of n-grams to consider. Default is (1, 1).
        top_n (int): Optional. The number of top n-grams to include in the plot. Default is 20.

    Returns:
        matplotlib.figure.Figure: The plotted figure.
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(corpus)
    ngram_freq = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0]))
    ngram_freq_sorted = dict(sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True)[:top_n])

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(ngram_freq_sorted.keys()), y=list(ngram_freq_sorted.values()), ax=ax)

    if label:
        plt_title = f"{ngram_range[1]}-Gram Frequency Distribution for Label '{label}'"
    else:
        plt_title = f"{ngram_range[1]}-Gram Frequency Distribution"

    ax.set_title(plt_title)
    ax.set_xlabel("N-grams")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=45)

    return fig


def plot_sentiment_distribution(data: pd.DataFrame, label: Optional[str] = None) -> plt.Figure:
    """
    Plots the distribution of sentiment polarity and subjectivity for the given data.

    Args:
        data (pd.DataFrame): A pandas DataFrame containing the data.
        label (str): Optional. The label associated with the data.

    Returns:
        fig (plt.Figure): The plotted figure.
    """
    if label:
        data = data[data["labels"].apply(lambda x: label in x)]
        plt_title_prefix = f"Sentiment Distribution for Label '{label}' -"
    else:
        plt_title_prefix = "Sentiment Distribution -"

    fig, axs = plt.subplots(nrows=2, figsize=(12, 12))
    sns.histplot(data["sentiment_polarity"], kde=True, bins=50, ax=axs[0])
    axs[0].set_title(plt_title_prefix + " Polarity")
    axs[0].set_xlabel("Sentiment Polarity")
    axs[0].set_ylabel("Frequency")

    sns.histplot(data["sentiment_subjectivity"], kde=True, bins=50, ax=axs[1])
    axs[1].set_title(plt_title_prefix + " Subjectivity")
    axs[1].set_xlabel("Sentiment Subjectivity")
    axs[1].set_ylabel("Frequency")

    plt.tight_layout()

    return fig


def plot_lexical_diversity_distribution(data: pd.DataFrame, label: Optional[str] = None) -> plt.Figure:
    """
    Plots the distribution of lexical diversity for the given data.

    Args:
        data (pd.DataFrame): A pandas DataFrame containing the data.
        label (str): Optional. The label associated with the data.

    Returns:
        fig (Figure): The plotted figure.
    """
    if label:
        data = data[data["labels"].apply(lambda x: label in x)]
        plt_title = f"Lexical Diversity Distribution for Label '{label}'"
    else:
        plt_title = "Lexical Diversity Distribution"

    fig = plt.figure(figsize=(12, 6))
    sns.histplot(data["lexical_diversity"], kde=True, bins=50)
    plt.title(plt_title)
    plt.xlabel("Lexical Diversity")
    plt.ylabel("Frequency")

    return fig


def plot_label_similarity_heatmap(similarity_matrix: np.ndarray, unique_labels: List[str]) -> plt.Figure:
    """
    Plots the heatmap of the similarity matrix for the given labels .

    Args:
        similarity_matrix (np.ndarray): A numpy array containing the similarity matrix.
        unique_labels (List[str]): A list of unique labels.

    Returns:
        fig (plt.Figure): The plotted figure.
    """
    fig = plt.figure(figsize=(12, 6))
    similarity_matrix = similarity_matrix.fillna(0).astype(float)
    sns.heatmap(
        similarity_matrix, annot=True, cmap="coolwarm", xticklabels=unique_labels, yticklabels=unique_labels
    )
    plt.title("Label Similarity Matrix")
    plt.xlabel("Labels")
    plt.ylabel("Labels")
    return fig


def visualize_label_embeddings(reduced_embeddings: dict) -> Dict[str, plt.Figure]:
    """
    Visualizes the reduced label embeddings using PCA and TSNE.

    Args:
        reduced_embeddings (dict): A dictionary containing the reduced embeddings and sample labels.

    Returns:
        figs (Dict[str, plt.Figure]): A dictionary of the plotted figures.
    """
    figs = {}
    for name, values in reduced_embeddings.items():
        fig = plot_reduced_embeddings(values["reduced_embeddings"], values["sample_labels"], name)
        fig_name = f"{name}_Reduced_Label_Embeddings"
        fig_name = re.sub(r"\W+", "_", fig_name)  # replace suboptimal characters with underscore
        figs[fig_name] = fig
    return figs


def visualize_multi_label_data(
    data: pd.DataFrame,
    unique_labels: list,
    label_freq: dict,
    co_occurrence_matrix: np.ndarray,
    correlations: pd.DataFrame,
    similarity_matrix: np.ndarray,
    reduced_embeddings: dict,
) -> Dict[str, plt.Figure]:
    """
    Visualizes the multi-label data.

    Args:
        data (pd.DataFrame): A pandas DataFrame containing the data.
        unique_labels (list): A list of unique labels.
        label_freq (dict): A dictionary containing the frequency of each label.
        co_occurrence_matrix (np.ndarray): A numpy array containing the co-occurrence matrix.
        correlations (pd.DataFrame): A pandas DataFrame containing the correlations between label properties.
        similarity_matrix (np.ndarray): A numpy array containing the similarity matrix.
        reduced_embeddings (dict): A dictionary containing the reduced embeddings and sample labels.

    Returns:
        figs (dict): A dictionary of the plotted figures.

    The function visualizes the multi-label data using various methods such as co-occurrence matrix,
    correlation matrix, similarity matrix, and reduced embeddings. The unique_labels argument
    should contain all the unique labels present in the data, while the label_freq argument
    should contain a dictionary containing the frequency of each label. The co_occurrence_matrix
    argument should contain a numpy array representing the co-occurrence matrix of labels,
    whereas the correlations argument should contain a pandas DataFrame containing the correlations
    between label properties. The similarity_matrix argument should contain a numpy array representing
    the similarity matrix. The reduced_embeddings argument should contain a dictionary containing the
    reduced embeddings and sample labels.
    """
    figs = {}

    # Plot label distribution and co-occurrence heatmap
    fig = plot_label_distribution(label_freq)
    figs["Label_Distribution"] = fig

    fig = plot_label_co_occurrence_heatmap(co_occurrence_matrix, unique_labels)
    figs["Label_Co-Occurrence_Heatmap"] = fig

    # Plot label-specific distributions and analyses
    for label in unique_labels:
        label_data = data[data["labels"].apply(lambda x: label in x)]

        # Plot average sentence length distribution for the label
        fig = plot_avg_sentence_length_distribution(label_data, label=label)
        fig_name = f"{label}_Avg_Sentence_Length_Distribution"
        fig_name = re.sub(r"\W+", "_", fig_name)  # replace suboptimal characters with underscore
        figs[fig_name] = fig

        # Plot n-gram frequency distribution for the label
        for ngram_range in [(1, 1), (2, 2), (3, 3)]:
            plot_ngram_distribution(label_data["message"], label=label, ngram_range=ngram_range)
            fig_name = f"{label}_{ngram_range[1]}-gram_Frequency_Distribution"
            fig_name = re.sub(r"\W+", "_", fig_name)  # replace suboptimal characters with underscore
            figs[fig_name] = fig

        # Plot message length distribution for the label
        fig = plot_message_length_distribution(label_data, label=label)
        fig_name = f"{label}_Message_Length_Distribution"
        fig_name = re.sub(r"\W+", "_", fig_name)  # replace suboptimal characters with underscore
        figs[fig_name] = fig

        # Plot sentiment distribution for the label
        fig = plot_sentiment_distribution(label_data, label=label)
        fig_name = f"{label}_Sentiment_Distribution"
        fig_name = re.sub(r"\W+", "_", fig_name)  # replace suboptimal characters with underscore
        figs[fig_name] = fig

        # Plot lexical diversity distribution for the label
        fig = plot_lexical_diversity_distribution(label_data, label=label)
        fig_name = f"{label}_Lexical_Diversity_Distribution"
        fig_name = re.sub(r"\W+", "_", fig_name)  # replace suboptimal characters with underscore
        figs[fig_name] = fig

        # Plot correlation between label properties for the label
        fig = plot_label_correlations(correlations, label)
        fig_name = f"{label}_Correlation_between_Properties"
        fig_name = re.sub(r"\W+", "_", fig_name)  # replace suboptimal characters with underscore
        figs[fig_name] = fig

    # Plot correlation matrix between labels
    fig = plot_label_similarity_heatmap(similarity_matrix, unique_labels)
    figs["Label_Similarity_Matrix"] = fig

    # Visualize label embeddings using PCA and TSNE
    figs.update(visualize_label_embeddings(reduced_embeddings))

    return figs
