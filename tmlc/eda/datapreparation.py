from typing import List

import numpy as np
import pandas as pd
import spacy
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from textblob import TextBlob
from transformers import AutoModel, AutoTokenizer


def prepare_dataframe(data, message_col: str, label_cols: list) -> pd.DataFrame:
    for col in label_cols:
        data[f"not_{col}"] = data[col].apply(lambda x: 1 if x == 0 else 0)
    not_label_cols = [f"not_{col}" for col in label_cols]
    all_label_cols = label_cols + not_label_cols
    data["labels"] = data[all_label_cols].apply(
        lambda x: [col for col, val in zip(all_label_cols, x) if val == 1], axis=1
    )
    data = data.rename(columns={message_col: "message"})
    return data[["message", "labels"]]


def calculate_label_frequency(data: pd.DataFrame) -> dict:
    return data.explode("labels")["labels"].value_counts().to_dict()


def analyze_linguistic_complexity(data: pd.DataFrame) -> pd.DataFrame:
    nlp = spacy.load("en_core_web_sm")
    pos_tag_freq_list = []
    sent_counts = [len(list(nlp(message).sents)) for message in data["message"]]
    data["sent_count"] = sent_counts
    data["avg_sent_length"] = data["message_length"] / data["sent_count"]
    for doc in nlp.pipe(data["message"]):
        pos_tag_freq = {}
        for token in doc:
            pos_tag_freq[token.pos_] = pos_tag_freq.get(token.pos_, 0) + 1
        pos_tag_freq_list.append(pos_tag_freq)

    data["pos_tag_freq"] = pos_tag_freq_list
    return data


# Feature extraction functions
def calculate_message_length(data: pd.DataFrame) -> pd.DataFrame:
    data["message_length"] = data["message"].str.split().str.len()
    return data


def analyze_sentiment(data: pd.DataFrame) -> pd.DataFrame:
    sentiments = data["message"].apply(lambda x: TextBlob(x).sentiment)
    data["sentiment_polarity"] = sentiments.apply(lambda x: x.polarity)
    data["sentiment_subjectivity"] = sentiments.apply(lambda x: x.subjectivity)
    return data


def analyze_lexical_diversity(data: pd.DataFrame) -> pd.DataFrame:
    data["lexical_diversity"] = data["message"].apply(lambda text: len(set(text.split())) / len(text.split()))
    return data


def get_bert_embeddings(messages, model, tokenizer):
    inputs = tokenizer(messages.tolist(), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings


def get_samples(data, sample_size=500):
    sample_data = data.sample(sample_size, replace=False)
    sample_indices = sample_data.index
    sample_labels = sample_data["labels"].values
    sample_messages = sample_data["message"].values
    return sample_indices, sample_labels, sample_messages


# TODO CORRECT THIS FUNCTION
def calculate_co_occurrence_matrix(data: pd.DataFrame, unique_labels: List[str]) -> pd.DataFrame:
    """
    Calculate the co-occurrence matrix of labels in a pandas DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame with a 'labels' column containing lists of labels.
        unique_labels (List[str]): A list of unique labels.

    Returns:
        pd.DataFrame: The co-occurrence matrix of size (label_count, label_count).
    """
    co_occurrence_matrix = pd.DataFrame(index=unique_labels, columns=unique_labels)

    for labels in data["labels"]:
        for label1 in labels:
            if label1 not in unique_labels:
                continue
            for label2 in labels:
                if label2 not in unique_labels:
                    continue
                co_occurrence_matrix.loc[label1, label2] += 1

    return co_occurrence_matrix


def calculate_label_correlations(data, unique_labels, label_properties):
    correlations = {}
    for label in unique_labels:
        label_data = data[data["labels"].apply(lambda x: label in x)]
        try:
            correlations[label] = label_data[label_properties].corr()
        except ValueError:
            correlations[label] = pd.DataFrame(index=label_properties, columns=label_properties)
    return correlations


def calculate_label_similarity(data, unique_labels, label_properties):
    similarity_matrix = pd.DataFrame(index=unique_labels, columns=unique_labels)

    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            if i == j:
                similarity_matrix.loc[label1, label2] = 1
            else:
                label1_data = data[data["labels"].apply(lambda x: label1 in x)][label_properties].values
                label2_data = data[data["labels"].apply(lambda x: label2 in x)][label_properties].values
                try:
                    similarity = np.corrcoef(label1_data, label2_data)[0, 1]
                except ValueError:
                    similarity = np.nan
                similarity_matrix.loc[label1, label2] = similarity

    return similarity_matrix


def calculate_label_embeddings(data, model, tokenizer, reducers=None):
    _, sample_labels, sample_messages = get_samples(data, sample_size=len(data))
    embeddings = get_bert_embeddings(sample_messages, model, tokenizer)
    reduced_embeddings = {}
    reducers = reducers or {"PCA": PCA, "TSNE": TSNE}
    for name, reducer in reducers.items():
        _reduced_embeddings = reducer(n_components=2).fit_transform(embeddings)
        reduced_embeddings[name] = {"reduced_embeddings": _reduced_embeddings, "sample_labels": sample_labels}
    return reduced_embeddings


def prepare_data_for_analysis(data: pd.DataFrame) -> pd.DataFrame:
    # Analyze message lengths, sentiment, and lexical diversity
    data = calculate_message_length(data)
    data = analyze_sentiment(data)
    data = analyze_lexical_diversity(data)

    # Analyze linguistic complexity
    data = analyze_linguistic_complexity(data)

    return data


def preprocess_data(data, message_col, label_cols):
    # Load data
    # Load and preprocess the dataset
    data = prepare_dataframe(data=data, message_col=message_col, label_cols=label_cols)

    # Prepare data for analysis
    data = prepare_data_for_analysis(data)

    # Calculate label frequency and co-occurrence
    label_freq = calculate_label_frequency(data)
    unique_labels = list(label_freq.keys())
    co_occurrence_matrix = calculate_co_occurrence_matrix(data, unique_labels)

    # Calculate correlation matrix between label properties
    correlations = calculate_label_correlations(
        data=data,
        unique_labels=unique_labels,
        label_properties=[
            "message_length",
            "sentiment_polarity",
            "sentiment_subjectivity",
            "lexical_diversity",
        ],
    )

    # Calculate label similarity matrix
    similarity_matrix = calculate_label_similarity(
        data=data,
        unique_labels=unique_labels,
        label_properties=[
            "message_length",
            "sentiment_polarity",
            "sentiment_subjectivity",
            "lexical_diversity",
        ],
    )

    # Load the pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased")

    # Calculate label embeddings using BERT
    reduced_embeddings = calculate_label_embeddings(data, model, tokenizer)
    return (
        data,
        unique_labels,
        label_freq,
        co_occurrence_matrix,
        correlations,
        similarity_matrix,
        reduced_embeddings,
    )
