from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from tmlc.exploratory.configclasses import (
    EDAClassifiersEvaluationConfig,
    SklearnClassifiersConfig,
    TransformerModelConfig,
)


def get_input_and_attention_masks(
    X: pd.Series, tokenizer: Any
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Get the input IDs and attention masks for a given Pandas Series of text data and tokenizer.

    Args:
        X (pd.Series): A Pandas Series containing text data.
        tokenizer (Any): A tokenizer object for a specific transformer model.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: A tuple containing two lists of
            torch tensors - input IDs and attention masks.
    """
    input_ids, attention_masks = [], []
    for text in X:
        encoded = tokenizer(text)
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])

    return input_ids, attention_masks


def get_transformer_embeddings(transformer_model: TransformerModelConfig, X: pd.Series) -> np.ndarray:
    """
    Get the transformer embeddings for a given transformer model and Pandas Series of text data.

    Args:
        transformer_model (TransformerModelConfig): A TransformerModelConfig object
            containing the transformer model and tokenizer.
        X (pd.Series): A Pandas Series containing text data.

    Returns:
        np.ndarray: A numpy array containing the transformer embeddings for the input
            text data.
    """
    input_ids, attention_masks = get_input_and_attention_masks(X, transformer_model.tokenizer)

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    with torch.no_grad():
        embeddings: torch.Tensor = transformer_model.pretrainedmodel.model(input_ids, attention_masks)[0]

    embeddings_flat = embeddings.view(embeddings.shape[0], -1)
    return embeddings_flat.numpy()


def transform_data(transformer_model: Any, *data: np.ndarray) -> List[np.ndarray]:
    """
    Transform multiple sets of text data into transformer embeddings.

    Args:
        transformer_model (Any): A transformer model object.
        *data (np.ndarray): One or more numpy arrays containing text data.

    Returns:
        List[np.ndarray]: A list of numpy arrays containing transformer embeddings
            for the input text data.
    """
    return [get_transformer_embeddings(transformer_model, d) for d in data]


def find_best_classifier(
    clf: Any, X_train: np.ndarray, y_train: np.ndarray, hyperparams: Dict[str, List]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Find the best classifier and its hyperparameters using GridSearchCV.

    Args:
        clf (Any): A classifier object.
        X_train (np.ndarray): A numpy array containing the training input data.
        y_train (np.ndarray): A numpy array containing the training output data.
        hyperparams (Dict[str, List]): A dictionary containing the hyperparameters
            for the classifier.

    Returns:
        Tuple[Any, Dict[str, Any]]: A tuple containing the best classifier and the
            best hyperparameters.
    """
    grid_search = GridSearchCV(clf, hyperparams, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_predictions(clf, X_val_transformed, y_val, X_test_transformed, y_test) -> Dict[str, float]:
    """
    Evaluate the predictions of a classifier on the validation and test datasets.

    Args:
        clf (Any): A classifier object.
        X_val_transformed (np.ndarray): The transformed validation input data.
        y_val (np.ndarray): The validation output data.
        X_test_transformed (np.ndarray): The transformed test input data.
        y_test (np.ndarray): The test output data.

    Returns:
        Dict[str, float]: A dictionary containing the F1 scores for the validation and
            test predictions.
    """
    val_predictions = clf.predict(X_val_transformed)
    test_predictions = clf.predict(X_test_transformed)
    return {
        "val_f1": f1_score(y_val, val_predictions),
        "test_f1": f1_score(y_test, test_predictions),
    }


def train_and_evaluate_classifier(
    clf_config: SklearnClassifiersConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train and evaluate a classifier on the given data.

    Args:
        clf_config (SklearnClassifiersConfig): A SklearnClassifiersConfig object
            containing the classifier and its hyperparameters.
        X_train (np.ndarray): The training input data.
        y_train (np.ndarray): The training output data.
        X_val (np.ndarray): The validation input data.
        y_val (np.ndarray): The validation output data.
        X_test (np.ndarray): The test input data.
        y_test (np.ndarray): The test output data.

    Returns:
        Tuple[Any, Dict[str, float]]: A tuple containing the best classifier and
            a dictionary of evaluation metrics.
    """
    clf = clf_config.clf.partial()
    best_estimator, best_params = find_best_classifier(clf, X_train, y_train, clf_config.hyperparams)
    best_estimator.set_params(**best_params)
    best_estimator.fit(X_train, y_train)
    metrics = evaluate_predictions(best_estimator, X_val, y_val, X_test, y_test)
    return best_estimator, metrics


def train_and_evaluate(
    transformer_model: TransformerModelConfig,
    classifiers: List[SklearnClassifiersConfig],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Train and evaluate multiple classifiers on the given data using a specific transformer model.

    Args:
        transformer_model (TransformerModelConfig): A TransformerModelConfig object
            containing the transformer model and tokenizer.
        classifiers (List[SklearnClassifiersConfig]): A list of SklearnClassifiersConfig
            objects containing classifiers and their hyperparameters.
        X_train (np.ndarray): The training input data.
        y_train (np.ndarray): The training output data.
        X_val (np.ndarray): The validation input data.
        y_val (np.ndarray): The validation output data.
        X_test (np.ndarray): The test input data.
        y_test (np.ndarray): The test output data.

    Returns:
        Dict[str, Any]: A dictionary containing the evaluation results for each classifier.
    """
    transformer_embeddings = transform_data(transformer_model, X_train, X_val, X_test)
    results = {}
    for clf_config in classifiers:
        best_estimator, metrics = train_and_evaluate_classifier(
            clf_config,
            transformer_embeddings[0],
            y_train,
            transformer_embeddings[1],
            y_val,
            transformer_embeddings[2],
            y_test,
        )
        results[clf_config.clf.func] = {"hyperparams": best_estimator.get_params(), "metrics": metrics}

    return results


def train_and_evaluate_classifiers(
    config: EDAClassifiersEvaluationConfig, data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Train and evaluate classifiers using a specific configuration and data.

    Args:
        config (EDAClassifiersEvaluationConfig):
            A EDAClassifiersEvaluationConfig object containing configuration settings.
        data (pd.DataFrame): A DataFrame containing the input data.

    Returns:
        Dict[str, Any]: A dictionary containing the evaluation results for each
            classifier and transformer model.
    """
    logger.info("Loading configuration and data")
    results = {}
    data = config.get_data.partial()
    train_df, val_df, test_df = config.split_data.partial(data)
    X_train = train_df[config.message_column]
    X_val = val_df[config.message_column]
    X_test = test_df[config.message_column]

    for transformer in config.transformer_models:
        for label in config.labels_columns:
            if label not in results.keys():
                results[label] = {}
            logger.info(
                f"Training and evaluating classifiers for label: {label}, \
                    transformer model: {transformer.pretrainedmodel.path}"
            )
            model_results = train_and_evaluate(
                transformer,
                config.classifiers,
                X_train,
                train_df[label],
                X_val,
                val_df[label],
                X_test,
                test_df[label],
            )
            results[label].update({transformer.pretrainedmodel.path: model_results})
            logger.info(model_results)

    logger.info("Evaluation completed successfully!")
    return results
