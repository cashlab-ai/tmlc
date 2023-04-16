from typing import List, Optional

import mlflow
from fastapi import FastAPI, HTTPException

from .schemas import (
    LogitsRequest,
    LogitsResponse,
    MetricRequest,
    MetricResponse,
    MetricsResponse,
    ModelLocation,
    PredictRequest,
    PredictResponse,
)

app = FastAPI()

model_cache = {}
model_cache_version = {}


@app.get("/mlflow_setup")
def mlflow_setup():
    # Get the MLflow tracking URI
    tracking_uri = mlflow.get_tracking_uri()
    # Get the MLflow registry URI
    registry_uri = mlflow.get_registry_uri()
    # Return the response data
    return dict(tracking_uri=tracking_uri, registry_uri=registry_uri)


@app.get("/metric", response_model=MetricResponse)
def get_metric(request: MetricRequest):
    """
    Returns the value of a specified metric for a specified model run.

    Args:
        request (MetricRequest): An instance of the MetricRequest model
            containing the request parameters.

    Returns:
        MetricResponse: An instance of the MetricResponse model containing
            the response data.

    Raises:
        HTTPException: If the model or metric is not found, or there is an error
            with retrieving the metric value.
    """
    # Load the run
    run = _get_run(request.model_name, request.run_id, request.experiment_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Get the metric value
    try:
        metric = run.data.metrics[request.metric_name]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Metric '{request.metric_name}' not found")

    # Return the response data
    return MetricResponse(
        model_name=request.model_name,
        metric_name=request.metric_name,
        metric_value=metric,
        run_id=run.info.run_id,
    )


def _get_model(model_name, model_version):
    # Load the registered model by name if not already in cache
    if (model_name, model_version) not in model_cache:
        try:
            model = mlflow.pytorch.load_model(model_name=model_name, version=model_version)
            model_cache_version[model_name] = model._metadata.version
            model_cache[model_name] = model
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to load the model") from e
    else:
        model = model_cache[model_name]
        model_version = model_cache_version[model_name]
    app.logger.info(f"Received request: model_name={model_name}, version={model_version}")
    return model


@app.post("/logits", response_model=LogitsResponse, responses={500: {"description": "Failed to make logits"}})
def logits(request: LogitsRequest) -> List[str]:
    """
    Predicts the probabilities of the input texts for each class label using the specified model.

    Args:
        request (LogitsRequest): The request body containing the model name,
            input texts, and optional model version.

    Returns:
        LogitsResponse: The predicted logits for each input text as a list of strings.

    Raises:
        HTTPException: If the input parameters are invalid or there is an error
            with loading the model or making predictions.
    """

    model = _get_model(request.model_name, request.model_version)

    # Split the input texts by the '|' character
    messages = request.texts.split("|")

    # Make logits
    try:
        logits = model.predict_logits(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to make logits") from e

    # Log the request and response
    app.logger.info(f"Returned response: logits={logits}")
    # Return predictions as JSON response
    return LogitsResponse(logits=logits)


@app.post(
    "/predict", response_model=PredictResponse, responses={500: {"description": "Failed to make predictions"}}
)
def predict(request: PredictRequest) -> List[str]:
    """
    Predicts the class labels of the input texts using the specified model.

    Args:
        request (PredictRequest): The request body containing the model name,
            input texts, and optional model version.

    Returns:
        PredictResponse: The predicted class labels for each input text as a list of strings.

    Raises:
        HTTPException: If the input parameters are invalid or there is an error with loading
            the model or making predictions.
    """

    model = _get_model(request.model_name, request.model_version)

    # Split the input texts by the '|' character
    messages = request.texts.split("|")

    # Make predictions
    try:
        predictions = model.predict(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to make predictions") from e

    # Log the request and response
    app.logger.info(f"Returned response: predictions={predictions}")
    # Return predictions as JSON response
    return predictions


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics(request: ModelLocation):
    """
    Returns the value of the specified metric for the specified model run.

    Args:
        model_name (str): The name of the registered model to retrieve metric for.
        run_id (str, optional): The ID of the run to retrieve metric from. If not provided,
            searches for the latest run for the model.
        experiment_id (str, optional): The ID of the experiment to search for the run in.
            If not provided, searches across all experiments.

    Returns:
        MetricsResponse: A response object containing the model name,
            metrics dictionary, and run ID.

    Raises:
        HTTPException: If the model or metric is not found,
            or there is an error with retrieving the metric value.
    """
    # Load the run
    run = _get_run(request.model_name, request.run_id, request.experiment_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return MetricsResponse(model_name=request.model_name, metrics=run.data.metrics, run_id=run.info.run_id)


@app.get("/health")
def health():
    """
    Checks if MLflow is available and returns a status response.

    Returns:
        dict: A dictionary containing the status response.

    Raises:
        HTTPException: If MLflow is not available.
    """
    # Check if MLflow is available
    try:
        mlflow.get_tracking_uri()
    except Exception as e:
        app.logger.error(e)
        raise HTTPException(status_code=500, detail="MLflow not available")

    return {"status": "ok"}


@app.get("/model/{model_name}", response_model=ModelLocation)
def get_model_location(model_name: str):
    """
    Returns the location information of the specified model.

    Args:
        model_name (str): The name of the registered model to retrieve location for.

    Returns:
        ModelLocation: A Pydantic model containing the model name, run ID,
            and experiment ID.

    Raises:
        HTTPException: If the model is not found.
    """
    # Load the model
    model = _get_model(model_name)
    # Get the run and experiment information
    experiment_id = model._metadata.experiment_id
    run_id = model._metadata.run_id

    return ModelLocation(model_name=model_name, run_id=run_id, experiment_id=experiment_id)


def _get_run(model_name: str, run_id: Optional[str], experiment_id: Optional[str]):
    """
    Retrieves a Run object from the Azure Machine Learning service based on the provided parameters.

    Args:
        model_name (str): The name of the registered model to retrieve the run for.
        run_id (Optional[str]): The ID of the specific run to retrieve. If not provided,
            the latest run will be retrieved.
        experiment_id (Optional[str]): The ID of the experiment the run belongs to.
            If not provided, the current experiment will be used.

    Returns:
        Optional[Run]: The Run object representing the specified run, or None if no run was found.
    """
    # If run ID is provided, load the run by ID
    if run_id:
        try:
            run = mlflow.get_run(run_id)
            if run.data.tags["mlflow.project"] != model_name:
                raise ValueError(f"Run {run_id} is not associated with model {model_name}")
            return run
        except Exception as e:
            app.logger.error(e)
            return None

    # If experiment ID is provided, search for the latest run with the matching tag
    if experiment_id:
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id], filter_string=f"tags.mlflow.project='{model_name}'"
        )
        if len(runs) == 0:
            return None
        return mlflow.get_run(runs[0].info.run_id)

    # If neither run ID nor experiment ID is provided, search for the latest run with the matching tag
    runs = mlflow.search_runs(filter_string=f"tags.mlflow.project='{model_name}'")
    if len(runs) == 0:
        return None
    return mlflow.get_run(runs[0].info.run_id)
