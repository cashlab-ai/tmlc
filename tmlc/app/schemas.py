from typing import List, Optional

from pydantic import BaseModel, Field


class LogitsRequest(BaseModel):
    """
    Request model for getting logits from a model.
    """

    model_name: str = Field(..., example="my_model")
    texts: str = Field(..., example="example text")
    model_version: Optional[int] = Field(None, example=1)


class LogitsResponse(BaseModel):
    """
    Response model for getting logits from a model.
    """

    logits: List[float] = Field(..., example=["0.5", "0.7", "0.2"])


class PredictRequest(BaseModel):
    """
    Request model for making predictions using a model.
    """

    model_name: str = Field(..., example="my_model")
    texts: str = Field(..., example="example text")
    model_version: Optional[int] = Field(None, example=1)


class PredictResponse(BaseModel):
    """
    Response model for making predictions using a model.
    """

    predictions: List[str] = Field(..., example=[True, False, True])


class MetricRequest(BaseModel):
    """
    Request model for getting a specific metric for a model run.
    """

    model_name: str = Field(..., example="my_model")
    metric_name: str = Field(..., example="accuracy")
    run_id: Optional[str] = Field(None, example="run_123")
    experiment_id: Optional[str] = Field(None, example="exp_456")


class MetricResponse(BaseModel):
    """
    Response model for getting a specific metric for a model run.
    """

    model_name: str = Field(..., example="my_model")
    metric_name: str = Field(..., example="f1_score")
    metric_value: float = Field(..., example=0.92)
    run_id: str = Field(..., example="run_123")


class MetricsResponse(BaseModel):
    """
    Response model for getting all metrics for a model run.
    """

    model_name: str = Field(..., example="my_model")
    metrics: dict = Field(..., example={"accuracy": 0.95, "f1_score": 0.92})
    run_id: str = Field(..., example="run_123")


class ModelLocation(BaseModel):
    """
    Request model for getting the location of a model.
    """

    model_name: str = Field(..., example="my_model")
    run_id: str = Field(..., example="run_123")
    experiment_id: str = Field(..., example="exp_456")
