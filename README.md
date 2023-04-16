# Text Multi-Label Classification (TMLC)

This repository contains a PyTorch Lightning implementation of a text multi-label classification model using Hugging Face's transformers library for the backbone architecture. The model is designed to perform binary classification for multiple labels, making it suitable for tasks like multi-label text classification, sentiment analysis, or topic identification.

## Table of Contents
- Installation
- Usage
  - Training
  - Prediction
- Dockerization
  - Build the Docker image
  - Run the Docker container
  - CURL Predict endpoint
- Repository Structure

## Installation
To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage
### Training
To train the `TextMultiLabelClassificationModel` using a YAML config file, run:

```bash
python train.py --file-path /path/to/config.yaml
```

If you want to resume training from an existing checkpoint, use the `--check-point` flag:

```bash
python train.py --file-path /path/to/checkpoint.ckpt --check-point True
```

## Prediction
To make predictions with a registered model, run:

```bash
python score.py --model-name model_name --version version_number --texts "text1|text2|text3"
```

`version_number` is optional. If not provided, the latest version of the model will be used.

## Dockerization
You can also run the TMLC model using Docker.

### Build the Docker image:
To build the Docker image, you can run the following command in the same directory as the Dockerfile:

```bash
export IMAGE_NAME=my_mlflow_image
export IMAGE_TAG=1.0.0
docker build -t $IMAGE_NAME:$IMAGE_TAG .
```

### Run the Docker container:
To run the Docker container, you can use the following command:

export MLFLOW_USER=my_username
export MLFLOW_PASSWORD=my_password
export SQL_URL=postgres://user:password@host:port/database

```bash
export MLFLOW_TRACKING_URI="http://host.docker.internal:5050"
docker run -d -p 8100:80 -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI $IMAGE_NAME:$IMAGE_TAG
```

### Execute a curl command to the predict endpoint:
To execute a curl command to the predict endpoint, you can use the following command (assuming the container is running on the same machine):

```bash
curl -X 'POST' \
  'http://0.0.0.0:8090/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{ "model_name": "TMLC", "texts": "example text"}'
```

Replace "your_data_here" with the appropriate input data for your application.

### Build documentation:

Installation:
```bash
poetry install
poetry run pip install -r docs/requirements.txt
```

This command is used to build your MkDocs site. It generates a static HTML version of
your site in a directory called site.

```bash
poetry run mkdocs build
```

To start a local development server that serves your MkDocs site.

```bash
poetry run mkdocs serve
```
