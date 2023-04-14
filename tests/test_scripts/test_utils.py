import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import mlflow
import pandas as pd
import pytest
import pytorch_lightning as pl

from tmlc import (
    add_suffix_to_filename,
    create_artifacts,
    create_input_output_examples,
    export_model_to_onnx,
    json_to_nested_tags,
    load_model_data_trainer_config,
    load_yaml_config,
    prepare_model_path,
    register_model,
    save_and_registry_log_pyfunc_model,
    set_mlflow_tags,
    setup_trainer,
)
from tmlc.configclasses import TrainerConfig
from tmlc.dataclasses import DataModule
from tmlc.model import TextMultiLabelClassificationModel

TEST_CONFIG_PATH = "path/to/test_config.yaml"
TEMP_DIR = "temp"


@pytest.fixture(scope="session", autouse=True)
def setup_teardown_temp_directory():
    os.makedirs(TEMP_DIR, exist_ok=True)
    yield
    shutil.rmtree(TEMP_DIR)


@pytest.fixture
def trainer_config():
    return TrainerConfig()


@pytest.fixture
def model():
    return TextMultiLabelClassificationModel()


@pytest.fixture
def data_module():
    return DataModule()


@pytest.fixture
def config_train(tmp_path):
    config = TrainerConfig(
        lightning_module_config=MagicMock(model_name="test_model"),
        data_module_config=MagicMock(dataset=MagicMock(tokenizer=MagicMock())),
        mlflow_config=MagicMock(
            model_path=str(tmp_path / "model.onnx"), tokenizer_path=str(tmp_path / "tokenizer")
        ),
    )
    return config


@pytest.fixture
def model_train():
    model = MagicMock(spec=TextMultiLabelClassificationModel)
    model.thresholds = [0.5]
    return model


class TestUtilities:
    def test_set_mlflow_tags(self, trainer_config):
        # This function is hard to test since it is calling a library function to set tags
        # But we can test if it raises any exceptions while running
        try:
            set_mlflow_tags(trainer_config)
        except Exception as e:
            pytest.fail(f"set_mlflow_tags raised an exception: {e}")

    def test_export_model_to_onnx(self, model, trainer_config):
        try:
            export_model_to_onnx(model, trainer_config)
        except Exception as e:
            pytest.fail(f"export_model_to_onnx raised an exception: {e}")

    # ... Add more tests for each utility function ...


class TestModelTraining:
    @pytest.fixture(scope="class")
    def config(self):
        return load_yaml_config(config_path=TEST_CONFIG_PATH, basemodel=TrainerConfig)

    def test_load_model_data_trainer_config(self, config):
        model, datamodule, _ = load_model_data_trainer_config(TEST_CONFIG_PATH)
        assert isinstance(model, TextMultiLabelClassificationModel)
        assert isinstance(datamodule, DataModule)
        assert config == _

    def test_setup_trainer(self, config):
        trainer = setup_trainer(config)
        assert isinstance(trainer, pl.Trainer)

    def test_training(self, config):
        model, datamodule, _ = load_model_data_trainer_config(TEST_CONFIG_PATH)
        trainer = setup_trainer(config)
        trainer.fit(model, datamodule)
        assert model.training_finished


class TestTrainingUtils:
    def test_set_mlflow_tags(self, config_train):
        set_mlflow_tags(config_train)
        assert mlflow.get_tags() == json_to_nested_tags(config_train.dict())

    def test_export_model_to_onnx(self, model_train, config_train):
        export_model_to_onnx(model_train, config_train)
        assert os.path.exists(config_train.mlflow_config.model_path)

    def test_create_artifacts(self, config_train):
        artifacts = create_artifacts(config_train)
        expected_artifacts = {
            "model": config_train.mlflow_config.model_path,
            "tokenizer": config_train.mlflow_config.tokenizer_path,
        }
        assert artifacts == expected_artifacts

    def test_prepare_model_path(self, config_train):
        model_path = prepare_model_path(config_train)
        assert model_path == f"model/{config_train.lightning_module_config.model_name}"

    def test_create_input_output_examples(self):
        input_example, output_example = create_input_output_examples()
        assert isinstance(input_example, pd.DataFrame)
        assert isinstance(output_example, pd.DataFrame)

    def test_save_and_registry_log_pyfunc_model(self, model_train, config_train):
        model_uri = save_and_registry_log_pyfunc_model(model_train, config_train)
        assert model_uri.startswith("runs:/")

    def test_register_model(self, model_train, config_train):
        model_uri = register_model(model_train, config_train)
        assert model_uri.startswith("runs:/")

    def test_json_to_nested_tags(self):
        data = {
            "outer": {
                "inner": {"key1": "value1", "key2": "value2"},
                "key3": "value3",
            },
            "key4": "value4",
        }
        expected_tags = {
            "outer.inner.key1": "value1",
            "outer.inner.key2": "value2",
            "outer.key3": "value3",
            "key4": "value4",
        }
        tags = json_to_nested_tags(data)
        assert tags == expected_tags

    def test_add_suffix_to_filename(self):
        filename = "test.txt"
        suffix = "_new"
        new_filename = add_suffix_to_filename(filename, suffix)
        expected_filename = Path("test_new.txt")
        assert new_filename == expected_filename

    def test_load_yaml_config(self, config_train, tmp_path):
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            f.write(config_train.to_yaml())
        loaded_config = load_yaml_config(config_path, TrainerConfig)
        assert isinstance(loaded_config, TrainerConfig)

    def test_load_model_data_trainer_config(self, config_train, tmp_path):
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            f.write(config_train.to_yaml())
        model_config, data_config, trainer_config = load_model_data_trainer_config(config_path, TrainerConfig)
        assert isinstance(model_config, dict)
        assert isinstance(data_config, dict)
        assert isinstance(trainer_config, dict)

    def test_setup_trainer(self, config_train):
        trainer = setup_trainer(config_train)
        assert isinstance(trainer, pl.Trainer)
        assert trainer.max_epochs == config_train.trainer_config.max_epochs
        assert trainer.gpus == config_train.trainer_config.gpus

    def test_data_module(self):
        data_module = DataModule("test_dataset")
        assert isinstance(data_module, DataModule)
        assert data_module.tokenizer is not None
