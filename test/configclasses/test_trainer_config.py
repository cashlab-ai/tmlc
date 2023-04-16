import pytest
import yaml
from hypothesis import given, strategies as st
from your_module import TrainerConfig, LightningModuleConfig, DataModuleConfig, MlflowConfig, PartialFunctionConfig

class Test_TrainerConfig:

    @given(
        lightning_module_config=st.builds(LightningModuleConfig),
        data_module_config=st.builds(DataModuleConfig),
        mlflow_config=st.builds(MlflowConfig),
        callbacks=st.lists(st.builds(PartialFunctionConfig)),
        loggers=st.lists(st.builds(PartialFunctionConfig)),
        kwargs=st.fixed_dictionaries({"key": st.text(min_size=1, max_size=20)}),
        seed=st.integers(min_value=0, max_value=2 ** 32 - 1),
        config_path=st.none(),
    )
    def test_trainer_config(
        self,
        lightning_module_config,
        data_module_config,
        mlflow_config,
        callbacks,
        loggers,
        kwargs,
        seed,
        config_path,
    ):
        config = TrainerConfig(
            lightning_module_config=lightning_module_config,
            data_module_config=data_module_config,
            mlflow_config=mlflow_config,
            callbacks=callbacks,
            loggers=loggers,
            kwargs=kwargs,
            seed=seed,
            config_path=config_path,
        )

        assert config.lightning_module_config == lightning_module_config
        assert config.data_module_config == data_module_config
        assert config.mlflow_config == mlflow_config
        assert config.callbacks == callbacks
        assert config.loggers == loggers
        assert config.kwargs == kwargs
        assert config.seed == seed
        assert config.config_path == config_path

    def test_from_yaml(self, tmpdir):
        config_path = tmpdir.join("config.yaml")

        config_dict = {
            "trainer_config": {
                "lightning_module_config": {},
                "data_module_config": {},
                "mlflow_config": {
                    "model_path": "model",
                    "tokenizer_path": "tokenizer",
                    "description": "description",
                    "tags": {"tag_key": "tag_value"},
                    "artifact_folder": "artifacts",
                },
                "callbacks": [],
                "loggers": [],
                "kwargs": {"key": "value"},
                "seed": 42,
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = TrainerConfig.from_yaml(config_path)
        assert config.kwargs == {"key": "value"}
        assert config.seed == 42
        assert config.config_path == config_path
