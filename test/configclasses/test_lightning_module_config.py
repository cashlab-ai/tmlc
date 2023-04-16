import pytest
from hypothesis import given, strategies as st
from your_module import LightningModuleConfig, ModelConfig, PartialFunctionConfig

class Test_LightningModuleConfig:

    @given(
        model_name=st.text(min_size=1, max_size=20),
        model=st.builds(ModelConfig),
        optimizer=st.builds(PartialFunctionConfig),
        calculate_loss_weights=st.one_of(st.none(), st.builds(PartialFunctionConfig)),
        predict=st.builds(PartialFunctionConfig),
        calculate_best_thresholds=st.builds(PartialFunctionConfig),
        calculate_metrics=st.builds(PartialFunctionConfig),
        pretrain_epochs=st.integers(min_value=0, max_value=100),
    )
    def test_lightning_module_config(
        self,
        model_name,
        model,
        optimizer,
        calculate_loss_weights,
        predict,
        calculate_best_thresholds,
        calculate_metrics,
        pretrain_epochs,
    ):
        config = LightningModuleConfig(
            model_name=model_name,
            model=model,
            optimizer=optimizer,
            calculate_loss_weights=calculate_loss_weights,
            predict=predict,
            calculate_best_thresholds=calculate_best_thresholds,
            calculate_metrics=calculate_metrics,
            pretrain_epochs=pretrain_epochs,
        )

        assert config.model_name == model_name
        assert config.model == model
        assert config.optimizer == optimizer
        assert config.calculate_loss_weights == calculate_loss_weights
        assert config.predict == predict
        assert config.calculate_best_thresholds == calculate_best_thresholds
        assert config.calculate_metrics == calculate_metrics
        assert config.pretrain_epochs == pretrain_epochs

    def test_from_yaml(self, tmpdir):
        config_path = tmpdir.join("config.yaml")

        config_dict = {
            "lightningmodule": {
                "model_name": "my_module",
                "model": {},
                "optimizer": {},
                "predict": {},
                "calculate_best_thresholds": {},
                "calculate_metrics": {},
                "pretrain_epochs": 10,
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = LightningModuleConfig.from_yaml(config_path)
        assert config.model_name == "my_module"
        assert config.pretrain_epochs == 10
