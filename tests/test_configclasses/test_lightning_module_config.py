from hypothesis import given
from hypothesis import strategies as st

from tmlc.configclasses import (
    DataModuleConfig,
    LightningModuleConfig,
    ModelConfig,
    PartialFunctionConfig,
)


class TestLightningModuleConfig:
    @given(
        num_classes=st.integers(min_value=1, max_value=100),
        model=st.builds(ModelConfig),
        data_module_config=st.builds(DataModuleConfig),
        optimizer=st.builds(PartialFunctionConfig),
        define_loss=st.builds(PartialFunctionConfig),
        predict=st.builds(PartialFunctionConfig),
        calculate_best_thresholds=st.builds(PartialFunctionConfig),
    )
    def test_from_yaml(
        self,
        num_classes,
        model,
        data_module_config,
        optimizer,
        define_loss,
        predict,
        calculate_best_thresholds,
    ):
        config_dict = {
            "lightningmodule": {
                "num_classes": num_classes,
                "model": model.dict(),
                "data_module_config": data_module_config.dict(),
                "optimizer": optimizer.dict(),
                "define_loss": define_loss.dict(),
                "predict": predict.dict(),
                "calculate_best_thresholds": calculate_best_thresholds.dict(),
            }
        }
        config = LightningModuleConfig(**config_dict["lightningmodule"])
        assert isinstance(config, LightningModuleConfig)
