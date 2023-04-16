import pytest
from hypothesis import given, strategies as st
from your_module import MlflowConfig

class TestMlflowConfig:
    @given(
        model_path=st.text(min_size=1, max_size=100),
        tokenizer_path=st.text(min_size=1, max_size=100),
        score_script_path=st.one_of(st.text(min_size=1, max_size=100), st.none()),
        description=st.text(min_size=1, max_size=100),
        tags=st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=20), max_size=5),
        artifact_folder=st.text(min_size=1, max_size=100),
    )
    def test_mlflow_config(self, model_path, tokenizer_path, score_script_path, description, tags, artifact_folder):
        config = MlflowConfig(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            score_script_path=score_script_path,
            description=description,
            tags=tags,
            artifact_folder=artifact_folder,
        )

        assert config.model_path == model_path
        assert config.tokenizer_path == tokenizer_path
        assert config.score_script_path == score_script_path
        assert config.description == description
        assert config.tags == tags
        assert config.artifact_folder == artifact_folder
