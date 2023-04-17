import mlflow.pyfunc
import onnxruntime as rt
import torch

from tmlc.configclasses import TokenizerConfig


class TextMultiLabelClassificationModelWrapperPythonModel(mlflow.pyfunc.PythonModel):
    """
    Wrapper class for multi-label text classification models.

    Args:
        model_path (str): Path to the ONNX model file.
        tokenizer_config (TokenizerConfig): Tokenizer configuration object.
        tokenizer_path (str): Path to the tokenizer model file.
        thresholds (torch.Tensor): Threshold values for class probabilities.

    Example:
        >>> from tmlc.configclasses import TokenizerConfig
        >>> from tmlc.modelwrapper import TextMultiLabelClassificationModelWrapperPythonModel

        >>> # Create an instance of TokenizerConfig.
        >>> tokenizer_config = TokenizerConfig(model_name="bert-base-uncased")

        >>> # Create an instance of TextMultiLabelClassificationModelWrapperPythonModel.
        >>> model = TextMultiLabelClassificationModelWrapperPythonModel(
        ...     model_path="model.onnx",
        ...     tokenizer_config=tokenizer_config,
        ...     tokenizer_path="tokenizer.pt",
        ...     thresholds=torch.tensor([0.5, 0.5, 0.5])
        ... )
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_config: TokenizerConfig,
        tokenizer_path: str,
        thresholds: torch.Tensor,
    ):
        self.model_path = model_path
        self.thresholds = thresholds
        self._set_tokenizer_config(tokenizer_config, tokenizer_path)
        self.model = None
        self.tokenizer = None

    def _set_tokenizer_config(self, tokenizer_config: TokenizerConfig, tokenizer_path: str):
        """
        Sets the tokenizer configuration and path.

        Args:
            tokenizer_config (TokenizerConfig): Tokenizer configuration object.
            tokenizer_path (str): Path to the tokenizer model file.
        """
        self.tokenizer_config = tokenizer_config
        self.tokenizer_config.path = tokenizer_path
        self.tokenizer_config.model_name = None
        self.tokenizer_config.instance = None

    def load_context(self, context):
        """
        Loads the model into memory.

        Args:
            context: MLflow context object.
        """
        self.tokenizer = self.tokenizer_config
        self.model = rt.InferenceSession(self.model_path)

    def predict_logits(self, context, input_data: dict) -> torch.Tensor:
        """
        Predicts class probabilities for the given input text.

        Args:
            context: MLflow context object.
            input_data (dict): Input data dictionary containing "input_text" key with a list of input texts.

        Returns:
            torch.Tensor: Tensor of class probabilities.

        Example:
            >>> context = None
            >>> input_data = {"input_text": ["example text 1", "example text 2"]}
            >>> probabilities = model.predict_logits(context, input_data)
        """
        encoding = self.tokenizer(input_data["input_text"].values.tolist())
        scores = self.model.run(None, encoding)[0]
        return torch.tensor(scores)

    def predict(self, context, input_data: dict, thresholds: torch.Tensor = None) -> torch.Tensor:
        """
        Predicts class labels for the given input text.

        Args:
            context: MLflow context object.
            input_data (dict): Input data dictionary containing "input_text" key with a list of input texts.
            thresholds (torch.Tensor): Threshold values for class probabilities.

        Returns:
            torch.Tensor: Tensor of predicted class labels.

        Example:
            >>> context = None
            >>> thresholds = None
            >>> input_data = {"input_text": ["example text 1", "example text 2"]}
            >>> probabilities = model.predict(context, input_data)
        """
        scores = self.predict_logits(context=context, input_data=input_data)
        if thresholds:
            return scores > thresholds
        else:
            return scores > self.thresholds
