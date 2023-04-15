from typing import Optional, Tuple, Dict, Union, List, Any
import numpy as np
import torch
from loguru import logger
import pytorch_lightning as pl
from transformers import AutoModel
from tmlc.configclasses import LightningModuleConfig

class TextMultiLabelClassificationModel(pl.LightningModule):
    """A PyTorch Lightning module for text multi-label classification.

    Args:
        config: An instance of LightningModuleConfig with the necessary hyperparameters.

    Attributes:
        backbone (AutoModel): A pretrained transformer model
            from the transformers library.
        classifier (torch.nn.Linear): A linear layer for classification.
        config (LightningModuleConfig): An instance of LightningModuleConfig
            with the necessary hyperparameters.
        loss (Callable): A loss function for training the model.
        thresholds (torch.Tensor): A tensor containing the best classification
            thresholds for each label.

    Methods:
        _epoch_end(element):
            Computes the loss and logits for an epoch and logs them to
            the appropriate logger.
        _step(batch, batch_idx, element):
            Processes one batch of data during training or validation.
        configure_optimizers():
            Configures the optimizer used for training.
        forward(data: Dict[str, torch.Tensor]) -> torch.Tensor:
            Defines the forward pass of the module.

        get_best_thresholds(probabilities, labels):
            Computes the best thresholds for converting the predicted logits
            into binary class predictions.
        load(path, map_location):
            Loads the state of the module from a checkpoint file.
        on_test_epoch_end():
            Processes outputs after each epoch of testing and logs the evaluation results.
        on_train_epoch_end():
            Called at the end of the training epoch to perform any necessary operations
            on the epoch outputs.
        on_validation_epoch_end():
            Performs end-of-epoch validation operations.
        predict(data: Dict[str, torch.Tensor]) -> torch.Tensor:
            Generates binary predictions for the input data.
        predict_logits(data: Dict[str, torch.Tensor]) -> torch.Tensor:
            Generates predictions in the form of logits.
        test_step(batch, batch_idx):
            Processes one batch of test data.
        training_step(batch, batch_idx):
            Processes one batch of training data.
        validation_step(batch, batch_idx):
            Processes one batch of validation data.

    Examples:
        Here's an example of how to use the `TextMultiLabelClassificationModel` class:

        ```
        from tmlc.configclasses import LightningModuleConfig, TrainConfig
        from tmlc.model import TextMultiLabelClassificationModel
        from tmlc.utils import load_yaml
        from tmlc.data import EmailDataLoader, EmailDataModule

        # Load the configuration YAML file.
        config = load_yaml("config.yaml")

        # Create an instance of the LightningModuleConfig class.
        module_config = LightningModuleConfig(**config["lightning_module_config"])

        # Create an instance of the TextMultiLabelClassificationModel class.
        model = TextMultiLabelClassificationModel(module_config)

        # Create an instance of the TrainConfig class.
        train_config = TrainConfig(**config["train_config"])

        # Create an instance of the EmailDataModule class.
        datamodule = EmailDataModule(train_dataloader=dataloader)

        # Train the model.
        trainer = pl.Trainer(**train_config.to_dict())
        trainer.fit(model, datamodule)
        ```

        The above code loads a configuration file, creates instances of
        `LightningModuleConfig`, `TextMultiLabelClassificationModel`, and
        `TrainConfig` classes, creates an instance of `EmailDataLoader` class,
        creates an instance of `EmailDataModule` class, and trains the
        `TextMultiLabelClassificationModel` using PyTorch Lightning's
        `Trainer` class.
    """

    def __init__(self, config: LightningModuleConfig):
        """Initializes the TextMultiLabelClassificationModel.

        Args:
            config: An instance of LightningModuleConfig with necessary hyperparameters.
        """
        logger.info(f"Initialize TextMultiLabelClassificationModel with config: {config}")

        super().__init__()
        self.config = config
        self.backbone = config.model.pretrained_model.model
        self.classifier = torch.nn.Linear(
            config.model.hidden_size,
            config.model.num_classes
        )
        self._step_outputs = {}

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Defines the forward pass of the module.

        Args:
            data: Contains a dictionary with the relevant inputs.

        Returns:
            The output logits of the classifier.

        Example:
            >>> data = ...
            >>> logits = model(data)
        """
        output = self.backbone(**data)
        pooled_output = output.pooler_output
        logits = self.classifier(pooled_output)
        return logits

    def _step(self, batch: dict, batch_idx: int, element: str) -> torch.Tensor:
        """Processes one batch of data during training or validation.

        Args:
            batch: A batch of training or validation data.
            batch_idx: The index of the batch.
            element: The element being processed, either "train" or "val".

        Returns:
            A torch.Tensor containing the loss for the batch.
        """
        try:
            labels = batch.pop('labels')
            data = batch
        except KeyError as e:
            logger.error(f"{element} batch is missing a required key: {e}")
            raise e

        try:
            logits = self(data)
        except Exception as e:
            logger.error(f"{element} batch processing failed with error: {e}")
            raise e

        if (element == "train") | (~hasattr(self, 'loss')):
            self.loss = self.config.define_loss.partial(labels)

        loss = self.loss(logits, labels)
        self.log(f'{element}_loss', loss)

        outputs = {'loss': loss, 'logits': logits, 'labels': labels}

        if element not in self._step_outputs.keys():
            self._step_outputs[element]: List[Dict[str, torch.Tensor]] = [outputs]
        else:
            self._step_outputs[element].append(outputs)

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Processes one batch of training data.

        Args:
            batch: A batch of training data.
            batch_idx: The index of the batch.

        Returns:
            The loss for the batch.

        Example:
            >>> train_dataloader = ...
            >>> for batch_idx, batch in enumerate(train_dataloader):
            ...     loss = model.training_step(batch, batch_idx)
        """
        return self._step(batch, batch_idx, element="train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Processes one batch of validation data.

        Args:
            batch: A batch of validation data.
            batch_idx: The index of the batch.

        Returns:
            The loss for the batch.

        Example:
            >>> val_dataloader = ...
            >>> for batch_idx, batch in enumerate(val_dataloader):
            ...     loss = model.validation_step(batch, batch_idx)
        """
        return self._step(batch, batch_idx, element="val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Processes one batch of test data.

        Args:
            batch: A batch of test data.
            batch_idx: The index of the batch.

        Returns:
            The loss for the batch.

        Example:
            >>> test_dataloader = ...
            >>> for batch_idx, batch in enumerate(test_dataloader):
            ...     loss = model.validation_step(batch, batch_idx)
        """
        return self._step(batch, batch_idx, element="test")

    def get_best_thresholds(self,
            probabilities: torch.Tensor,
            labels: torch.Tensor
        ) -> np.ndarray:
        """Computes the best thresholds for converting the predicted logits into binary class predictions.

        Args:
            probabilities (torch.Tensor): A tensor of shape (batch_size, num_classes)
                representing the predicted logits for each class.
            labels (torch.Tensor): A tensor of shape (batch_size, num_classes)
                representing the ground truth labels for each class.

        Returns:
            np.ndarray. The best thresholds, it is also stored in the `best_thresholds`
            attribute of the model.
        """
        best_thresholds = self.config.calculate_best_thresholds.partial(
            probabilities=probabilities, labels=labels
        )
        if hasattr(self, 'thresholds'):
            self.thresholds = (self.thresholds + best_thresholds) / 2
        else:
            self.thresholds = torch.tensor([0.5] * self.config.model.num_classes)
        return self.thresholds

    def _epoch_end(self, element: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the loss and logits for an epoch, and logs them to the appropriate logger.

        Args:
            element: A string representing the element for which the epoch is being
            computed (e.g., "train" or "val").

        Returns:
            A tuple containing the loss, logits, and labels for the epoch.
        """
        loss, logits, labels = self.aggregate_outputs(self._step_outputs[element])
        probabilities = self.config.predict.partial(logits)
        self.get_best_thresholds(probabilities, labels)
        predictions = self.config.model.calculate_predictions.partial(probabilities=probabilities, thresholds=self.thresholds)
        metrics = self.config.calculate_metrics.partial(labels=labels, predictions=predictions, element=element)
        self.log(f'{element}_epoch_loss', loss)
        self.log_dict(metrics, on_epoch=True)
        return loss, logits, labels

    def on_train_epoch_end(self) -> None:
        """Called at the end of the training epoch to perform any necessary operations on the epoch outputs.

        Args:
            outputs: A list of dictionaries containing the outputs from each training
            batch in the epoch.

        Returns:
            None
        """
        loss, _, _ = self.aggregate_outputs(self._step_outputs["train"])
        self.log(f'train_epoch_loss', loss)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'])

    def on_validation_epoch_end(self) -> None:
        """Performs end-of-epoch validation operations, including computing loss, logits, and metrics for the
        validation set, logging these values to the appropriate logger, and updating the best thresholds for
        calculating metrics.

        Returns:
            None
        """
        self._epoch_end(element="val")

    def on_test_epoch_end(self) -> None:
        """Processes outputs after each epoch of testing and logs the evaluation results."""
        self._epoch_end(element="test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer used for training the model.

        Returns:
            An instance of the optimizer to use for training.

        Example:
            >>> optimizer = model.configure_optimizers()
        """
        self.optimizer = self.config.optimizer.partial(params=self.parameters())
        return self.optimizer

    @staticmethod
    def aggregate_outputs(
        outputs: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate the outputs of a batch of samples.

        Args:
            outputs (List[Dict[str, Any]]): A list of output dictionaries for each sample in the batch.

        Returns:
            loss, logits, labels (tuple): A tuple containing the mean loss (as a tensor), the concatenated logits (as a tensor),
            and the concatenated labels (as a tensor).
        
        Example:
            Here's an example of how to use the `aggregate_outputs` function:

            >>> import torch
            >>> from typing import Any, Dict, List, Tuple
            >>> from your_module import aggregate_outputs

            >>> # Create a list of output dictionaries for each sample in the batch
            >>> outputs = [
            ...     {"loss": torch.tensor(0.5), "logits": torch.tensor([[0.1, 0.2], [0.3, 0.4]]), "labels": torch.tensor([[0, 1], [1, 0]])},
            ...     {"loss": torch.tensor(0.3), "logits": torch.tensor([[0.5, 0.6], [0.7, 0.8]]), "labels": torch.tensor([[1, 0], [0, 1]])},
            ...     {"loss": torch.tensor(0.2), "logits": torch.tensor([[0.9, 0.1], [0.2, 0.8]]), "labels": torch.tensor([[1, 0], [1, 0]])}
            ... ]

            >>> # Call the aggregate_outputs function to compute the mean loss and concatenate the logits and labels
            >>> mean_loss, concatenated_logits, concatenated_labels = aggregate_outputs(outputs)

            >>> # Print the results
            >>> print("Mean Loss:", mean_loss.item())
            >>> print("Concatenated Logits:\n", concatenated_logits)
            >>> print("Concatenated Labels:\n", concatenated_labels)

            Output:
            Mean Loss: 0.3333333432674408
            Concatenated Logits:
            tensor([[0.1000, 0.2000],
                    [0.3000, 0.4000],
                    [0.5000, 0.6000],
                    [0.7000, 0.8000],
                    [0.9000, 0.1000],
                    [0.2000, 0.8000]])
            Concatenated Labels:
            tensor([[0, 1],
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [1, 0],
                    [1, 0]])
        """
        # Initialize lists for storing the loss, predictions, and labels
        loss_list, logits_list, labels_list = [], [], []

        # Extract the loss, predictions, and labels from each output dictionary
        try:
            for output in outputs:
                loss_list.append(output["loss"])
                logits_list.append(output["logits"])
                labels_list.append(output["labels"])
        except Exception as e:
            logger.error(f"Error aggregating outputs: {e}")
            raise e

        # Compute the mean loss
        loss = torch.stack(loss_list).mean()

        # Concatenate the logits and labels across the batch
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        # Check if the predictions and labels are on the GPU and move them to
        # the CPU if needed to avoid memory errors.
        if logits.is_cuda:
            logits = logits.cpu()

        if labels.is_cuda:
            labels = labels.cpu()

        # Return the aggregated loss, predictions, and labels
        return loss, logits, labels

    def save(self, filename: str) -> None:
        """Saves the state of the module to a file.

        Args:
            filename: A string representing the filename to save the state to.

        Example:
            >>> model.save('model_checkpoint.pth')
        """
        state_dict = {'config': self.config, 'state_dict': self.state_dict()}
        torch.save(state_dict, filename)

    @classmethod
    def load(
        cls,
        path: str,
        map_location: Optional[str] = None
    ) -> 'TextMultiLabelClassificationModel':
        """Load a trained model from a checkpoint file.

        Args:
            path: A string representing the path to the checkpoint file.
            map_location: An optional string representing the device location
            on which to load the model.

        Returns:
            An instance of the TextMultiLabelClassificationModel model class with
            the saved state.

        Example:
            >>> model = TextMultiLabelClassificationModel.load('model_checkpoint.pth')
        """
        kwargs = dict(map_location=map_location or {})
        state_dict = torch.load(path, **kwargs)
        model = cls(config=state_dict['config'])
        model.load_state_dict(state_dict['state_dict'])
        return model

    def predict_logits(
        self, data: Dict[str,torch.tensor], ouput_tensors: bool = True
    ) -> Union[float, List[float]]:
        """Generates predictions in the form of logits.

        Args:
            data: A string or a list of strings to generate predictions for.

        Returns:
            A float or a list of floats representing the logits for the input data.

        Example:
            >>> input_ids, attention_mask = ...
            >>> logits = model.predict_logits(input_ids, attention_mask)
        """

        self.eval()

        data = {key:value.to(self.device) for key, value in data.items()}
        with torch.no_grad():
            logits = self(data)

        if ouput_tensors:
            return logits
        return logits.cpu().numpy().tolist() if len(logits.shape) > 1 else logits.cpu().item()

    def predict(self, data: Dict[str,torch.tensor]) -> torch.Tensor:
        """Generates binary predictions for the input data.

        Args:
            data: A string or a list of strings to generate predictions for.

        Returns:
            A tensor of shape (batch_size, num_classes) containing the binary predictions
            for the input data, where batch_size is the number of input examples and
            num_classes is the number of output classes.

        Example:
            >>> input_ids, attention_mask = ...
            >>> logits = model.predict(input_ids, attention_mask)
        """
        logits = self.predict_logits(data)
        probabilities = self.config.predict.partial(logits)

        if ~hasattr(self, 'thresholds'):
            logger.warning("A generic best threshold of 0.5 is being used.")
            self.thresholds = torch.tensor([0.5] * self.config.model.num_classes)

        predictions = self.config.model.calculate_predictions.partial(
            probabilities=probabilities,
            thresholds=self.thresholds
        )

        return predictions


