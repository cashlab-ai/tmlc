from typing import Optional, Tuple, Dict, Union, List, Any
import numpy as np
import torch
from loguru import logger
import pytorch_lightning as pl
from transformers import PreTrainedModel
from tmlc.configclasses import LightningModuleConfig
from tmlc.model.classifier import GeneralizedClassifier
from tmlc.exceptions import ConfigError  # import the custom error class

class TextMultiLabelClassificationModel(pl.LightningModule):
    """
    A PyTorch Lightning module for text multi-label classification.

    Args:
        config (LightningModuleConfig): An instance of `LightningModuleConfig` with the necessary hyperparameters.

    Attributes:
        backbone (PreTrainedModel): A pretrained transformer model from the transformers library.
        classifier (GeneralizedClassifier): A linear layer for classification.
        config (LightningModuleConfig): An instance of `LightningModuleConfig` with the necessary hyperparameters.
        thresholds (torch.Tensor): A tensor containing the best classification thresholds for each label.

    Methods:
        _epoch_end(element: str) -> Dict[str, torch.Tensor]:
            Computes the loss and logits for an epoch and logs them to the appropriate logger.
        _step(batch: Dict[str, torch.Tensor], batch_idx: int, element: str) -> Dict[str, torch.Tensor]:
            Processes one batch of data during training or validation.
        configure_optimizers() -> torch.optim.Optimizer:
            Configures the optimizer used for training.
        forward(data: Dict[str, torch.Tensor]) -> torch.Tensor:
            Defines the forward pass of the module.
        update_thresholds(probabilities: torch.Tensor, labels: torch.Tensor) -> None:
            Computes the best thresholds for converting the predicted logits into binary class predictions.
        load(path: str, map_location: Optional[Union[torch.device, str]] = None) -> None:
            Loads the state of the module from a checkpoint file.
        on_test_epoch_end() -> None:
            Processes outputs after each epoch of testing and logs the evaluation results.
        on_train_epoch_end() -> None:
            Called at the end of the training epoch to perform any necessary operations on the epoch outputs.
        on_validation_epoch_end() -> None:
            Performs end-of-epoch validation operations.
        predict(data: Dict[str, torch.Tensor]) -> torch.Tensor:
            Generates binary predictions for the input data.
        predict_logits(data: Dict[str, torch.Tensor]) -> torch.Tensor:
            Generates predictions in the form of logits.
        test_step(batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
            Processes one batch of test data.
        training_step(batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
            Processes one batch of training data.
        validation_step(batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:

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

        The above code loads a configuration file, creates instances of `LightningModuleConfig`,
        `TextMultiLabelClassificationModel`, and `TrainConfig` classes, creates an instance of `EmailDataLoader` class,
        creates an instance of `EmailDataModule` class, and trains the `TextMultiLabelClassificationModel` using PyTorch
        Lightning's `Trainer` class.
    """

    def __init__(self, config: LightningModuleConfig):
        """
        Initializes the `TextMultiLabelClassificationModel` with the specified configuration.

        Args:
            config (LightningModuleConfig): An instance of `LightningModuleConfig` with the necessary hyperparameters.
        """
        logger.info(f"Initializing `TextMultiLabelClassificationModel` with configuration: {config}")
        
        super().__init__()
        
        # Save the configuration object as an attribute of the model.
        self.config = config
        
        # Initialize the backbone pretrained transformer model.
        self.backbone: PreTrainedModel = config.model.pretrained_model.model
        self.freeze_backbone()
        
        # Initialize the classifier linear layer.
        self.classifier: GeneralizedClassifier = config.model.classifier.partial()
        if not isinstance(self.classifier, GeneralizedClassifier):
            raise ConfigError("Failed to create GeneralizedClassifier object from config.model.classifier.")
        
        # Initialize the attribute that will hold the best classification thresholds for each label.
        if not hasattr(self, "thresholds"):
            self.thresholds = None

        # Initialize a dictionary to store the outputs from each step of the training/validation process.
        self._step_outputs = {}


    def freeze_backbone(self):
        """
        Freezes the parameters of the backbone pretrained transformer model so that they are not updated during training.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreezes the parameters of the backbone pretrained transformer model so that they can be updated during training.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                classifier_additional: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Defines the forward pass of the module.

        Args:
            input_ids (Optional[torch.Tensor]): Token indices from the tokenizer. Shape: (batch_size, sequence_length).
            attention_mask (Optional[torch.Tensor]): Mask to avoid performing attention on padding token indices. Shape: (batch_size, sequence_length).
            token_type_ids (Optional[torch.Tensor]): Segment token indices to indicate first and second portions of the inputs. Shape: (batch_size, sequence_length).
            position_ids (Optional[torch.Tensor]): Indices of positions of each input sequence tokens in the position embeddings. Shape: (batch_size, sequence_length).
            classifier_additional (Optional[torch.Tensor]): Additional tensor to be concatenated with the pooled_output before passing to the classifier. Shape: (batch_size, additional_features).

        Returns:
            torch.Tensor: The output logits of the classifier. Shape: (batch_size, num_labels).

        Example:
            >>> input_ids = ...
            >>> attention_mask = ...
            >>> token_type_ids = ...
            >>> position_ids = ...
            >>> classifier_additional = ...
            >>> logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, classifier_additional=classifier_additional)
        """
        # Prepare the data dictionary to be passed to the backbone pretrained transformer model.
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids
        }
        
        # Pass the input data through the backbone pretrained transformer model.
        output = self.backbone(**data)
        
        # Extract the pooled_output from the output of the backbone pretrained transformer model.
        pooled_output = output.pooler_output

        # Concatenate the pooled_output with any additional features before passing to the classifier linear layer.
        if classifier_additional is not None:
            classifier_inputs = torch.cat((pooled_output, classifier_additional), dim=-1)
        else:
            classifier_inputs = pooled_output

        # Pass the concatenated input data to the classifier linear layer to generate the output logits.
        logits = self.classifier(classifier_inputs)
        
        return logits


    def _step(self, batch: dict, batch_idx: int, element: str) -> torch.Tensor:
        """Processes one batch of data during training or validation.

        Args:
            batch (dict): A batch of training or validation data.
            batch_idx (int): The index of the batch.
            element (str): The element being processed, either "train" or "val".

        Returns:
            torch.Tensor: A tensor containing the loss for the batch.

        Raises:
            KeyError: If the batch is missing a required key: 'labels'.

        """
        labels = batch.pop('labels', None)
        if labels is None:
            raise KeyError(f"{element} batch is missing a required key: 'labels'")

        # Compute logits for the batch
        with torch.no_grad():
            backbone_output = self.backbone(**batch)
            pooled_output = backbone_output.pooler_output

        classifier_additional = batch.pop('classifier_additional', None)

        if classifier_additional is not None:
            classifier_inputs = torch.cat((pooled_output, classifier_additional), dim=-1)
        else:
            classifier_inputs = pooled_output

        logits = self.classifier(classifier_inputs)

        # Compute loss for the batch
        if hasattr(self.config.calculate_loss_weights, 'partial'):
            loss_kwargs = self.config.calculate_loss_weights.partial(labels)

        # TODO make the loss function a config, added here as .partial()
        loss_fn = torch.nn.BCEWithLogitsLoss(**loss_kwargs)
        loss = loss_fn(logits, labels)

        # Log loss to appropriate logger
        self.log(f'{element}_loss', loss)

        # Store step outputs for logging at epoch end
        outputs = {'loss': loss, 'logits': logits, 'labels': labels}
        self._step_outputs.setdefault(element, []).append(outputs)

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

    def update_thresholds(self, probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes the best thresholds for converting the predicted logits into binary class predictions.

        Args:
            probabilities (torch.Tensor): A tensor of shape (batch_size, num_labels)
                representing the predicted logits for each class.
            labels (torch.Tensor): A tensor of shape (batch_size, num_labels)
                representing the ground truth labels for each class.

        Returns:
            torch.Tensor: The best thresholds, which are also stored in the `thresholds` attribute of the model.
        """
        self.thresholds = self.config.calculate_best_thresholds.partial(probabilities=probabilities, labels=labels)
        return self.thresholds

    def on_epoch_start(self):
        """Method called at the beginning of each epoch. 
        
        If `pretrain_classifier` is `True`, and the current epoch equals the pretrain epoch specified
        in the configuration, set `pretrain_classifier` to `False` and unfreeze the backbone.
        """
        if self.pretrain_classifier and self.current_epoch == self.config.pretrain_epochs:
            self.pretrain_classifier = False
            self.unfreeze_backbone()

    def _epoch_end(self, element: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the loss and logits for an epoch, and logs them to the appropriate logger.

        Args:
            element (str): A string representing the element for which the epoch is being
                computed (e.g., "val" or "test").

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the loss,
            logits, and labels for the epoch.

        Example:
            >>> element = 'val'
            >>> loss, logits, labels = model._epoch_end(element)
        """
        loss, logits, labels = self.aggregate_outputs(self._step_outputs[element]).values()
        probabilities = self.config.predict.partial(logits)
        if element == "val":
            self.update_thresholds(probabilities, labels)
        predictions = self.config.model.calculate_predictions.partial(probabilities=probabilities, thresholds=self.thresholds)
        metrics = self.config.calculate_metrics.partial(labels=labels, predictions=predictions, element=element)
        self.log(f'{element}_epoch_loss', loss)
        self.log_dict(metrics, on_epoch=True)
        return loss, logits, labels


    def on_train_epoch_end(self) -> None:
        """Called at the end of the training epoch to perform any necessary operations on the epoch outputs.

        Returns:
            None
        """
        train_outputs = self._step_outputs.get("train", [])
        if not train_outputs:
            return
            
        loss, _, _ = self.aggregate_outputs(train_outputs).values()
        self.log(f'train_epoch_loss', loss)

        optimizer = self.optimizers()
        if optimizer is not None:
            lr = optimizer.param_groups[0].get("lr")
            if lr is not None:
                self.log('learning_rate', lr)

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
    ) -> Dict[str, torch.Tensor]:
        """Aggregate the outputs of a batch of samples.

        Args:
            outputs (List[Dict[str, Any]]):
                A list of output dictionaries for each sample in the batch. Each dictionary must contain
                the following keys:
                    - loss (torch.Tensor): The loss value for the sample.
                    - logits (torch.Tensor): The predicted logits for the sample.
                    - labels (torch.Tensor): The true labels for the sample.

        Returns:
            A dictionary containing the following keys:
                - loss (torch.Tensor):
                    The mean loss value across the batch of samples.
                - logits (torch.Tensor):
                    The concatenated predicted logits for the batch of samples, with shape (batch_size, num_classes).
                - labels (torch.Tensor):
                    The concatenated true labels for the batch of samples, with shape (batch_size, num_classes).

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
            >>> aggregated_outputs = aggregate_outputs(outputs)
            >>> mean_loss = aggregated_outputs["loss"]
            >>> concatenated_logits = aggregated_outputs["logits"]
            >>> concatenated_labels = aggregated_outputs["labels"]

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
        for output in outputs:
            loss_list.append(output["loss"])
            logits_list.append(output["logits"])
            labels_list.append(output["labels"])

        # Compute the mean loss
        loss = torch.stack(loss_list).mean()

        # Concatenate the logits and labels across the batch
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        # Concatenate the logits and labels across the batch
        logits = torch.cat(logits_list, dim=0).detach()
        labels = torch.cat(labels_list, dim=0).detach()

        # Return the aggregated loss, predictions, and labels
        return {"loss": loss, "logits": logits, "labels": labels}

    def save(self, filename: str) -> None:
        """Saves the state of the module to a file.

        Args:
            filename: A string representing the filename to save the state to.

        Example:
            >>> model.save('model_checkpoint.pth')
        """
        state_dict = {'config': self.config, 'state_dict': self.state_dict(), 'thresholds': self.thresholds}
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
        model.thresholds = state_dict['thresholds']
        return model

    def predict_logits(
        self, data: Dict[str, torch.Tensor], output_tensors: bool = True
    ) -> Union[float, List[float]]:
        """Generate predictions in the form of logits.

        Args:
            data (Dict[str, torch.Tensor]): A dictionary containing the input data to generate predictions for.
            output_tensors (bool, optional): A boolean flag indicating whether to return logits as torch.Tensor objects
                or as Python lists of floats. Defaults to True.

        Returns:
            Union[float, List[float]]: A float or a list of floats representing the logits for the input data.

        Example:
            >>> data = {
            ...     'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            ...     'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 0]]),
            ... }
            >>> logits = model.predict_logits(data)
        """
        
        # Set the model to evaluation mode
        self.eval()

        # Move the data to the device where the model is located
        data = {key: value.to(self.device) for key, value in data.items()}

        # Generate the logits using the model
        with torch.no_grad():
            logits = self(**data)

        # Return the logits as tensors or as Python lists of floats
        if not output_tensors:
            logits = logits.cpu().numpy().tolist() if len(logits.shape) > 1 else logits.cpu().item()
        return logits

    def calculate_predictions(self, probabilities: torch.Tensor) -> torch.Tensor:
            """
            Calculates binary predictions from the input probabilities and applies thresholding.

            Args:
                probabilities (torch.Tensor): A tensor of shape (batch_size, num_labels) containing the probabilities of each label
                    for each input example, where batch_size is the number of input examples and num_labels is the number of output classes.

            Returns:
                torch.Tensor: A tensor of shape (batch_size, num_labels) containing the binary predictions for the input data,
                    where batch_size is the number of input examples and num_labels is the number of output classes.

            Raises:
                ValueError: If the shape of the input tensor is not (batch_size, num_labels).

            Example:
                >>> probabilities = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
                >>> model.thresholds = torch.tensor([0.4, 0.6])
                >>> predictions = model.calculate_predictions(probabilities)
                >>> print(predictions)
                tensor([[0, 1],
                        [1, 0],
                        [0, 1]])
            """
            if probabilities.shape != (len(probabilities), self.config.model.num_labels):
                raise ValueError(f"Invalid input tensor shape. Expected (batch_size, num_labels)={self.config.model.num_labels}, got {probabilities.shape}")
            
            return self.config.model.calculate_predictions.partial(
                probabilities=probabilities,
                thresholds=self.thresholds
            )


    def predict(self, data: Dict[str, torch.Tensor], include_probabilities: bool = False) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]]]:
        """
        Generates binary predictions for the input data and optionally computes explainability attributions.

        Args:
            data (Dict[str, torch.tensor]): A dictionary containing input tensors (input_ids, attention_mask, etc.).
            include_probabilities (bool): If True, returns probabilities along with predictions. Default: False.

        Returns:
            Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]]]: A tensor of shape (batch_size, num_labels) containing the binary predictions for the input data,
            where batch_size is the number of input examples and num_labels is the number of output classes.
            If include_probabilities is True, returns a dictionary with keys "logits", "probabilities", and "predictions",
            where "logits" contains the model logits, "probabilities" contains the predicted probabilities, and "predictions"
            contains the binary predictions. If include_probabilities is False, returns a dictionary with keys "logits" and "predictions",
            where "logits" contains the model logits and "predictions" contains the binary predictions.

        Example:
            >>> data = {'input_ids': ..., 'attention_mask': ..., ...}
            >>> predictions = model.predict(data, include_probabilities=True)
        """
        logits = self.predict_logits(data)
        probabilities = self.config.predict.partial(logits)
        predictions = self.calculate_predictions(probabilities)
        if include_probabilities:
            return {"logits": logits, "probabilities": probabilities, "predictions": predictions}
        return {"logits": logits, "predictions": predictions}
