import pytorch_lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader

from tmlc.configclasses import DataModuleConfig
from tmlc.dataclasses.dataset import Dataset


class DataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for loading and preparing data.

    Args:
        config (DataModuleConfig): A configuration object specifying data loading, preparation,
            and processing settings.

    Attributes:
        config (DataModuleConfig): A configuration object specifying data loading, preparation,
            and processing settings.
        train_data (Optional): Training data.
        val_data (Optional): Validation data.
        test_data (Optional): Testing data.

    Examples:
        To use this DataModule, first create an instance of `DataModuleConfig` with the desired settings
        for data loading, preparation, and processing. Then, pass that instance to the `DataModule`
        constructor:

        >>> from tmlc.configclasses import DataModuleConfig
        >>> config = DataModuleConfig()
        >>> data_module = DataModule(config)

        After creating the `DataModule` instance, you can pass it to a PyTorch Lightning `Trainer`
        instance to train your model:

        >>> from pytorch_lightning import Trainer
        >>> trainer = Trainer(gpus=1)
        >>> trainer.fit(model, datamodule=data_module)
    """

    def __init__(self, config: DataModuleConfig):
        """
        Initialize a new instance of DataModule.

        Args:
            config (DataModuleConfig): A configuration object specifying data loading, preparation, and
                processing settings.
        """
        logger.info(f"Initialize DataModule with config: {config}")

        super().__init__()
        self.config = config
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self) -> None:
        """
        A method that runs once at the very beginning of training. It is responsible for downloading
        and processing data.

        In this implementation, this method is not needed because the data is already preprocessed.
        """
        pass

    def setup(self, stage=None) -> None:
        """
        A method that runs once per process at the beginning of training. It is responsible for
        splitting data into training, validation, and test sets.

        Args:
            stage (Optional): If specified, this argument tells whether we're at the "fit" stage
                or the "test" stage.
        """
        # Load data
        self._data = {}
        if self.config.state_file:
            try:
                state_dict = torch.load(self.config.state_file)
                self.config.dataset = state_dict["config"].dataset
                for element in ["train", "val", "test"]:
                    self._data[element] = state_dict[f"{element}_data"]
                logger.info(f"Loaded datasets with config: {self.config}")
                return
            except FileNotFoundError:
                logger.debug("State file not found.")
                logger.info("Continuing with setup without load.")
            except Exception as e:
                logger.debug(f"Error loading state: {e}.")
                logger.info("Continuing with setup without load.")

        data = self.config.load_data.partial()
        self._data["train"], self._data["val"], self._data["test"] = self.config.split.partial(data)

    def _dataloader(self, element: str) -> DataLoader:
        """
        Helper method for creating a DataLoader from a given dataset.

        Args:
            element (str): One of "train", "val", or "test".

        Returns:
            DataLoader: A DataLoader object that can be used for iterating over the data.
        """
        messages = self.config.process_data.partial(self._data[element])
        dataset = Dataset(messages=messages, config=self.config.dataset)
        kwargs = self.config.dataset.kwargs or {}
        return DataLoader(dataset, batch_size=self.config.dataset.batch_size, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """
        Method that returns a DataLoader for the training set.

        Returns:
            DataLoader: A DataLoader object that can be used for iterating over the training data.
        """
        return self._dataloader(element="train")

    def val_dataloader(self) -> DataLoader:
        """
        Method that returns a DataLoader for the validation set.

        Returns:
            DataLoader: A DataLoader object that can be used for iterating over the validation data.
        """
        return self._dataloader(element="val")

    def test_dataloader(self) -> DataLoader:
        """
        Method that returns a DataLoader for the test set.

        Returns:
            DataLoader: A DataLoader object that can be used for iterating over the test data.
        """
        return self._dataloader(element="test")
