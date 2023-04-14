from pydantic import BaseModel

from tmlc.configclasses import DatasetConfig, PartialFunctionConfig


class DataModuleConfig(BaseModel):
    """
    Configuration class for a PyTorch Lightning DataModule.

    Attributes:
        state_file (str): Path to the file to store the state of the DataModule.
        dataset (DatasetConfig): Configuration for the dataset to use.
        load_data (PartialFunctionConfig): Configuration for the function to load the data.
        split (PartialFunctionConfig): Configuration for the function to split the dataset.
        process_data (PartialFunctionConfig): Configuration for the function to process the data.
    """

    state_file: str
    dataset: DatasetConfig
    load_data: PartialFunctionConfig
    split: PartialFunctionConfig
    process_data: PartialFunctionConfig
