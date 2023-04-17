from typing import Dict, Any, List, Optional, Callable

import torch
from captum.attr import (
    LayerConductance,
    LayerIntegratedGradients,
    DeepLift,
    visualization
)
from tmlc.model import TextMultiLabelClassificationModel
from tmlc.model.interpretability.baseline import calculate_baseline

"""
def predict(data: Dict[str, torch.Tensor], *args: torch.Tensor) -> torch.Tensor:
    # Combine the input data and additional arguments
    combined_data = [data] + list(args)

    # Create a dictionary with the proper keys and tensors converted to "cpu" and long dtype
    formatted_data = {
        key: combined_data[i].to("cpu").long()
        for i, key in enumerate(self.tokenizer.output_keys)
    }

    return self.model(**formatted_data)
"""

class InterpretabilityModule:
    """
    Interpretability module for TextMultiLabelClassificationModel, providing methods for
    computing attributions, visualizing attributions, and perturbing tokens.
    """

    def __init__(self, model: TextMultiLabelClassificationModel, tokenizer: Any, forward: Optional[Callable] = None):
        """
        Initializes the InterpretabilityModule object with a model and tokenizer.

        Args:
            model (TextMultiLabelClassificationModel): The model to be interpreted.
            tokenizer (Any): The tokenizer used for the model.
            forward (Optional[Callable]): An optional function to compute the forward pass of the model. If None, 
                self.model will be used.
        """
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        forward = forward or self.model

        self.layer_integrated_gradients = LayerIntegratedGradients(
            forward, self.model.backbone.embeddings
        )
        self.deep_lift = DeepLift(forward)
        self.layer_conductance = LayerConductance(
            forward, self.model.backbone.embeddings
        )

    def attribute(
        self,
        data: List[str],
        target: int,
        n_steps: int = 50,
        baseline_method: Optional[str] = "padding",
        negative_cases: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute attributions for the given input data and target using the specified baseline method.

        Args:
            data (List[str]): A list containing the input text.
            target (int): The target class for which to compute attributions.
            n_steps (int, optional): The number of steps for the Layer Integrated Gradients method. Defaults to 50.
            baseline_method (Optional[str], optional): The method to use for calculating the baseline. Defaults to "padding".
            negative_cases (Optional[List[str]], optional): A list of negative cases for the average_embedding method. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the attributions.
        """

        # Tokenize the input data
        encoding = self.tokenizer(data)
        data = {key: torch.tensor(value).to("cpu").long() for key, value in encoding.items()}

        # Prepare the input IDs and additional arguments
        input_ids = data.pop("input_ids")
        additional_args = tuple(data.values())

        if baseline_method:
            baseline = calculate_baseline(
                method=baseline_method,
                input_text=data[0],
                negative_cases=negative_cases,
            )
        else:
            baseline = None

        # Compute the Layer Integrated Gradients attributions
        lig_attributions = self.layer_integrated_gradients.attribute(
            inputs=input_ids,
            baselines=baseline,
            target=target,
            additional_forward_args=additional_args,
            n_steps=n_steps,
        )

        # Return the attributions in a dictionary
        attributions = {"layer_integrated_gradients": lig_attributions}
        return attributions

    def visualize_attributions(
        self,
        attributions: Dict[str, torch.Tensor],
        data: List[str],
        target: int,
    ) -> None:
        """
        Visualize the attributions for the given input data and target.
        
        Args:
            attributions (Dict[str, torch.Tensor]): A dictionary containing the attributions.
            data (List[str]): A list containing the input text.
            target (int): The target class for which to visualize attributions.
        """
        # Decode input_ids to original text
        data = self.tokenizer(data)
        data = {key: torch.tensor(value).to("cpu").long() for key, value in data.items()}
        text_tokens = self.tokenizer.tokenizer.convert_ids_to_tokens(data['input_ids'][0])

        # Prepare a list of visualizations
        visualizations = []

        for method, attribution in attributions.items():
            # Sum attributions across all layers (only for Layer Integrated Gradients and Layer Conductance)
            if method == "layer_integrated_gradients" or method == "layer_conductance":
                attribution = attribution.sum(dim=-1)

            # Convert tensor to numpy
            attribution = attribution.detach().cpu().numpy()

            # Filter out padding tokens
            filtered_tokens = []
            filtered_attributions = []

            for token, attr in zip(text_tokens, attribution[0]):
                if token != self.tokenizer.tokenizer.pad_token:
                    filtered_tokens.append(token)
                    filtered_attributions.append(attr)

            # Create a visualization object for each attribution method
            visualizations.append(
                visualization.VisualizationDataRecord(
                    filtered_attributions,
                    self.model.predict(data, include_probabilities=True)["probabilities"][target].item(),
                    target,
                    target,
                    method,
                    sum(filtered_attributions),
                    filtered_tokens,
                    convergence_score=None,
                )
            )

        # Create a unified visualization of all methods
        html_output = visualization.visualize_text(visualizations)

        output_path = "image.html"
        # Save the HTML output to a file
        with open(output_path, "w") as f:
            f.write(html_output.data)

    def explain(
        self,
        data: List[str],
        target: int,
        n_steps: int = 50,
    ) -> None:
        """
        Compute and visualize attributions for the given input data and target.
        
        Args:
            data (List[str]): A list containing the input text.
            target (int): The target class for which to compute and visualize attributions.
            n_steps (int, optional): The number of steps for the Layer Integrated Gradients method. Defaults to 50.
        """
        # Compute attributions
        attributions = self.attribute(data, target, n_steps)

        # Visualize attributions
        self.visualize_attributions(attributions, data, target)

    def perturb_token(self, input_text: str, token_to_perturb: str):
        """
        Perturb a token in the input text and compare the model's output before and after perturbation.
        
        Args:
            input_text (str): The input text.
            token_to_perturb (str): The token to perturb in the input text.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The model's output before and after perturbation.
        """
        # Tokenize the input text
        input_tokens = self.tokenizer.tokenize(input_text)
        perturbed_tokens = input_tokens.copy()

        # Find the index of the token to perturb and replace it with the mask token
        token_index = perturbed_tokens.index(token_to_perturb)
        perturbed_tokens[token_index] = self.tokenizer.mask_token

        # Convert tokens to input tensors
        original_inputs = self.tokenizer.encode_plus(input_text, return_tensors="pt")
        perturbed_inputs = self.tokenizer.encode_plus(
            self.tokenizer.convert_tokens_to_string(perturbed_tokens),
            return_tensors="pt"
        )

        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            original_outputs = self.model(**original_inputs).logits
            perturbed_outputs = self.model(**perturbed_inputs).logits

        return original_outputs, perturbed_outputs

