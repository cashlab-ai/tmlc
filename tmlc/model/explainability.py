from typing import Dict, Tuple, Any, List

import torch
from captum.attr import (
    LayerConductance,
    LayerIntegratedGradients,
    GradientShap,
    DeepLift,
)
from tmlc.model import TextMultiLabelClassificationModel
from captum.attr import visualization as viz

class InterpretabilityModule:
    def __init__(self, model: TextMultiLabelClassificationModel, tokenizer: Any):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer

        def predict(data: Dict[str, torch.Tensor], *args: torch.Tensor) -> torch.Tensor:
            # Combine the input data and additional arguments
            combined_data = [data] + list(args)

            # Create a dictionary with the proper keys and tensors converted to "cpu" and long dtype
            formatted_data = {
                key: combined_data[i].to("cpu").long()
                for i, key in enumerate(self.tokenizer.output_keys)
            }

            return self.model(formatted_data)

        self.layer_integrated_gradients = LayerIntegratedGradients(
            predict, self.model.backbone.embeddings
        )
        self.deep_lift = DeepLift(predict)
        self.layer_conductance = LayerConductance(
            predict, self.model.backbone.embeddings
        )

    def attribute(
        self,
        data: List[str],
        target: int,
        n_steps: int = 50,
    ) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(data)
        data = {key: torch.tensor(value).to("cpu").long() for key, value in encoding.items()}

        input_ids = data.pop("input_ids")
        additional_forward_args = tuple(data.values())
        baseline = torch.zeros_like(input_ids)

        attributions = {
            "layer_integrated_gradients": self.layer_integrated_gradients.attribute(
                inputs=input_ids,
                baselines=baseline,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
            )
        }

        return attributions

    def visualize_attributions(
        self,
        attributions: Dict[str, torch.Tensor],
        data: List[str],
        target: int,
    ) -> None:
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
                if token != self.tokenizer.pad_token:
                    filtered_tokens.append(token)
                    filtered_attributions.append(attr)

            # Create a visualization object for each attribution method
            visualizations.append(
                viz.VisualizationDataRecord(
                    filtered_attributions,
                    self.model.predict(data)[target].item(),
                    target,
                    target,
                    method,
                    sum(filtered_attributions),
                    filtered_tokens,
                    convergence_score=None,
                )
            )

        # Create a unified visualization of all methods
        html_output = viz.visualize_text(visualizations)
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

        # Compute attributions
        attributions = self.attribute(data, target, n_steps)

        # Visualize attributions
        self.visualize_attributions(attributions, data, target)

    def perturb_token(self, input_text: str, token_to_perturb: str):
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
