# Interpretability Techniques for Text Multi-label Classification Models

Interpretability of machine learning models is crucial, especially in compliance applications where human experts may need to understand and audit the predictions. This document introduces various interpretability techniques that can be applied to Text Multi-label Classification models serving compliance teams.

## Layer Integrated Gradients

Layer Integrated Gradients (LIG) is an extension of Integrated Gradients, which is a feature attribution method that assigns importance scores to input features. It does so by approximating the integral of gradients with respect to the input features along a straight path from a baseline input to the actual input.

!!! success "Pros"
    - Provides fine-grained attributions for each token in the input text.
    - Applies to any differentiable model.
    - The attributions sum up to the difference in the output scores between the input and the baseline.

!!! failure "Cons"
    - Requires multiple forward passes, which might be computationally expensive for large models or large input datasets.
    - The choice of the baseline input may affect the attributions.

## DeepLift

DeepLift is a feature attribution method that assigns importance scores to input features based on their contribution to the output. It does so by comparing the activation of each neuron to a "reference activation" and computing the differences in activations throughout the network.

!!! success "Pros"
    - Provides importance scores for each input feature.
    - Applicable to any differentiable model.
    - Less computationally expensive compared to Integrated Gradients or Layer Integrated Gradients.

!!! failure "Cons"
    - The choice of reference activations may affect the attributions.
    - Attributions may not be as fine-grained as Layer Integrated Gradients.

## Layer Conductance

Layer Conductance is a generalization of Integrated Gradients for analyzing the importance of individual neurons in the hidden layers. It assigns importance scores to neurons in a particular layer with respect to the output.

!!! success "Pros"
    - Provides insight into the importance of neurons in the hidden layers.
    - The attributions sum up to the difference in the output scores between the input and the baseline.
    - Applicable to any differentiable model.

!!! failure "Cons"
    - Requires multiple forward passes, which might be computationally expensive for large models or large input datasets.
    - The choice of the baseline input may affect the attributions.

## Baseline input

The choice of baseline input plays a crucial role in determining the quality of attributions in interpretability methods like Integrated Gradients, Layer Integrated Gradients, and DeepLift. A baseline input represents a neutral or uninformative input for the model, against which the actual input is compared to explain the model's predictions.

In the context of Text Multi-label Classification models, an appropriate baseline could be:

1. **All padding tokens:** Replacing all tokens in the input text with padding tokens (e.g., [PAD] in the case of BERT-based models). This baseline represents a completely uninformative input and is a common choice for NLP models.
2. **Average embedding:** Calculate the average of embeddings for a set of representative texts, and use this as the baseline. This approach represents the "average" input in the dataset.
3. **Random baseline:** Generate a random input text by sampling tokens from the vocabulary or dataset. This can be done multiple times to get different baseline inputs, and the results can be averaged to get a more robust explanation.

The choice of baseline input depends on the problem domain, the model architecture, and the dataset. It is essential to experiment with different baseline inputs and evaluate their effect on attributions. The ideal baseline should provide clear and meaningful attributions, allowing users to understand the model's decisions and gain insights into the underlying model behavior.
