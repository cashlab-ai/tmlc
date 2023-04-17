# Interpretability Techniques for Text Multi-label Classification Models

Interpretability of machine learning models is crucial, especially in compliance applications where human experts may need to understand and audit the predictions. This document introduces various interpretability techniques that can be applied to Text Multi-label Classification models serving compliance teams.

## Layer Integrated Gradients

Layer Integrated Gradients (LIG) is an extension of Integrated Gradients, which is a feature attribution method that assigns importance scores to input features. It does so by approximating the integral of gradients with respect to the input features along a straight path from a baseline input to the actual input.

**Pros:**

- Provides fine-grained attributions for each token in the input text.
- Applies to any differentiable model.
- The attributions sum up to the difference in the output scores between the input and the baseline.

**Cons:**

- Requires multiple forward passes, which might be computationally expensive for large models or large input datasets.
- The choice of the baseline input may affect the attributions.

## DeepLift

DeepLift is a feature attribution method that assigns importance scores to input features based on their contribution to the output. It does so by comparing the activation of each neuron to a "reference activation" and computing the differences in activations throughout the network.

**Pros:**

- Provides importance scores for each input feature.
- Applicable to any differentiable model.
- Less computationally expensive compared to Integrated Gradients or Layer Integrated Gradients.

**Cons:**

- The choice of reference activations may affect the attributions.
- Attributions may not be as fine-grained as Layer Integrated Gradients.

## Layer Conductance

Layer Conductance is a generalization of Integrated Gradients for analyzing the importance of individual neurons in the hidden layers. It assigns importance scores to neurons in a particular layer with respect to the output.

**Pros:**

- Provides insight into the importance of neurons in the hidden layers.
- The attributions sum up to the difference in the output scores between the input and the baseline.
- Applicable to any differentiable model.

**Cons:**

- Requires multiple forward passes, which might be computationally expensive for large models or large input datasets.
- The choice of the baseline input may affect the attributions.

## Baseline input

The choice of baseline input plays a crucial role in determining the quality of attributions in interpretability methods like Integrated Gradients, Layer Integrated Gradients, and DeepLift. A baseline input represents a neutral or uninformative input for the model, against which the actual input is compared to explain the model's predictions.

In the context of Text Multi-label Classification models, an appropriate baseline could be:

1. **All padding tokens:** Replacing all tokens in the input text with padding tokens (e.g., [PAD] in the case of BERT-based models). This baseline represents a completely uninformative input and is a common choice for NLP models.

2. **Average embedding:** Calculate the average of embeddings for a set of representative texts, and use this as the baseline. This approach represents the "average" input in the dataset.

3. **Random baseline:** Generate a random input text by sampling tokens from the vocabulary or dataset. This can be done multiple times to get different baseline inputs, and the results can be averaged to get a more robust explanation.

The choice of baseline input depends on the problem domain, the model architecture, and the dataset. It is essential to experiment with different baseline inputs and evaluate their effect on attributions. The ideal baseline should provide clear and meaningful attributions, allowing users to understand the model's decisions and gain insights into the underlying model behavior.


### Creating an average embedding

Creating an average embedding of negative cases as a baseline input can be an interesting approach to understanding why positive cases were labeled as positive. By comparing the model's output when provided with the actual input text (positive case) against the average embedding of negative cases, you can get insights into what features of the input text made it stand out as positive.

However, there are a few points to consider when using this approach:

1. **Dataset representation:** The average embedding of negative cases should be representative of the dataset. It's important to ensure that the chosen negative cases cover a diverse range of examples and that the average embedding captures the general characteristics of negative cases.

2. **Interpretability:** When comparing the attributions between the actual input and the average embedding of negative cases, the interpretation of the results might be less intuitive. It can be more challenging to map the attributions back to specific tokens in the input text and understand the reasoning behind the model's predictions.

3. **Model assumptions:** The choice of baseline input should align with the assumptions and inductive biases of the model. Some models may have a built-in preference for a specific type of baseline, such as padding tokens, and using an average embedding of negative cases might not be optimal in such cases.

Overall, using the average embedding of negative cases as a baseline input can provide valuable insights into why positive cases were labeled as positive. However, it's important to carefully evaluate the quality of the explanations and the interpretability of the results. Comparing the attributions obtained with different baseline inputs can help you choose the most suitable baseline for your specific problem domain and model architecture.
