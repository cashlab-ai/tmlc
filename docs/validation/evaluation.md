
# Evaluating a Text Multi-label Classification Model

In this project, we use a PyTorch Lightning Module for text multi-label classification. To evaluate the model's performance, we focus on several key metrics and their mathematical foundations. The insights provided by these metrics help us understand the strengths and weaknesses of the model.

## Metrics

The following metrics are crucial for evaluating the performance of our text multi-label classification model:

### 1. Precision (Positive Predictive Value): 

Precision measures the proportion of true alarms among the instances predicted as alarms. High precision is important in this context because it indicates that when the model predicts an alarm, it is likely to be a true alarm. A low precision may lead to a high number of false alarms, which could cause unnecessary concern or action.

**Insights:**

- High precision indicates that when the model predicts an alarm, it is likely to be a true alarm.
- Precision is particularly important in scenarios where false alarms may have significant consequences, such as causing unnecessary concern, wasting resources, or triggering unwanted actions.

**Issues:**

- A model with high precision may have low recall, which can lead to missed alarms.
- Precision alone does not provide a complete picture of the model's performance, as it does not consider the proportion of true alarms that the model identifies.

Mathematically, precision is defined as:

```
precision = TP / (TP + FP)
```

### 2. Recall (Sensitivity, True Positive Rate):

Recall measures the proportion of true alarms among the actual alarms. High recall is important because it indicates that the model is able to identify most of the true alarms. A low recall may lead to a high number of missed alarms, which could result in potential problems being overlooked.

**Insights:**

- High recall indicates that the model is able to identify most of the true alarms.
- Recall is particularly important in scenarios where missed alarms may have significant consequences, such as overlooking potential problems, safety hazards, or time-sensitive issues.

**Issues:**

- A model with high recall may have low precision, which can lead to a high number of false alarms.
- Recall alone does not provide a complete picture of the model's performance, as it does not consider the proportion of true alarms among the instances predicted as alarms.


Mathematically, recall is defined as:

```
recall = TP / (TP + FN)
```

### 3. F1 Score:

The F1 Score is the harmonic mean of precision and recall. It provides a balance between the two metrics and is particularly useful when there is an imbalance between the number of alarms and non-alarms in the dataset. A high F1 Score indicates that both precision and recall are high, which is desirable for detecting alarms in messages.

**Insights:**

-  The F1 Score provides a balance between precision and recall, making it particularly useful when there is an imbalance between the number of alarms and non-alarms in the dataset.
-  A high F1 Score indicates that both precision and recall are high, which is desirable for detecting alarms in messages.

**Issues:**

- The F1 Score assumes equal importance for precision and recall, which may not always be the case. Depending on the application, different trade-offs between precision and recall might be necessary.

Mathematically, the F1 Score is defined as:

```
F1 = 2 * (precision * recall) / (precision + recall)
```

### 4. Estimating TPR, FPR, FNR, and TNR

**Importance:**

- True Positive Rate (TPR) is equivalent to recall, and it measures the model's ability to identify true alarms.
- False Positive Rate (FPR) measures the proportion of false alarms among the instances predicted as alarms. Low FPR is desirable to minimize false alarms.
- False Negative Rate (FNR) measures the proportion of missed alarms among the actual alarms. - Low FNR is desirable to minimize missed alarms.
- True Negative Rate (TNR) measures the model's ability to identify true non-alarms. High TNR ensures that non-alarm instances are correctly classified, reducing the chance of false alarms.
- Estimating these rates, along with their average values and error bars, provides a comprehensive evaluation of the model's performance in terms of alarm detection and classification.

**Issues:**

- These rates alone may not provide a complete picture of the model's performance, as they focus on different aspects.
- Depending on the application and dataset, some rates may be more important than others. For example, in safety-critical scenarios, a higher TPR (recall) might be prioritized over a lower FPR.

By considering these insights and issues, we can make informed decisions about model development and optimization, ensuring that the model effectively detects and classifies alarms in messages.


## Estimating Performance Metrics using Bootstrapping

In this section, we will discuss how bootstrapping is used to estimate the average and error bars for performance metrics such as True Positive Rate (TPR), False Positive Rate (FPR), False Negative Rate (FNR), and True Negative Rate (TNR) in the context of multi-label text classification.

### Bootstrapping

Bootstrapping is a valuable technique for estimating the average and error bars for performance metrics, such as TPR, FPR, FNR, and TNR, in multi-label text classification. It involves generating multiple bootstrap samples from the original data to estimate the distribution of an estimator without making assumptions about the underlying data distribution. This can be particularly useful when the sample size is small or the underlying distribution is unknown.

Bootstrapping estimates the true sampling distribution by averaging over the distributions obtained from the bootstrap samples. In the context of performance metrics estimation, we generate multiple bootstrap samples of the true labels and predicted labels, calculate the TPR, FPR, FNR, and TNR for each sample, and estimate the average and error bars for each rate using the bootstrap samples.

The main benefits of using bootstrapping include robust estimation and no distribution assumptions, making it suitable for cases with small sample sizes or unknown distributions. However, bootstrapping can be computationally expensive, sensitive to the quality of the original sample, and does not guarantee improved estimation in all cases. In some instances, other techniques like cross-validation or analytical methods might be more appropriate.

### Mathematics behind Bootstrapping

For a given dataset with `n` samples, the probability of any sample being included in a bootstrap sample is `1 - (1 - 1/n)^n`, which approaches `1 - 1/e` (approximately 63.2%) as `n` goes to infinity. Consequently, about 63.2% of the original samples will be included in each bootstrap sample on average, while the remaining 36.8% will be left out.

The main idea behind bootstrapping is to approximate the true sampling distribution by averaging over the distributions obtained from the bootstrap samples. In the context of performance metrics estimation, we generate multiple bootstrap samples of the true labels and predicted labels, and then calculate the TPR, FPR, FNR, and TNR for each sample. Finally, we estimate the average and error bars for each rate using the bootstrap samples.

The error bars are calculated using the specified percentile (e.g., 2.5), such that the lower error bar is the value at the lower percentile, and the upper error bar is the value at the higher percentile (100 - percentile).

### Benefits and Limitations

The main benefits of using bootstrapping to estimate the error bars of different rates are:

1. **Robust estimation:** Bootstrapping provides a more robust estimation of the model's performance by taking into account the variability in the performance metrics.

2. **No distribution assumptions:** Bootstrapping does not require assumptions about the underlying data distribution, making it suitable for cases with small sample sizes or unknown distributions.

However, bootstrapping also has some limitations:

1. **Computationally expensive:** Bootstrapping can be computationally expensive, especially when the sample size is large or the number of iterations is high.

2. **Sensitive to the quality of the original sample:** If the original sample is not representative of the population, the bootstrapped estimates may be biased.

3. **No guarantee of improved estimation:** Although bootstrapping can provide a more robust estimation of error bars, there is no guarantee that it will always result in better estimates. In some cases, other techniques like cross-validation or analytical methods might be more appropriate.
