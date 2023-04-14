# Dataset Requirements

This documentation page provides an overview of the dataset requirements and structure for the `TextMultiLabelClassificationModel` project.

## Overview

The dataset used in this project is a CSV file containing text samples along with their respective multi-label classifications. Each sample can have multiple labels, and each label represents a different aspect of the text.

## Structure

The dataset should be structured as a CSV file with the following structure:

- `id`: Unique identifier for each data point. Is optional
- `comment_text`: The text of the comment or message.
- The remaining columns represent the labels for each sample. In the following example, the labels are: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

Each label column should have binary values, with `1` indicating the presence of the label and `0` indicating its absence.

## Example
Here is an example of the dataset structure:

```bash
"id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
"1","I really enjoyed the movie. It had great acting and a compelling plot.","0","0","0","0","0","0"
"2","This restaurant serves terrible food. I got food poisoning from eating here.","1","0","1","0","1","0"
"3","The speaker at the conference made several offensive comments about women and minorities.","1","0","1","1","1","1"
"4","I disagree with your opinion on this topic, but I respect your right to express it.","0","0","0","0","0","0"
"5","The graffiti on this building is a blight on the neighborhood.","1","0","1","0","1","0"
"6","I can't believe you would say something so racist. You should be ashamed of yourself.","1","0","1","0","1","1"
"7","I think the author of this article is completely wrong. Their arguments are flawed.","0","0","0","0","1","0"
"8","I'm sorry, but your behavior is unacceptable. Please stop harassing me.","1","1","0","1","1","1"
"9","This product is a ripoff. Don't waste your money on it.","1","0","1","0","1","0"
"10","I'm so sick of all the hate and vitriol on social media these days.","1","0","1","0","1","1"
```

## Preprocessing

Before using the dataset for training and evaluation, it should be preprocessed to ensure the text data is properly tokenized and encoded. The `TextMultiLabelClassificationModel` project provides a data preprocessing pipeline to handle this process.
