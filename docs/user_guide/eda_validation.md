# Build EDA

To generate an EDA report for the "toxic_multi_class.csv" dataset with custom options,
run the following command in the terminal:

```bash
python your_script.py create_eda --filepath "data/toxic_multi_class.csv" \
    --message_col "comment_text" --label_cols "toxic" "severe_toxic" \
        --output_file "output.md" --img_folder "."
```

Check the page [Validation/EDA](../validation/eda.md) to view the content output of the analysis,
and the default resutls for the mock data.

## Example Configuration

Here's an example configuration file for the `EDAClassifiersEvaluationConfig`. This file includes settings necessary to create the EDA.

```yaml
eda:
  output_file: docs/validation/eda.md
  message_column: "comment_text" # Column name for input text
  labels_columns: ["toxic"] # List of column names for label(s)
  get_data:
    module: "tmlc.components.get_data"
    func: "get_data"
    kwargs:
      file_path: "data/toxic_multi_class.csv" # File path for input data
  split_data:
    module: "tmlc.components.split_data"
    func: "split_data"
    kwargs:
      train_ratio: 0.8 # Train split
      val_ratio: 0.1 # Validation split
      test_ratio: 0.1 # Test split
      random_state: 42 # Random seed for splitting data
      labels_columns: ["toxic"] # List of column names for label(s)
  transformer_models:
    - pretrainedmodel:
        path: bert-base-cased
      tokenizer:
        model_name: bert-base-uncased
        path: bert-base-cased
        max_length: 128
        kwargs:
          add_special_tokens: true
          padding: "max_length"
          truncation: true
    - pretrainedmodel:
        model_name: distilbert-base-uncased
        path: distilbert-base-uncased
      tokenizer:
        model_name: distilbert-base-uncased
        path: distilbert-base-uncased
        max_length: 128
        kwargs:
          add_special_tokens: true
          padding: "max_length"
          truncation: true
    - pretrainedmodel:
        path: bert-base-uncased
      tokenizer:
        model_name: bert-base-uncased
        path: bert-base-uncased
        max_length: 128
        kwargs:
          add_special_tokens: true
          padding: "max_length"
          truncation: true
  classifiers:
      - clf:
          module: sklearn.ensemble
          func: RandomForestClassifier
        hyperparams:
          n_estimators: [50, 100, 200]
          max_depth: [2, 5, 10]
```