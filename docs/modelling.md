# Modelling Documentation

This document covers the implementation and examples of various functions used in the `riskpy` library for data modeling, feature selection, and evaluation.

## Table of Contents

- [Modelling Documentation](#modelling-documentation)
  - [Table of Contents](#table-of-contents)
  - [Functions](#functions)
    - [1. summary\_clean](#1-summary_clean)
    - [2. backward\_elimination](#2-backward_elimination)
    - [3. drop\_insignificant\_features](#3-drop_insignificant_features)
    - [4. get\_variable\_weights](#4-get_variable_weights)
    - [5. get\_coefs](#5-get_coefs)
    - [6. score\_df](#6-score_df)
    - [7. build\_model](#7-build_model)
    - [8. dynamic\_keys](#8-dynamic_keys)
  - [Examples](#examples)
  - [Back to Index](#back-to-index)

---

## Functions

### 1. summary_clean

**Description:**
Converts a dictionary into a structured DataFrame for summary purposes.

**Code:**
```python
def summary_clean(input_dict, dataset_name="Train"):
    # Function implementation...
```

**Parameters:**
- `input_dict` (dict): Input dictionary to structure.
- `dataset_name` (str, optional): Default is "Train".

**Returns:**
- `DataFrame`: Structured summary DataFrame.

**Example:**
```python
input_data = {"Feature1": [0.1], "Feature2": [0.2]}
summary_df = summary_clean(input_data)
print(summary_df)
```

---

### 2. backward_elimination

**Description:**
Performs backward elimination on logistic regression models.

**Code:**
```python
def backward_elimination(df, target_column, sl):
    # Function implementation...
```

**Parameters:**
- `df` (DataFrame): Dataset containing features and target.
- `target_column` (str): Target column name.
- `sl` (float): Significance level for p-values.

**Returns:**
- `DataFrame`: Features remaining after elimination.
- `LogitResults`: Final logistic regression model.

**Example:**
```python
result_features, model = backward_elimination(data, "target", 0.05)
```

---

### 3. drop_insignificant_features

**Description:**
Drops features that are not significant based on individual logistic regression.

**Code:**
```python
def drop_insignificant_features(df, target_column="target", significance_level=0.05):
    # Function implementation...
```

**Parameters:**
- `df` (DataFrame): Input dataset.
- `target_column` (str): Name of the target column.
- `significance_level` (float): Threshold for p-values.

**Returns:**
- `DataFrame`: Dataset with only significant variables.

**Example:**
```python
cleaned_data = drop_insignificant_features(df, "target", 0.05)
```

---

### 4. get_variable_weights

**Description:**
Calculates standardized weights of variables for logistic regression.

**Code:**
```python
def get_variable_weights(df, target_col, cols):
    # Function implementation...
```

**Parameters:**
- `df` (DataFrame): Dataset.
- `target_col` (str): Target column name.
- `cols` (list): List of feature column names.

**Returns:**
- `DataFrame`: Standardized weights and percentage contributions.

**Example:**
```python
weights = get_variable_weights(data, "target", ["Feature1", "Feature2"])
```

---

### 5. get_coefs

**Description:**
Extracts coefficients, p-values, and confidence intervals from a model.

**Code:**
```python
def get_coefs(model):
    # Function implementation...
```

**Parameters:**
- `model`: A fitted logistic regression model.

**Returns:**
- `DataFrame`: Summary of model coefficients.

**Example:**
```python
coef_summary = get_coefs(fitted_model)
```

---

### 6. score_df

**Description:**
Adds additional fields to financial data for cohort analysis.

**Code:**
```python
def score_df(scored_financial_dev, train_data, y="Class"):
    # Function implementation...
```

**Parameters:**
- `scored_financial_dev` (DataFrame): Scored financial data.
- `train_data` (DataFrame): Training data for mapping.
- `y` (str): Target column name.

**Returns:**
- `DataFrame`: Updated financial data.

**Example:**
```python
updated_data = score_df(scored_data, train_data)
```

---

### 7. build_model

**Description:**
Builds and evaluates a model using train, test, and OOT datasets.

**Code:**
```python
def build_model(feature_vars, train_data, test_data, oot_data, ...):
    # Function implementation...
```

**Parameters:**
- `feature_vars` (list): Features for the model.
- `train_data`, `test_data`, `oot_data` (DataFrame): Datasets.
- `target_col` (str): Target column name.

**Returns:**
- Multiple model evaluation metrics and plots.

**Example:**
```python
outputs = build_model(features, train_data, test_data, oot_data, ...)
```

---

### 8. dynamic_keys

**Description:**
Generates a dynamic dictionary of key-value pairs for model outputs.

**Code:**
```python
def dynamic_keys(features, variable_name, train_data, ...):
    # Function implementation...
```

**Parameters:**
- `features` (list): Features for modeling.
- `variable_name` (str): Variable set identifier.
- `train_data`, `test_data`, `oot_data` (DataFrame): Datasets.

**Returns:**
- `dict`: Dictionary containing model results.

**Example:**
```python
results_dict = dynamic_keys(features, "VariableSet", train_data, ...)
```

---

## Examples

Here are detailed examples demonstrating the usage of the above functions:

1. **Data Cleaning with `summary_clean`**
    ```python
    data = {"Feature1": 0.3, "Feature2": 0.7}
    df = summary_clean(data)
    print(df)
    ```

2. **Backward Elimination for Feature Selection**
    ```python
    selected_features, model = backward_elimination(dataset, "target", 0.05)
    ```

3. **Dropping Insignificant Features**
    ```python
    clean_df = drop_insignificant_features(df, target_column="Outcome")
    ```

4. **Building a Model**
    ```python
    outputs = build_model(features, train_data, test_data, oot_data, train_df, test_df, oot_df)
    ```

---

## [Back to Index](index.md)
