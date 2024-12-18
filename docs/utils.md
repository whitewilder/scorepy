Here is a more detailed version of your `utils.md` documentation, starting with function names and class names, followed by their descriptions and examples.

---

# **utils.md** Documentation

## Table of Contents

### Functions

1. [`bins_to_df(bins)`](#bins_to_df-bins)
2. [`is_monotonic(series)`](#is_monotonicseries)
3. [`filter_list(items, substring='perc')`](#filter_listitems-substringperc)
4. [`generate_pairs(results, variable_name)`](#generate_pairsresults-variable_name)
5. [`Transformation`](#transformation)
6. [`CapturePrint`](#captureprint)



### `bins_to_df(bins)`
Converts a dictionary of bins into a DataFrame sorted by Information Value (IV).

#### Parameters:
- **bins** (`dict`): A dictionary containing variable information with keys as variable names and values as dictionaries that include 'total_iv'.

#### Returns:
- **pd.DataFrame**: A DataFrame with columns 'Variable' and 'IV', sorted by IV in descending order.

#### Example:
```python
bins = {
    'var1': {'total_iv': [0.3]},
    'var2': {'total_iv': [0.5]},
    'var3': {'total_iv': [0.1]}
}

df = bins_to_df(bins)
print(df)
```

**Output:**
```
  Variable   IV
0     var2  0.5
1     var1  0.3
2     var3  0.1
```

---

### `is_monotonic(series)`
Checks if a series is monotonic (either non-increasing or non-decreasing).

#### Parameters:
- **series** (`pd.Series`): The input series to check for monotonicity.

#### Returns:
- **bool**: Returns `True` if the series is monotonic, otherwise `False`.

#### Example:
```python
import pandas as pd

series = pd.Series([1, 2, 3, 4, 5])
print(is_monotonic(series))  # Output: True

series2 = pd.Series([5, 4, 3, 2, 1])
print(is_monotonic(series2))  # Output: True

series3 = pd.Series([1, 2, 3, 2, 5])
print(is_monotonic(series3))  # Output: False
```

---

### `filter_list(items, substring='perc')`
Filters a list to include only items containing a specific substring.

#### Parameters:
- **items** (`list`): List of strings to filter.
- **substring** (`str`, optional): Substring to search for in each item. Default is 'perc'.

#### Returns:
- **list**: Filtered list containing only items with the specified substring.

#### Example:
```python
items = ['accuracy_perc', 'loss_perc', 'score', 'value_perc']
filtered_items = filter_list(items, 'perc')
print(filtered_items)  # Output: ['accuracy_perc', 'loss_perc', 'value_perc']
```

---

### `generate_pairs(results, variable_name)`
Generates grouped results for reporting or visualization.

#### Parameters:
- **results** (`dict`): Dictionary containing model outputs.
- **variable_name** (`str`): Identifier for the variable set.

#### Returns:
- **tuple**: A tuple containing two lists:
  - **plots**: List of plots (e.g., KS plot, ROC plot).
  - **data**: List of results, summary, table, and weights.

#### Example:
```python
results = {
    'ks_plot_var1': 'KS plot data',
    'roc_plot_var1': 'ROC plot data',
    'oot_roc_plot_var1': 'OOT ROC plot data',
    'result_var1': 'Model result',
    'final_summary_var1': 'Final summary',
    'calibration_table_var1': 'Calibration table',
    'weights_var1': 'Weights'
}

plots, data = generate_pairs(results, 'var1')
print(plots)  # Output: ['KS plot data', 'ROC plot data', 'OOT ROC plot data']
print(data)   # Output: ['Model result', 'Final summary', 'Calibration table', 'Weights']
```

---

### `generate_model_strings(num_models=4)`
Generates formatted strings for model outputs.

#### Parameters:
- **num_models** (`int`, optional): Number of models to generate strings for. Default is 4.

#### Returns:
- **None**: Prints formatted strings for each model in the form of `Model i`.

#### Example:
```python
generate_model_strings(2)
# Output:
# 'Model 1': generate_pairs(results=result_1, variable_name=1),
# 'Model 2': generate_pairs(results=result_2, variable_name=2),
```

---

## Classes

### `CapturePrint`
Context manager to capture printed output.

#### Attributes:
- **output** (`str`): Captured output as a string.

#### Methods:
- `__enter__(self)`: Initializes the context manager by redirecting the standard output to a `StringIO` buffer.
- `__exit__(self, exc_type, exc_val, exc_tb)`: Retrieves the captured output and restores the original standard output.

#### Example:
```python
from io import StringIO
import sys

with CapturePrint() as cp:
    print("Captured output")
print(cp.output)  # Output: "Captured output\n"
```

---

### `Transformation`
Class for applying various transformations to a DataFrame.

#### Methods:
- `__init__(self, df)`: Initializes the class with a copy of the provided DataFrame.

- `_check_column_exists(self, col)`: Checks if a column exists in the DataFrame. Raises a `ValueError` if not.

- `normalize_standardize(self, suffix='_normalize')`: Applies Z-score normalization (standardization) to numeric columns.

- `min_max_scaling(self, suffix='_minmax')`: Applies Min-Max scaling to numeric columns.

- `log_transformation(self, cols, suffix='_log')`: Applies log transformation to specified columns.

- `percentile_transformation(self, col, n=10, suffix='_perc')`: Transforms a column into percentiles (e.g., deciles).

- `ratio_terms(self, col1, col2, suffix='_ratio')`: Creates interaction terms between two columns.

- `normalize_financial_ratios(self, ratio_cols, industry_avg, suffix='_mill')`: Normalizes financial ratios relative to industry averages.

- `winsorize(self, cols, lower_quantile, upper_quantile, suffix='_winsor')`: Winsorizes columns to handle outliers by clipping values at specified quantiles.

- `differencing(self, col, suffix='_diff')`: Computes the difference between successive periods for a column.

- `encode_categorical(self, cols, suffix='_encoded')`: One-hot encodes categorical variables.

- `floor(self, col, min_value, suffix='_floor')`: Applies floor transformation to a column, setting a minimum value.

- `cap(self, col, max_value, suffix='_cap')`: Applies cap transformation to a column, setting a maximum value.

- `apply_transformations(self, ...)`: Applies all transformations to the DataFrame based on specified parameters.

#### Example:
```python
import pandas as pd

# Sample DataFrame
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50]
}
df = pd.DataFrame(data)

# Initialize Transformation class
transformer = Transformation(df)

# Apply transformations
transformed_df = transformer.apply_transformations(
    normalize=True, 
    min_max=True, 
    log_cols=['feature1'], 
    percentile_cols={'feature2': 10}
)

print(transformed_df)
```

---

