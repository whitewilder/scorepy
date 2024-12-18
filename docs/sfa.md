### Function Documentation for `summarize_data` and `calculate_binned_statistic`

#### 1. `summarize_data`

This function computes summary statistics for the provided variables in a DataFrame. It supports both numeric and categorical variables, including specific metrics like mean, standard deviation, and quantiles for numeric variables, and the top category for categorical ones.

##### **Parameters:**
- `df` (`pd.DataFrame`): The input DataFrame containing the data.
- `variables` (`list`): List of column names in the DataFrame to summarize.
- `include_modality` (`bool`): Flag indicating whether to include modality calculations (for numeric variables). If set to `True`, the function will calculate if the variable is unimodal or multimodal.

##### **Returns:**
- `pd.DataFrame`: A DataFrame containing summary statistics for each variable, including completeness, missing values percentage, special values percentage, and more. For numeric variables, additional statistics such as modality, quantiles, and mean are provided.

##### **Example:**

```python
import pandas as pd
import numpy as np

# Example DataFrame
data = {
    'age': [23, 25, 27, np.nan, 22],
    'income': [50000, 55000, 60000, 65000, np.inf],
    'gender': ['M', 'F', 'M', 'F', 'M']
}

df = pd.DataFrame(data)

# Summarize numeric and categorical variables
summary = summarize_data(df, variables=['age', 'income', 'gender'], include_modality=True)
print(summary)
```

This will print a summary of the numeric variables `age` and `income`, along with the categorical variable `gender`.

#### 2. `calculate_binned_statistic`

This function calculates binned statistics for a given `risk_driver` and `target` variable, with the option to apply equal-width binning or monotonic binning. The statistics calculated per bin include the number of observations, defaults, and derived metrics such as the default rate (dr), probit, and logit transformations of the default rate.

##### **Parameters:**
- `risk_driver` (`array-like`): The independent variable (e.g., a predictor or risk factor).
- `target` (`array-like`): The dependent variable (e.g., default indicator or target outcome).
- `num_bins` (`int`, default=10): Number of bins to divide the data into.
- `binning_method` (`str`, default='equal'): Binning method to use. Options:
  - `'equal'`: Equal-width binning.
  - `'monotonic'`: Automated monotonic binning, sorting data by the risk driver and calculating cumulative default rates.
  
##### **Returns:**
- `summary_df_pre` (`pd.DataFrame`): A DataFrame containing the statistics for each bin pre-winsorization.

##### **Example:**

```python
import numpy as np

# Example data
risk_driver = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
target = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0])

# Calculate binned statistics using equal-width binning
binned_stats = calculate_binned_statistic(risk_driver, target, num_bins=5, binning_method="equal")
print(binned_stats)
```


## [Back to Index](index.md)