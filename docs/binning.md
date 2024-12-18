# Binning Documentation

This document provides an overview of functions for binning data, calculating Weight of Evidence (WOE), and Information Value (IV). Examples are included for each function.

---

## Function: `prepare_bins`

### Description
Prepares bins for a column in a DataFrame based on quantiles or monotonicity.

### Parameters
- **bin_data** (`pd.DataFrame`): The input DataFrame containing data to bin.
- **column** (`str`): The column to bin.
- **target_col** (`str`): The target column for calculating means.
- **max_bins** (`int`): Maximum number of bins to create.
- **quantile** (`bool`): If `True`, uses quantile binning; otherwise, uses monotonic binning.

### Returns
- **str**: Remarks about the binning process.
- **pd.DataFrame**: The updated DataFrame with the binned column.

### Example
```python
import pandas as pd

data = pd.DataFrame({
    "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

column, remarks, binned_data = prepare_bins(data, column="feature", target_col="target", max_bins=4)
print(remarks)
print(binned_data)
```

---

## Function: `iv_woe_4iter`

### Description
Calculates Weight of Evidence (WOE) and Information Value (IV) for each bin/class in a binned feature.

### Parameters
- **binned_data** (`pd.DataFrame`): Input DataFrame containing binned data.
- **target_col** (`str`): Name of the target column (binary: 0 or 1).
- **class_col** (`str`): Name of the column to compute WOE/IV for.

### Returns
- **pd.DataFrame**: A DataFrame with calculated IV, WOE, and distribution metrics for each bin/class.

### Example
```python
binned_data = pd.DataFrame({
    "feature_bins": ["Bin1", "Bin2", "Bin1", "Bin3", "Bin2"],
    "target": [0, 1, 0, 1, 0]
})

result = iv_woe_4iter(binned_data, target_col="target", class_col="feature")
print(result)
```

---

## Function: `var_iter`

### Description
Calculates WOE and IV for all features in a dataset.

### Parameters
- **data** (`pd.DataFrame`): Input DataFrame containing features and the target column.
- **target_col** (`str`): The name of the binary target column (0 or 1 values).
- **n_bins** (`int`): Maximum number of bins for continuous features.

### Returns
- **woe_iv** (`pd.DataFrame`): DataFrame containing WOE and IV values for all classes of features.
- **remarks_list** (`pd.DataFrame`): Remarks about the binning process for each feature.

### Example
```python
data = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
    "feature2": [8, 7, 6, 5, 4, 3, 2, 1],
    "target": [0, 1, 0, 1, 0, 1, 0, 1]
})

woe_iv, remarks = var_iter(data, target_col="target", n_bins=4)
print(woe_iv)
print(remarks)
```

---

## Function: `get_iv_woe`

### Description
Aggregates IV and WOE for all features in a dataset.

### Parameters
- **data** (`pd.DataFrame`): Input DataFrame containing features and the target column.
- **target_col** (`str`): The name of the binary target column (0 or 1 values).
- **n_bins** (`int`): Maximum number of bins for continuous features.

### Returns
- **iv_summary** (`pd.DataFrame`): Summary of IV values for each feature.
- **woe_iv** (`pd.DataFrame`): Detailed WOE and IV values for all classes of features.

### Example
```python
data = pd.DataFrame({
    "feature1": [10, 20, 30, 40, 50],
    "feature2": [15, 25, 35, 45, 55],
    "target": [0, 1, 0, 1, 0]
})

iv_summary, woe_iv = get_iv_woe(data, target_col="target", n_bins=3)
print(iv_summary)
print(woe_iv)
```

---

## Notes
- For large datasets, consider optimizing the binning process to avoid performance bottlenecks.
- Ensure the target column contains only binary values (0 and 1) before applying these functions.
- Handle missing values appropriately to ensure consistent results.

