Here is the documentation for the `save_charts` function in the form of a markdown file (`save.md`), explaining the function's purpose, parameters, and providing an example:

---

# `save_charts` Function Documentation

## Purpose
The `save_charts` function saves a collection of charts and tables into an Excel file, with each sheet containing its own set of charts and tables. This function is useful for organizing and presenting multiple visualizations and datasets in a single Excel file.

## Function Signature

```python
def save_charts(data_dict, file_name="output.xlsx"):
```

## Parameters

### `data_dict` (dict)
- **Description**: A dictionary where each key is the name of a sheet and each value is a tuple containing two elements:
  1. A list of charts (figures) to be inserted into the sheet.
  2. A list of tables (pandas DataFrames) to be inserted into the sheet.
- **Example**: 

```python
{
    "Sheet1": ([chart1, chart2], [table1]),
    "Sheet2": ([chart3], [table2, table3])
}
```

### `file_name` (str, default = "output.xlsx")
- **Description**: The name of the Excel file to save the charts and tables to.
- **Example**: `"analysis_results.xlsx"`

## Returns
- **None**: The function saves the charts and tables to an Excel file with the specified name.

## Function Overview
This function performs the following tasks:
1. Creates a new Excel workbook.
2. Iterates over the `data_dict` and adds:
   - Charts (images saved in PNG format) to each sheet.
   - Tables (pandas DataFrames) with formatted headers and data.
3. Saves the workbook to the specified file.



## Example Usage

### Example 1: Saving Charts and Tables

```python
import matplotlib.pyplot as plt
import pandas as pd

# Create sample charts
chart1 = plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Chart 1")

chart2 = plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3], [9, 16, 25])
plt.title("Chart 2")

# Create sample tables
table1 = pd.DataFrame({
    'Category': ['A', 'B', 'C'],
    'Value': [100, 200, 300]
})

# Prepare the data_dict
data_dict = {
    "Sheet1": ([chart1, chart2], [table1])
}

# Save charts and tables to an Excel file
save_charts(data_dict, "output_analysis.xlsx")
```

### Example 2: Saving Multiple Sheets with Charts and Tables

```python
import matplotlib.pyplot as plt
import pandas as pd

# Create sample charts
chart1 = plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Chart 1")

chart2 = plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3], [9, 16, 25])
plt.title("Chart 2")

chart3 = plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3], [2, 3, 4])
plt.title("Chart 3")

# Create sample tables
table1 = pd.DataFrame({
    'Category': ['A', 'B', 'C'],
    'Value': [100, 200, 300]
})

table2 = pd.DataFrame({
    'Category': ['D', 'E', 'F'],
    'Value': [400, 500, 600]
})

# Prepare the data_dict for multiple sheets
data_dict = {
    "Sheet1": ([chart1, chart2], [table1]),
    "Sheet2": ([chart3], [table2])
}

# Save charts and tables to an Excel file
save_charts(data_dict, "multiple_sheets_analysis.xlsx")
```


---

