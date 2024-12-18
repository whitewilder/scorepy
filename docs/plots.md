### Documentation of plots with Examples

Documentation with Examples
## Table of Contents

## Table of Contents
- [Functions](#functions)
  - [1. add_labels](#function-add_labels)
  - [2. perf_eva](#function-perf_eva)
  - [3. eva_proc](#function-eva_proc)
  - [4. eva_pks](#function-eva_pks)
  - [5. plot_chart](#function-plot_chart)
  - [6. exp_fit](#function-exp_fit)
  - [7. plot_curve](#function-plot_curve)
  - [8. generate_formula](#function-generate_formula)
  - [9. get_plots_results](#function-get_plots_results)
  - [10. auc](#function-auc)
  - [Back to Index](#back-to-index)


---

### Function: `add_labels`

**Purpose:**  
Adds data labels to a plot for better readability and visual clarity.

**Parameters:**  
- `ax` (Matplotlib axis): Axis to which labels should be added.
- `x` (array-like): X-axis values.
- `y` (array-like): Y-axis values.
- `color` (str, optional): Color of the labels (default is `'black'`).
- `fontsize` (str or int, optional): Font size of the labels (default is `'8'`).

**Example:**
```python
import matplotlib.pyplot as plt

# Data
x = [0, 1, 2, 3, 4]
y = [10, 20, 25, 30, 40]

fig, ax = plt.subplots()
ax.plot(x, y)

# Add labels to the plot
add_labels(ax, x, y, color='blue', fontsize=10)

plt.show()
```

This will display a plot with data points labeled at each (x, y) position.

---

### Function: `perf_eva`

**Purpose:**  
Evaluates and visualizes the performance of a model by calculating the Kolmogorov-Smirnov (KS) statistic and Receiver Operating Characteristic (ROC) curve.

**Parameters:**  
- `calib_tab` (DataFrame): Calibration data containing the model's results.
- `title` (str, optional): Title for the plots (default is `'Performance'`).
- `plot_type` (list of str, optional): List of plots to generate (`'ks'`, `'roc'`).
- `show_plot` (bool, optional): Flag to display the plots.

**Returns:**  
- A dictionary with KS, AUC, and Gini values along with the corresponding plots.

**Example:**
```python
import pandas as pd

# Sample calibration data
calib_data = pd.DataFrame({
    'fraction_defs': [0.1, 0.2, 0.3, 0.4],
    'fraction_non_defs': [0.9, 0.8, 0.7, 0.6],
    'diff': [0.05, 0.1, 0.12, 0.15]
})

# Performance evaluation
result = perf_eva(calib_data, title='Model Evaluation', plot_type=['ks', 'roc'])
print(result)
```

This will output a dictionary containing KS, AUC, Gini values and display the performance plots for the KS and ROC curves.

---

### Function: `eva_proc`

**Purpose:**  
Evaluates and visualizes the ROC curve and calculates the Area Under the Curve (AUC).

**Parameters:**  
- `df` (DataFrame): Calibration data containing `fraction_defs` (True Positive Rate) and `fraction_non_defs` (False Positive Rate).
- `title` (str, optional): Title for the ROC plot (default is `'ROC Curve'`).

**Returns:**  
- A Matplotlib figure displaying the ROC curve.

**Example:**
```python
# Sample calibration data
calib_data = pd.DataFrame({
    'fraction_defs': [0.1, 0.2, 0.3, 0.4],
    'fraction_non_defs': [0.9, 0.8, 0.7, 0.6]
})

# Evaluate and plot the ROC curve
eva_proc(calib_data, title="ROC Curve Example")
```

This will display the ROC curve and print the calculated AUC value.

---

### Function: `eva_pks`

**Purpose:**  
Evaluates and visualizes the Kolmogorov-Smirnov (K-S) statistic from calibration data.

**Parameters:**  
- `calib_tab` (DataFrame): Calibration data containing `fraction_obligors` (cumulative population) and `diff` (difference between Goods and Bads).
- `title` (str, optional): Title for the K-S plot (default is `'K-S Plot'`).

**Returns:**  
- A Matplotlib figure displaying the K-S plot.

**Example:**
```python
# Sample calibration data
calib_data = pd.DataFrame({
    'fraction_obligors': [0.1, 0.2, 0.3, 0.4],
    'diff': [0.05, 0.1, 0.15, 0.2]
})

# Evaluate and plot the K-S curve
eva_pks(calib_data, title="K-S Plot Example")
```

This will display the K-S plot and indicate the point of maximum KS value.

---

### Function: `plot_chart`

**Purpose:**  
Plots a line chart or a combined bar and line chart to visualize operating leverage, default rate, and observation count.

**Parameters:**  
- `operating_leverage` (list): List of operating leverage values.
- `default_rate` (list): List of default rates (probit).
- `obs_count` (list): List of observation counts.
- `chart_type` (str, optional): Type of chart to plot (`'line'` or `'bar'`).

**Example:**
```python
# Sample data
operating_leverage = [0.1, 0.2, 0.3, 0.4]
default_rate = [0.05, 0.1, 0.15, 0.2]
obs_count = [100, 150, 200, 250]

# Plot a line chart
plot_chart(operating_leverage, default_rate, obs_count, chart_type='line')

# Plot a combined bar and line chart
plot_chart(operating_leverage, default_rate, obs_count, chart_type='bar')
```

The first example will generate a line chart, and the second will generate a combined bar and line chart.

--- 

### Backlinks:

- [add_labels](#add_labels)
- [perf_eva](#perf_eva)
- [eva_proc](#eva_proc)
- [eva_pks](#eva_pks)
- [plot_chart](#plot_chart)



### Function: `exp_fit`

**Purpose:**  
Fits an exponential curve to the data using the equation:  
\[ y = a \cdot \exp(-b \cdot x) \]

**Parameters:**  
- `x` (array-like): Independent variable values.
- `a` (float): Exponential coefficient.
- `b` (float): Decay rate coefficient.

**Returns:**  
- `y` (array-like): The fitted values based on the exponential function.

---

### Function: `linear`

**Purpose:**  
Fits a linear curve to the data using the equation:  
\[ y = a \cdot x + b \]

**Parameters:**  
- `x` (array-like): Independent variable values.
- `a` (float): Slope of the line.
- `b` (float): Intercept of the line.

**Returns:**  
- `y` (array-like): The fitted values based on the linear equation.

---

### Function: `parabolic`

**Purpose:**  
Fits a parabolic curve to the data using the equation:  
\[ y = a \cdot x^2 + b \cdot x + c \]

**Parameters:**  
- `x` (array-like): Independent variable values.
- `a` (float): Quadratic coefficient.
- `b` (float): Linear coefficient.
- `c` (float): Constant term.

**Returns:**  
- `y` (array-like): The fitted values based on the parabolic equation.

---

### Function: `plot_curve`

**Purpose:**  
Fits a specified curve (exponential, linear, or parabolic) to the data and plots the result.

**Parameters:**  
- `dfs` (pd.DataFrame): DataFrame containing the data.
- `var` (str): Independent variable column name.
- `col` (str): Dependent variable column name.
- `var_name` (str): Name of the independent variable.
- `curve_type` (str): Type of curve to fit (`'exp_fit'`, `'linear'`, `'parabolic'`).
- `show_plot` (bool): Flag to display the plot (default is `True`).

**Returns:**  
- `r_square` (float): R-squared value of the fitted curve.
- `fig` (matplotlib.figure.Figure or None): Matplotlib figure object (if `show_plot=True`).
- `formula` (str): Formula of the fitted curve.

**Example:**
```python
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 6, 8, 10]
})

# Plot and fit a linear curve
r_square, fig, formula = plot_curve(data, var='x', col='y', var_name='X Variable', curve_type='linear', show_plot=True)
print(f"R-squared: {r_square:.2f}")
print(f"Fitted formula: {formula}")
```

---

### Function: `generate_formula`

**Purpose:**  
Generates a formula string for the fitted curve based on the coefficients.

**Parameters:**  
- `curve_type` (str): Type of curve (`'exp_fit'`, `'linear'`, `'parabolic'`).
- `coefficients` (list): List of coefficients for the curve.

**Returns:**  
- `formula` (str): Formula string.

---

### Function: `plot_dr`

**Purpose:**  
Plots and calculates the relationship between a variable and default rate (DR) or logit.

**Parameters:**  
- `final_df` (pd.DataFrame): Input DataFrame containing the data.
- `var` (str): Variable name to group and analyze.
- `y` (str): Target variable (usually default status).
- `col` (str): Column to plot (`'logit'` or `'DR'`).
- `plot` (str): Type of curve to fit (`'exp_fit'`, `'linear'`, `'parabolic'`).
- `n_bin` (int): Number of bins for grouping.
- `show_plot` (bool): Flag to display the plot (default is `False`).

**Returns:**  
- `r_square` (float): R-squared value of the fitted curve.
- `fig` (matplotlib.figure.Figure or None): Matplotlib figure object (if `show_plot=True`).
- `formula` (str): Formula of the fitted curve.
- `dfs` (pd.DataFrame): Aggregated DataFrame with calculated values.

---

### Function: `get_plots_results`

**Purpose:**  
Computes the R-squared values and formulas for various curve fits on a list of variables.

**Parameters:**  
- `final_df` (pd.DataFrame): Input DataFrame containing the data.
- `continuous_vars` (list): List of continuous variable names to evaluate.
- `y` (str): Target variable.
- `col` (str): Column to analyze (`'logit'` or `'DR'`).
- `n_bin` (int): Number of bins for grouping.
- `show_plot` (bool): Flag to display plots (default is `False`).

**Returns:**  
- `linear_results` (list): R-squared values for linear fits.
- `parabolic_results` (list): R-squared values for parabolic fits.
- `exponential_results` (list): R-squared values for exponential fits.
- `linear_formulas` (list): Formulas for linear fits.
- `parabolic_formulas` (list): Formulas for parabolic fits.
- `exponential_formulas` (list): Formulas for exponential fits.

---

### Function: `auc`

**Purpose:**  
Calculates the Area Under the Receiver Operating Characteristic (AUROC) curve for a given variable and target.

**Parameters:**  
- `dataframe` (pd.DataFrame): DataFrame containing the variable and target columns.
- `variable` (str): Feature/variable for which AUROC is calculated.
- `target` (str, optional): Target column (default is `'intodef_gid'`).

**Returns:**  
- `float`: The calculated AUROC value.

---

### Function: `bins_plot`

**Purpose:**  
Creates a dual-axis plot to analyze a variable's impact on default rates (DR) across bins. Displays both the count of observations and the logit of default rates.

**Parameters:**  
- `data` (pd.DataFrame): Input data containing the relevant columns.
- `final_df` (pd.DataFrame): Final dataframe used for additional calculations.
- `by_column` (str): The column to group data by (bins).
- `default_column` (str): Column indicating default status.
- `variable` (str): The feature/variable being analyzed.
- `is_na` (bool, optional): Whether to include NaN values (default is `False`).
- `return_fig` (bool, optional): Whether to return the figure (default is `False`).

**Returns:**  
- `fig` (matplotlib.figure.Figure or None): The generated plot figure (if `return_fig=True`).

---


## [Back to Index](index.md)
