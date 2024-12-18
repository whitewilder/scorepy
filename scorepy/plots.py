import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from .utils import Transformation
from scipy.stats import norm
from .metrics import df_AR_AUC

def add_labels(ax, x, y, color='black', fontsize='8'):
        """
        Adds data labels to the plot.

        Parameters:
        -----------
        ax : Matplotlib axis
            Axis to which labels should be added.
        x : array-like
            X-axis values.
        y : array-like
            Y-axis values.
        color : str, optional (default='black')
            Color of the labels.
        fontsize : str or int, optional (default='8')
            Font size of the labels.
        """
        for i in range(len(x)):
            ax.text(
                i,
                y[i] + 0.05,
                y[i].round(2),
                color=color,
                fontstyle='italic',
                fontsize=fontsize
            )


def perf_eva(calib_tab, title=None, plot_type=["ks", "roc"], show_plot=True):
    """
    Performance evaluation function for KS and ROC calculations.
    Args:
        calib_tab: Input calibration table (DataFrame).
        title: Plot title.
        plot_type: List of plots to generate ['ks', 'roc'].
        show_plot: Flag to display plots.

    Returns:
        Dictionary containing KS, AUC, and Gini values along with plots.
    """

    # Add title suffix
    title = "Performance" if title is None else str(title)

    # Make copies of input table for processing
    dfrocpr = calib_tab.copy()
    df_ks = calib_tab.copy()
    df_roc = calib_tab.copy()
    dfrocpr['TPR'] = dfrocpr['fraction_defs']
    dfrocpr['FPR'] = dfrocpr['fraction_non_defs']

    # Initialize result dictionary
    rt = {}

    # -----------------------------
    # Plot: KS
    # -----------------------------
    if 'ks' in plot_type:
        # KS Calculation: maximum difference in 'diff' column
        rt['KS'] = round(
            calib_tab.loc[lambda x: x['diff'] == x['diff'].max(), 'diff'].iloc[0],
            4
        )

    # -----------------------------
    # Plot: ROC and AUC
    # -----------------------------
    if 'roc' in plot_type:
        # Calculate AUC
        auc = pd.concat(
            [
                dfrocpr[['FPR', 'TPR']],
                pd.DataFrame({'FPR': [0, 1], 'TPR': [0, 1]})
            ],
            ignore_index=True
        ).sort_values(['FPR', 'TPR']).assign(
            TPR_lag=lambda x: x['TPR'].shift(1),
            FPR_lag=lambda x: x['FPR'].shift(1)
        ).assign(
            auc=lambda x: (x['TPR'] + x['TPR_lag']) * (x['FPR'] - x['FPR_lag']) / 2
        )['auc'].sum()

        rt['AUC'] = round(auc, 4)
        rt['Gini'] = round(2 * auc - 1, 4)

    # -----------------------------
    # Export Plots
    # -----------------------------
    if show_plot:
        plist = ["eva_p" + i + "(df_" + i + ", title)" for i in plot_type]
        fig = plt.figure()
        for i, j in zip(np.arange(len(plist)), plot_type):
            rt[j + '_plt'] = eval(plist[i])
        plt.tight_layout()
        plt.show()

    return rt


def eva_proc(df, title="ROC Curve"):
    """
    Evaluate and visualize the ROC curve and calculate AUC (Area Under the Curve).

    Parameters:
    - df (DataFrame): DataFrame containing calibration data with at least the following columns:
                      - 'fraction_defs': Cumulative percentage of Bads (True Positive Rate, TPR).
                      - 'fraction_non_defs': Cumulative percentage of Goods (False Positive Rate, FPR).
    - title (str, optional): Title for the ROC plot. Default is "ROC Curve".

    Returns:
    - Matplotlib Figure: ROC curve plot with AUC and Gini index.
    """
    # Create a copy of the input DataFrame
    df_roc = df.copy()

    # Calculate TPR (True Positive Rate) and FPR (False Positive Rate)
    df_roc['TPR'] = df_roc['fraction_defs']
    df_roc['FPR'] = df_roc['fraction_non_defs']

    # Prepare data for the ROC curve by adding boundary points (0,0) and (1,1)
    df_roc = pd.concat(
        [df_roc[['FPR', 'TPR']], pd.DataFrame({'FPR': [0, 1], 'TPR': [0, 1]})],
        ignore_index=True
    ).sort_values(['FPR', 'TPR'])

    # Calculate the AUC using the trapezoidal rule
    df_roc = df_roc.assign(
        TPR_lag=lambda x: x['TPR'].shift(1),
        FPR_lag=lambda x: x['FPR'].shift(1)
    )
    auc = df_roc.assign(
        auc_segment=lambda x: (x['TPR'] + x['TPR_lag']) * (x['FPR'] - x['FPR_lag']) / 2
    )['auc_segment'].sum()

    # Plot the ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df_roc['FPR'], df_roc['TPR'], color="red", label="ROC Curve")

    # Plot the diagonal line for a random classifier
    x = np.arange(0, 1.1, 0.1)
    ax.plot(x, x, color="black", linestyle="--", label="Random Classifier")

    # Fill the area under the ROC curve
    ax.fill_between(df_roc['FPR'], 0, df_roc['TPR'], color="grey", alpha=0.1, label="AUC")

    # Set labels, title, and limits
    ax.set(
        title=f"{title} (AUC: {round(auc, 4)})",
        xlabel="False Positive Rate (FPR)",
        ylabel="True Positive Rate (TPR)",
        xlim=(0, 1),
        ylim=(0, 1),
        aspect="equal"
    )

    # Annotate the Gini index
    gini = 2 * auc - 1
    ax.text(0.55, 0.25, f"GINI: {round(gini, 2)}", color="red", 
            horizontalalignment="center", fontsize=12, bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

    # Add a legend
    ax.legend()

    # Add gridlines
    plt.grid(visible=True, linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()

    return fig


def eva_pks(calib_tab, title="K-S Plot"):
    """
    Evaluate and visualize the Kolmogorov-Smirnov (K-S) statistic from calibration data.

    Parameters:
    - calib_tab (DataFrame): DataFrame containing calibration data with at least the following columns:
                             - 'fraction_obligors': Cumulative percentage of the population.
                             - 'diff': Difference between cumulative percentages of Goods and Bads.
    - title (str, optional): Title for the K-S plot. Default is "K-S Plot".

    Returns:
    - Matplotlib Figure: K-S plot figure.
    """
    # Create a copy of the input DataFrame
    dfkslift = calib_tab.copy()

    # Calculate the K-S statistic (max difference)
    ks_stat = dfkslift['diff'].max()

    # Identify the point of maximum K-S
    ks_point_idx = dfkslift.loc[dfkslift['diff'] == ks_stat, 'fraction_obligors'].iloc[0]

    # Prepare data for plotting the K-S curve
    dfkslift1 = pd.concat([
        pd.DataFrame({'fraction_obligors': [0], 'diff': [0]}),
        dfkslift[['fraction_obligors', 'diff']]
    ], ignore_index=True)

    # Plot the K-S curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dfkslift1['fraction_obligors'], dfkslift1['diff'], color="grey", linestyle="-", label="K-S Curve")

    # Add the vertical line for the K-S point
    ax.plot([ks_point_idx, ks_point_idx], [0, ks_stat], color="red", linestyle="--", label="K-S Point")

    # Set labels, title, and limits
    ax.set(
        title=title,
        xlabel="Cumulative % of Population",
        ylabel="Difference in % (Goods - Bads)",
        xlim=(0, 1),
        ylim=(0, 1),
        aspect="equal"
    )

    # Annotate the plot with additional information
    ax.text(ks_point_idx, ks_stat, f"KS: {round(ks_stat, 4)}", color="red", 
            horizontalalignment="center", fontsize=12, bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))
    ax.text(0.2, 0.8, "Bad", horizontalalignment="center", fontsize=10, color="blue")
    ax.text(0.8, 0.55, "Good", horizontalalignment="center", fontsize=10, color="green")

    # Add a legend
    ax.legend()

    # Show the plot
    plt.grid(visible=True, linestyle="--", alpha=0.7)
    plt.show()

    return fig


def plot_chart(operating_leverage, default_rate, obs_count, chart_type='line'):
    """
    Plots a line chart or a combined bar and line chart.

    Parameters:
    - operating_leverage: List of operating leverage values.
    - default_rate: List of default rates (probit).
    - obs_count: List of observation counts.
    - chart_type: 'line' for a line chart, 'bar' for a combined bar and line chart.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    if chart_type == 'line':
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(operating_leverage, default_rate)
        r_squared = r_value**2

        # Generate linear fit line
        linear_fit = [slope * x + intercept for x in operating_leverage]

        # Plot Default Rate (probit) in red
        ax1.plot(operating_leverage, default_rate, 'o-', color='red', label='Default Rate (probit)')
        ax1.plot(operating_leverage, linear_fit, '--', color='black', label=f'Linear Fit (R² = {r_squared:.4f})')
        ax1.set_ylabel('Default Rate (probit)', color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        
        # Annotate the equation and R^2 value
        equation_text = f"y = {slope:.4f}x + {intercept:.4f}"
        ax1.text(0.25, 0.95, equation_text, transform=ax1.transAxes, fontsize=10, 
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))


    elif chart_type == 'bar':
        # Plot Observation Count as bars
        ax1.bar(operating_leverage, obs_count, color='gray', alpha=0.7, label='Observation Count',width=.05)
        ax1.set_ylabel('Observation Count', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # Overlay Default Rate (probit) as a red line
        ax2 = ax1.twinx()
        ax2.plot(operating_leverage, default_rate, 'o-', color='red', label='Default Rate (probit)')
        ax2.set_ylabel('Default Rate (probit)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    ax1.set_xlabel('Operating Leverage')
    ax1.grid(alpha=0.3)
    plt.title('Operating Leverage Chart')

    # Add legends
    ax1.legend(loc='upper right')
    if chart_type == 'bar':
        ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    if chart_type == 'line':
        print(f"Linear Equation: y = {slope:.4f}x + {intercept:.4f}")
        print(f"R² = {r_squared:.4f}")
             
# Curve-fitting functions
def exp_fit(x, a, b):
    """Exponential fit: y = a * exp(-b * x)"""
    return a * np.exp(-b * x)

def linear(x, a, b):
    """Linear fit: y = a * x + b"""
    return a * x + b

def parabolic(x, a, b, c):
    """Parabolic fit: y = a * x^2 + b * x + c"""
    return a * x**2 + b * x + c

# Plotting function
def plot_curve(dfs, var, col, var_name, curve_type="linear", show_plot=True):
    """
    Fits a curve (exponential, linear, or parabolic) to the given data and plots the result.

    Parameters:
    - dfs (pd.DataFrame): DataFrame containing the data.
    - var (str): Independent variable column name (e.g., 'median').
    - col (str): Dependent variable column name.
    - curve_type (str): Type of curve to fit ('exp_fit', 'linear', 'parabolic').
    - show_plot (bool): Whether to display the plot.

    Returns:
    - r_square (float): R-squared value of the fit.
    - fig (matplotlib.figure.Figure or None): Matplotlib figure object (if `show_plot=True`).
    - formula (str): Formula of the fitted curve.
    """
    # Select the curve-fitting function
    fit_func_map = {
        "exp_fit": exp_fit,
        "linear": linear,
        "parabolic": parabolic
    }
    if curve_type not in fit_func_map:
        raise ValueError(f"Invalid curve_type '{curve_type}'. Choose from: {list(fit_func_map.keys())}")

    fit_func = fit_func_map[curve_type]

    # Fit the curve
    x = dfs[var].values
    y = dfs[col].values
    try:
        popt, _ = curve_fit(fit_func, x, y)
    except Exception as e:
        raise ValueError(f"Curve fitting failed: {e}")

    # Generate fitted values and formula
    y_smooth = fit_func(x, *popt)
    formula = generate_formula(curve_type, popt)
    r_square = 1 - np.sum((y - y_smooth) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Plot the results if requested
    fig = None
    if show_plot:
        fig = plt.figure(figsize=(10, 5), layout="constrained")
        plt.scatter(x, y, color='red', alpha=0.6, label="Data points")
        plt.plot(x, y_smooth, color='maroon', linewidth=2, label=f"Fitted line: {formula}")
        plt.text(0.8, 0.9, f"$R^2$: {r_square:.2f}", transform=plt.gca().transAxes, fontsize=10)
        plt.xlabel(var)
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.title(f"{curve_type.capitalize()} Curve for {col} vs {var_name}")
        plt.show()

    return r_square, fig, formula

def generate_formula(curve_type, coefficients):
    """
    Generates a formula string for the fitted curve based on the coefficients.

    Parameters:
    - curve_type (str): Type of curve ('exp_fit', 'linear', 'parabolic').
    - coefficients (list): List of coefficients for the curve.

    Returns:
    - formula (str): Formula string.
    """
    if curve_type == "exp_fit":
        return f"{coefficients[0]:.2e} * exp(-{coefficients[1]:.2e} * x)"
    elif curve_type == "linear":
        return f"{coefficients[0]:.2e} * x + {coefficients[1]:.2e}"
    elif curve_type == "parabolic":
        return f"{coefficients[0]:.2e} * x^2 + {coefficients[1]:.2e} * x + {coefficients[2]:.2e}"


def plot_dr(    final_df,     var,     y,    col='logit', 
    plot='exp_fit',    n_bin=10,     show_plot=False):
    """
    Plot and calculate the relationship between a variable and default rate or logit.

    Parameters:
    - final_df (pd.DataFrame): Input DataFrame containing the data.
    - var (str): Variable name to group and analyze.
    - col (str): Column to plot ('logit' or 'DR').
    - plot (str): Type of curve to fit ('exp_fit', 'linear', 'parabolic').
    - show_plot (bool): Whether to display the plot (default: False).
    - logit (bool): Whether to calculate logit transformation of default rate.

    Returns:
    - r_square (float): R-squared value of the fitted curve.
    - fig (matplotlib.figure.Figure or None): Matplotlib figure object (if `show_plot=True`).
    - formula (str): Formula of the fitted curve.
    """
    
    transform = Transformation(final_df)
    final_df=  transform.apply_transformations(
        percentile_cols=dict(zip([var],[n_bin]))
    )
    
    # Grouping the data by percentile and aggregating required metrics
    dfs = final_df.groupby([f"{var}_perc"]).agg(
        minimum=(var, "min"),
        maximum=(var, "max"),
        mean=(var, "mean"),
        median=(var, "median"),
        obs=(var, "count"),
        defaults=(y, "sum")
    )

    # Calculate default rate (DR)
    dfs['DR'] = dfs['defaults'] / dfs['obs']

    # Calculate logit transformation if required
    
    dfs['logit'] = np.log(dfs['DR'] / (1 - dfs['DR']))
    dfs['probit'] = norm.ppf(dfs['DR'])
    
    # Ensure no invalid values (like NaN or Inf) are present in the DataFrame
    dfs = dfs[np.isfinite(dfs).all(axis=1)]

    # Fit and plot the curve
    r_square, fig, formula = plot_curve(
        dfs, 
        var='median', 
        var_name=var  ,      
        col=col, 
        curve_type=plot, 
        show_plot=show_plot
    )

    # Print the processed DataFrame (optional)
    # print(dfs)

    # Return the R-squared value, figure, and curve formula
    return r_square, fig, formula, dfs



def get_plots_results(final_df, continuous_vars,y='Class', col='logit',n_bin=10, show_plot=False):
    """
    Compute the R-squared values and formulas for various curve fits on a list of variables.

    Parameters:
    - final_df (pd.DataFrame): Input DataFrame containing the data.
    - continuous_vars (list): List of variable names to evaluate.
    - col (str): Column to analyze ('logit' or other target columns). Default is 'logit'.
    - show_plot (bool): Whether to display plots (default: False).

    Returns:
    - linear_results (list): R-squared values for linear fits.
    - parabolic_results (list): R-squared values for parabolic fits.
    - exponential_results (list): R-squared values for exponential fits.
    - linear_formulas (list): Formulas for linear fits.
    - parabolic_formulas (list): Formulas for parabolic fits.
    - exponential_formulas (list): Formulas for exponential fits.
    """
    # Initialize result containers
    linear_results, parabolic_results, exponential_results = [], [], []
    linear_formulas, parabolic_formulas, exponential_formulas = [], [], []

   
    # Iterate over each variable in the list of continuous variables
    for var in continuous_vars:
        # Linear fit
        linear_r2, _, linear_formula,_ = plot_dr(final_df, var,y, col=col, plot='linear',n_bin=n_bin, show_plot=show_plot)
        linear_results.append(linear_r2)
        linear_formulas.append(linear_formula)
 
        # Exponential fit
        exp_r2, _, exp_formula,_ = plot_dr(final_df, var,y,  col=col, plot='exp_fit',n_bin=n_bin, show_plot=show_plot)
        exponential_results.append(exp_r2)
        exponential_formulas.append(exp_formula)
    
        # Parabolic fit
        parabolic_r2, _, parabolic_formula,_ = plot_dr(final_df, var, y, col=col, plot='parabolic',n_bin=n_bin, show_plot=show_plot)
        parabolic_results.append(parabolic_r2)
        parabolic_formulas.append(parabolic_formula)

    # Return all results
    return linear_results, parabolic_results, exponential_results, linear_formulas, parabolic_formulas, exponential_formulas



def auc(dataframe, variable, target="intodef_gid"):
    """
    Calculates the Area Under the Receiver Operating Characteristic (AUROC) curve 
    for a given variable and target using a specified method.

    Parameters:
    -----------
    dataframe : DataFrame
        The input data containing the variable and target columns.
    variable : str
        The feature/variable for which the AUROC is to be calculated.
    target : str, optional (default="intodef_gid")
        The target column, typically indicating default status (binary).

    Returns:
    --------
    float
        The calculated AUROC value.
    """
    # Ensure only relevant columns are considered and drop rows with missing values
    dataframe = dataframe[[variable, target]].dropna(axis=0)

    # Set ascending order flag based on AUROC threshold
    ascending = True
    training_auc = df_AR_AUC(
        dataframe, 
        Score=variable, 
        target=target, 
        ascending=ascending, 
        method="CONCORDENT"
    )[1]["AUROC"]

    # Adjust ascending order if initial AUROC is below 0.50
    if training_auc < 0.50:
        ascending = not ascending
        training_auc = df_AR_AUC(
            dataframe, 
            Score=variable, 
            target=target, 
            ascending=ascending, 
            method="CONCORDENT"
        )[1]["AUROC"]

    return training_auc



def bins_plot(data, final_df, by_column, default_column, variable, is_na=False, return_fig=False):
    """
    Function to create a dual-axis plot to analyze a variable's impact on default rates (DR) 
    across bins. Displays both the count of observations and the logit of default rates.

    Parameters:
    -----------
    data : DataFrame
        Input data containing the relevant columns.
    final_df : DataFrame
        Final dataframe used for additional calculations.
    by_column : str (median, max, min, mean)
        The column to group data by (bins).
    default_column : str
        Column indicating default status (binary or rate).
    variable : str
        The feature/variable being analyzed.
    is_na : bool, optional (default=False)
        Whether to include NaN values in the analysis.
    return_fig : bool, optional (default=False)
        Whether to return the plot figure object.

    Returns:
    --------
    fig : Matplotlib figure object (optional)
        The generated figure, if `return_fig` is True.
    """
    # Filter data based on whether to include or exclude NaN values
    if is_na:
        filtered_data = data[data['feature'] == variable].copy()
    else:
        filtered_data = data[data['feature'] == variable].dropna(axis=0).copy()

    # Prepare data
    filtered_data[by_column] = filtered_data[by_column].round(3).astype(str)  # Format bin column
    filtered_data.index = filtered_data[by_column]  # Set bin column as index

    # Calculate additional metrics
    filtered_data['default_rate'] = filtered_data['event_rate']
    filtered_data['logit'] = np.log(filtered_data['default_rate'] / (1 - filtered_data['default_rate']))

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(9, 6), layout="constrained")
    plt.title(f"Logit of DR across bins of {variable}", fontweight='bold', fontsize=15, color='navy')

    # Primary axis: Bar plot for observation counts
    ax1.bar(
        filtered_data[by_column],
        filtered_data['sample_count'],
        color='#767676',
        label='Count of Observations',
        width=0.5,
        alpha=0.2
    )
    ax1.set_ylabel("#Observations", fontsize=12)
    ax1.set_xticklabels(filtered_data.index, rotation=45, ha='right', fontsize=12)

    # Secondary axis: Line plot for logit of default rate
    ax2 = ax1.twinx()
    filtered_data.plot(
        y='logit',
        kind='line',
        color='#D80011',
        marker='o',
        linewidth=2,
        ax=ax2
    )
    add_labels(ax2, filtered_data.index, filtered_data['logit'], color='#080011')
    ax2.set_ylabel("Logit of DR", fontsize=12)

    # Calculate Gini coefficient and display
    auc_value = auc(final_df, variable,default_column)  # Assuming `auc` is a predefined function
    gini_coefficient = abs((2 * auc_value - 1) * 100).round(2)
    ax2.text(
        x=0.0,
        y=filtered_data['logit'].median(),
        s=f"Gini: {gini_coefficient:.2f}",
        fontweight='bold'
    )
    # ax2.legend(
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, 0.4, 0, 0),
    #     ncol=4,
    #     fontsize='large'
    # )

    # Add grid lines to the secondary axis
    ax2.set_yticks(np.linspace(ax2.get_ylim()[0], ax2.get_ylim()[-1], len(ax1.get_yticks())))
    plt.grid(True)

    # Display or return the plot
    if return_fig:
        return fig
    else:
        plt.show()

    # Helper function to add labels to the secondary axis
    def add_labels(ax, x, y, color='black', fontsize='8'):
        """
        Adds data labels to the plot.

        Parameters:
        -----------
        ax : Matplotlib axis
            Axis to which labels should be added.
        x : array-like
            X-axis values.
        y : array-like
            Y-axis values.
        color : str, optional (default='black')
            Color of the labels.
        fontsize : str or int, optional (default='8')
            Font size of the labels.
        """
        for i in range(len(x)):
            ax.text(
                i,
                y[i] + 0.05,
                y[i].round(2),
                color=color,
                fontstyle='italic',
                fontsize=fontsize
            )


