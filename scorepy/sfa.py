# This module contains the functions related to Single factor analysis

# Defined libraries
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import binned_statistic
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.signal import find_peaks

def summarize_data(df, variables, include_modality=False):
    """
    Generate summary statistics for a list of variables in a dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    variables (list): List of column names to summarize.
    include_modality (bool): Whether to calculate modality for numeric variables.

    Returns:
    pd.DataFrame: A dataframe containing summary statistics.
    """
    summary = []

    # Filter only requested variables
    available_vars = [var for var in variables if var in df.columns]
    missing_vars = [var for var in variables if var not in df.columns]

    if missing_vars:
        print(f"The following variables are not in the dataframe: {missing_vars}")

    # Numeric columns
    numeric_vars = [var for var in available_vars if pd.api.types.is_numeric_dtype(df[var])]
    if numeric_vars:
        data = df[numeric_vars]
        completeness = data.notnull().mean() * 100
        missing_pct = data.isnull().mean() * 100
        special_values_pct = data.apply(lambda x: x.isin([np.inf, -np.inf]).mean() * 100)
        mean = data.mean()
        std_dev = data.std()
        min_val = data.min()
        max_val = data.max()
        quantiles = data.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).transpose()
        negative_pct = (data < 0).mean() * 100
        zero_pct = (data == 0).mean() * 100

        modality = None
        if include_modality:
            def calculate_modality(series):
                try:
                    kde = gaussian_kde(series.dropna())
                    kde_values = kde(series.dropna())
                    peaks, _ = find_peaks(kde_values)
                    return "Bimodal/Multimodal" if len(peaks) > 1 else "Unimodal"
                except Exception:
                    return "Unknown"

            modality = data.apply(calculate_modality)

        for var in numeric_vars:
            summary_entry = {
                "Variable": var,
                "Completeness": f"{completeness[var]:.2f}%",
                "% Missing": missing_pct[var],
                "% Special Values": special_values_pct[var],
                "Mean": mean[var],
                "Std Dev": std_dev[var],
                "Min": min_val[var],
                "Max": max_val[var],
                "1% Quantile": quantiles.loc[var, 0.01],
                "5% Quantile": quantiles.loc[var, 0.05],
                "25% Quantile": quantiles.loc[var, 0.25],
                "50% Quantile": quantiles.loc[var, 0.5],
                "75% Quantile": quantiles.loc[var, 0.75],
                "95% Quantile": quantiles.loc[var, 0.95],
                "99% Quantile": quantiles.loc[var, 0.99],
                "% Negative Values": negative_pct[var],
                "% Zero Values": zero_pct[var]
            }
            if include_modality:
                summary_entry["Modality"] = modality[var]
            summary.append(summary_entry)

    # Categorical columns
    categorical_vars = [var for var in available_vars if not pd.api.types.is_numeric_dtype(df[var])]
    if categorical_vars:
        data = df[categorical_vars]
        completeness = data.notnull().mean() * 100
        missing_pct = data.isnull().mean() * 100

        top_categories = data.apply(lambda x: x.value_counts().idxmax() if not x.value_counts().empty else None)
        top_category_pct = data.apply(lambda x: (x.value_counts().max() / len(x) * 100) if not x.value_counts().empty else 0)

        for var in categorical_vars:
            summary.append({
                "Variable": var,
                "Completeness": f"{completeness[var]:.2f}%",
                "% Missing": missing_pct[var],
                "Top Category": top_categories[var],
                "Top Category %": top_category_pct[var]
            })

    return pd.DataFrame(summary)

def calculate_binned_statistic(
    risk_driver,
    target,
    num_bins=10,
    binning_method="equal",):
    """
    Calculate binned statistics for target vs. risk driver.

    Parameters:
    - risk_driver (array-like): The risk driver values.
    - target (array-like): The dependent variable (e.g., default indicator).
    - num_bins (int): Number of bins for equal-width binning.
    - binning_method (str): 'equal' for equal-width binning, 'monotonic' for automated monotonic binning.
    - winsorize_limits (tuple): Winsorization limits as percentiles (lower, upper).

    Returns:
    - summary_df_pre (DataFrame): Binned statistics pre-winsorization.
    - summary_df_post (DataFrame): Binned statistics post-winsorization.
    """
    # Ensure input is numpy array
    risk_driver = np.asarray(risk_driver)
    target = np.asarray(target)


    # Define binning method
    if binning_method == "equal":
        bin_edges  = pd.qcut(risk_driver, q=num_bins, duplicates="drop", retbins=True)[1] 
    elif binning_method == "monotonic":
        sorted_indices = np.argsort(risk_driver)
        sorted_risk = risk_driver[sorted_indices]
        sorted_target = target[sorted_indices]
        
        cumulative_target = np.cumsum(sorted_target)
        cumulative_target_normalized = cumulative_target / cumulative_target[-1]

        bin_edges = np.interp(
            np.linspace(0, 1, num_bins + 1),
            cumulative_target_normalized,
            sorted_risk
        )

    # Compute bin indices
    if binning_method == "monotonic":
        bin_indices = np.digitize(risk_driver, bin_edges, right=False)
    elif binning_method == "equal":
        bin_indices =  pd.qcut(risk_driver, q=num_bins, labels=False, duplicates="drop")+1

    # Helper function to calculate statistics for each bin
    def calculate_bin_statistics(bin_indices, risk_driver, target):
        bin_stats = []
        for i in range(1, len(bin_edges)):
            in_bin = bin_indices == i
            if np.any(in_bin):
                bin_risk_driver = risk_driver[in_bin]
                bin_target = target[in_bin]
                num_obs = len(bin_risk_driver)
                num_defaults = np.sum(bin_target)
                dr = np.mean(bin_target)
                probit_dr = expit(np.mean(bin_target)) if num_obs > 0 else np.nan
                logit_dr = logit(np.mean(bin_target)) if num_obs > 0 else np.nan
                bin_stats.append({
                    "bin": i,
                    "num_obs": num_obs,
                    "num_defaults": num_defaults,
                    "dr": dr,
                    "probit_dr": probit_dr,
                    "logit_dr": logit_dr,
                    "min": np.min(bin_risk_driver),
                    "max": np.max(bin_risk_driver),
                    "mean": np.mean(bin_risk_driver),
                    "median": np.median(bin_risk_driver),
                })
            else:
                bin_stats.append({
                    "bin": i,
                    "num_obs": 0,
                    "num_defaults": 0,
                    "probit_dr": np.nan,
                    "logit_dr": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "median": np.nan,
                })
        return pd.DataFrame(bin_stats)

    # Calculate statistics pre-winsorization
    summary_df_pre = calculate_bin_statistics(bin_indices, risk_driver, target)


    return summary_df_pre






