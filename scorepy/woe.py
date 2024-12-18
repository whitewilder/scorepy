import numpy as np
import pandas as pd

def iv_woe_cat(binned_data, target_col, class_col):
    """
    Calculate the Information Value (IV) and Weight of Evidence (WoE) for categorical variables.
    
    Parameters:
    - binned_data (DataFrame): Data containing the binned categorical feature and target variable.
    - target_col (str): Name of the target variable column (binary: 0/1).
    - class_col (str): Name of the categorical feature column.
    
    Returns:
    - DataFrame: A summary table with WoE, IV, and distribution metrics for the categories.
    """
    # Handle missing values in the categorical column
    binned_data[class_col] = binned_data[class_col].fillna("Missing")
    
    # Group by the categorical feature and calculate aggregated metrics
    temp_groupby = binned_data.groupby(class_col).agg({
        target_col: ["count", "sum"]
    }).reset_index()
    
    # Rename columns for clarity
    temp_groupby.columns = ["sample_class", "sample_count", "event_count"]
    
    # Calculate additional metrics
    temp_groupby["non_event_count"] = temp_groupby["sample_count"] - temp_groupby["event_count"]
    temp_groupby["event_rate"] = temp_groupby["event_count"] / temp_groupby["sample_count"]
    temp_groupby["non_event_rate"] = 1 - temp_groupby["event_rate"]
    
    # Add ranking and feature name for traceability
    temp_groupby["percentile"] = temp_groupby.reset_index().index + 1
    temp_groupby["feature"] = class_col
    
    # Rearrange columns
    temp_groupby = temp_groupby[[
        "feature", "percentile", "sample_class", "sample_count", 
        "non_event_count", "non_event_rate", "event_count", "event_rate"
    ]]
    
    # Calculate distribution of non-events and events
    temp_groupby["distbn_non_event"] = temp_groupby["non_event_count"] / temp_groupby["non_event_count"].sum()
    temp_groupby["distbn_event"] = temp_groupby["event_count"] / temp_groupby["event_count"].sum()
    
    # Calculate Weight of Evidence (WoE)
    temp_groupby["woe"] = np.log(temp_groupby["distbn_non_event"] / temp_groupby["distbn_event"].replace(0, np.nan))
    temp_groupby["woe"] = temp_groupby["woe"].replace([np.inf, -np.inf, np.nan], 0)
    
    # Calculate Information Value (IV)
    temp_groupby["iv"] = (temp_groupby["distbn_non_event"] - temp_groupby["distbn_event"]) * temp_groupby["woe"]
    temp_groupby["iv"] = temp_groupby["iv"].replace([np.inf, -np.inf, np.nan], 0)
    
    return temp_groupby


def cat_var_iter(data, target_col, var_list):
    """
    Iteratively calculate WoE and IV for a list of categorical variables.

    Parameters:
    - data (DataFrame): Input dataset.
    - target_col (str): Name of the target column (binary: 0/1).
    - var_list (list): List of categorical variable names to evaluate.

    Returns:
    - DataFrame: Consolidated WoE/IV information for all categorical variables.
    """
    woe_iv = []  # List to store WoE/IV results for each variable

    for col in var_list:
        if col != target_col:  # Ensure the target column is not included in the analysis
            # Calculate WoE and IV using the `iv_woe_cat` function
            temp_groupby = iv_woe_cat(data.copy(), target_col, col)
            woe_iv.append(temp_groupby)
    
    # Combine all results into a single DataFrame
    woe_iv_df = pd.concat(woe_iv, ignore_index=True)
    
    return woe_iv_df


def map_cat_woe(df, woe_df, var_list):
    """
    Map WoE values to the original dataset for specified categorical variables.

    Parameters:
    - df (DataFrame): Original dataset.
    - woe_df (DataFrame): DataFrame containing WoE values for each category.
    - var_list (list): List of categorical variables to map WoE.

    Returns:
    - DataFrame: Original dataset with WoE values mapped to the specified variables.
    """
    df_copy = df.copy()  # Create a copy of the original dataset
    
    for var in var_list:
        # Filter WoE data for the current variable
        var_woe = woe_df[woe_df["feature"] == var]
        
        # Create a dictionary to map categories to their respective WoE
        woe_dict = dict(zip(var_woe["sample_class"].astype(object), var_woe["woe"]))
        
        # Map the WoE values to the original data, using a default value for missing categories
        df_copy[var] = df_copy[var].fillna("Missing").map(woe_dict).fillna(-999)
    
    return df_copy
