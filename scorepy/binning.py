import time
import numpy as np
import pandas as pd



def prepare_bins(bin_data, column, target_col, max_bins, quantile=True):
    """
    Prepares bins for a column in a DataFrame based on quantiles or monotonicity.

    Parameters:
    - bin_data (pd.DataFrame): The input DataFrame containing data to bin.
    - column (str): The column to bin.
    - target_col (str): The target column for calculating means.
    - max_bins (int): Maximum number of bins to create.
    - quantile (bool): If True, uses quantile binning; otherwise, uses monotonic binning.

    Returns:
    - str: Remarks about the binning process.
    - pd.DataFrame: The updated DataFrame with the binned column.
    """
    remarks = None
    bin_column = f"{column}_bins"

    # Quantile binning
    if quantile:
        try:
            bin_data[bin_column] = pd.qcut(bin_data[column], max_bins, duplicates="drop")
            remarks = "Binned using quantile method"
        except Exception as e:
            remarks = f"Quantile binning failed: {e}"
        return column, remarks, bin_data[[column, target_col, bin_column]].copy()

    # Monotonic binning
    for n_bins in range(max_bins, 2, -1):  # Iterate from max_bins to 3
        try:
            bin_data[bin_column] = pd.qcut(bin_data[column], n_bins, duplicates="drop")
            grouped_mean = bin_data.groupby(bin_column)[target_col].mean()
            
            # Check monotonicity
            if is_monotonic(grouped_mean):
                remarks = f"Binned into {n_bins} bins using monotonic method"
                break
        except Exception as e:
            print(f"Failed for {n_bins} bins: {e}")
            continue

    if not remarks:
        remarks = "Binning failed for all configurations."

    return column, remarks, bin_data[[column, target_col, bin_column]].copy()


def iv_woe_4iter(binned_data, target_col, class_col):
    """
    Calculates WOE (Weight of Evidence) and IV (Information Value) for each bin/class in a binned feature.

    Parameters:
    - binned_data (pd.DataFrame): Input DataFrame containing binned data.
    - target_col (str): Name of the target column (binary: 0 or 1).
    - class_col (str): Name of the column to compute WOE/IV for.

    Returns:
    - pd.DataFrame: A DataFrame with calculated IV, WOE, and distribution metrics for each bin/class.
    """
    # Add a "Missing" category to handle NaN values
    binned_data[class_col + "_bins"] = (
        binned_data[class_col + "_bins"]
        .cat.add_categories(['Missing'])
        .fillna("Missing")
    )

    # Group by bins and calculate aggregated metrics
    temp_groupby = binned_data.groupby(class_col + "_bins").agg({
        target_col: ["count", "sum", "mean"]
    }).reset_index()

    # Rename columns for clarity
    temp_groupby.columns = [
        "sample_class",
        "sample_count",
        "event_count",
        "event_rate"
    ]

    # Add additional statistics for each bin
    stats_group = binned_data.groupby(class_col + "_bins").agg({
        class_col.replace("_bins", ""): ["min", "max", "median"]
    }).reset_index()
    stats_group.columns = [
        "sample_class",
        "min_value",
        "max_value",
        "median_value"
    ]

    # Merge statistical information with grouped metrics
    temp_groupby = pd.merge(temp_groupby, stats_group, on="sample_class")

    # Calculate non-event metrics
    temp_groupby["non_event_count"] = temp_groupby["sample_count"] - temp_groupby["event_count"]
    temp_groupby["non_event_rate"] = 1 - temp_groupby["event_rate"]

    # Handle missing values
    if "Missing" in temp_groupby["sample_class"].values:
        temp_groupby["min_value"] = temp_groupby["min_value"].replace({"Missing": np.nan})
        temp_groupby["max_value"] = temp_groupby["max_value"].replace({"Missing": np.nan})

    # Add percentile for bins
    temp_groupby = temp_groupby.reset_index(drop=True)
    temp_groupby["percentile"] = temp_groupby.index + 1

    # Add feature name for context
    temp_groupby["feature"] = class_col

    # Calculate distribution of good and bad outcomes
    temp_groupby["dist_non_event"] = temp_groupby["non_event_count"] / temp_groupby["non_event_count"].sum()
    temp_groupby["dist_event"] = temp_groupby["event_count"] / temp_groupby["event_count"].sum()

    # Calculate WOE and handle infinities
    temp_groupby["woe"] = np.log(temp_groupby["dist_event"] / temp_groupby["dist_non_event"].replace(0, np.nan))
    temp_groupby["woe"] = temp_groupby["woe"].replace([np.inf, -np.inf], 0)

    # Calculate IV
    temp_groupby["iv"] = (temp_groupby["dist_event"] - temp_groupby["dist_non_event"]) * temp_groupby["woe"]

    # Replace infinities in IV with 0
    temp_groupby["iv"] = temp_groupby["iv"].replace([np.inf, -np.inf], 0)

    # Select relevant columns for the final output
    temp_groupby = temp_groupby[[
        "feature", "percentile", "sample_class", "sample_count", "min_value", "max_value", 
        "median_value", "non_event_count", "non_event_rate", "event_count", "event_rate", 
        "dist_non_event", "dist_event", "woe", "iv"
    ]]

    return temp_groupby


def var_iter(data, target_col, n_bins):
    """
    Calculates WOE (Weight of Evidence) and IV (Information Value) for all features in the dataset.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing features and the target column.
    - target_col (str): The name of the binary target column (0 or 1 values).
    - n_bins (int): Maximum number of bins for continuous features.

    Returns:
    - woe_iv (pd.DataFrame): DataFrame containing WOE and IV values for all classes of features.
    - remarks_list (pd.DataFrame): Remarks about the binning process for each feature.
    """
    woe_iv = pd.DataFrame()
    remarks_list = []

    # Iterate over all columns in the dataset
    for col in data.columns:
        if col != target_col:  # Skip the target column
            start_time = time.time()  # Start timer for this feature

            # Check if the column is numeric and non-binary for binning
            if np.issubdtype(data[col], np.number) and data[col].nunique() > 2:
                # Prepare bins and calculate WOE/IV
                class_col, remarks, binned_data = prepare_bins(data[[col, target_col]].copy(), col, target_col, n_bins)
                agg_data = iv_woe_4iter(binned_data.copy(), target_col, class_col)
                remarks_list.append({"feature": col, "remarks": remarks})
            else:
                # For categorical features, calculate WOE/IV directly
                agg_data = iv_woe_4iter(data[[col, target_col]].copy(), target_col, col)
                remarks_list.append({"feature": col, "remarks": "Categorical"})

            # Append WOE/IV results
            woe_iv = pd.concat([woe_iv, agg_data], axis=0)

            # Log time taken for this feature
            elapsed_time = round(time.time() - start_time, 2)
            print(f"Processed {col} in {elapsed_time} seconds.")

    # Return WOE/IV data and binning remarks
    return woe_iv, pd.DataFrame(remarks_list)


def get_iv_woe(data, target_col, n_bins):
    """
    Aggregates IV (Information Value) and WOE (Weight of Evidence) for all features in the dataset.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing features and the target column.
    - target_col (str): The name of the binary target column (0 or 1 values).
    - n_bins (int): Maximum number of bins for continuous features.

    Returns:
    - iv_summary (pd.DataFrame): Summary of IV values for each feature.
    - woe_iv (pd.DataFrame): Detailed WOE and IV values for all classes of features.
    """
    start_time = time.time()  # Start timer for the function

    # Calculate IV and WOE for all features
    woe_iv, binning_remarks = var_iter(data, target_col, n_bins)
    print("IV and WOE calculated for individual groups.")

    # Aggregate IV values for features
    woe_iv["feature"] = woe_iv["feature"].replace("bins", "", regex=True)
    iv_summary = woe_iv.groupby("feature")[["iv"]].agg(["sum", "count"]).reset_index()
    iv_summary.columns = ["feature", "iv_sum", "number_of_classes"]
    print("Aggregated IV values for features calculated.")

    # Calculate null percentages for features
    null_percent_data = pd.DataFrame(data.isnull().mean()).reset_index()
    null_percent_data.columns = ["feature", "feature_null_percent"]
    iv_summary = iv_summary.merge(null_percent_data, on="feature", how="left")
    print("Null percentages calculated for features.")

    # Merge binning remarks and final IV summary
    iv_summary = iv_summary.merge(binning_remarks, on="feature", how="left")
    woe_iv = woe_iv.merge(iv_summary[["feature", "iv_sum", "remarks"]], on="feature", how="left")

    # Log time taken for the function
    elapsed_time = round((time.time() - start_time) / 60, 3)
    print(f"Binning remarks added, and process completed in {elapsed_time} minutes.")

    # Return final IV summary and WOE/IV details
    return iv_summary, woe_iv.replace({"Missing": np.nan})
