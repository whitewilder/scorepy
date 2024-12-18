
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
# from .utils import summary_clean ,calculate_auc_roc
from .plots import perf_eva
from .metrics import df_AR_AUC,calculate_auc_roc


def summary_clean(input_dict, dataset_name="Train"):
    """
    Clean and summarize a dictionary into a structured DataFrame.

    Parameters:
    - input_dict (dict): Dictionary to convert into a DataFrame.
    - dataset_name (str, optional): Name of the dataset for reference. Default is "Train".

    Returns:
    - DataFrame: Cleaned and structured summary.
    """
    # Convert the dictionary to a DataFrame
    summary_df = pd.DataFrame(input_dict.items()).T.iloc[:, 0:2]

    # Use the first row as column names and drop it
    summary_df.columns = summary_df.iloc[0].values
    summary_df = summary_df.drop([0], axis=0)

    # Add a new column to identify the dataset
    summary_df["dataset"] = dataset_name

    return summary_df


def backward_elimination(df, target_column, sl):
    """
    Perform backward elimination for logistic regression based on p-values.
    
    Parameters:
        df (pd.DataFrame): The dataset including features and target.
        target_column (str): The name of the target variable column in the DataFrame.
        sl (float): The significance level for p-values. Variables with p-values greater 
                    than this threshold are removed.
    
    Returns:
        pd.DataFrame: The DataFrame containing only the features that remained 
                      after backward elimination.
        sm.LogitResults: The final logistic regression model after backward elimination.
    """
    # Extract target variable (Y) and features (X)
    Y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Add constant to the model (intercept)
    X = sm.add_constant(X)
    
    # Perform backward elimination
    while True:
        try:
            # Fit logistic regression model
            regressor_OLS = sm.Logit(Y.astype(float), X.astype(float)).fit(disp=False)
            
            # Get p-values of the model
            pvalues = regressor_OLS.pvalues
            max_pvalue = pvalues.max()
            
            # Check if the maximum p-value exceeds the significance level
            if max_pvalue > sl:
                # Remove the variable with the highest p-value
                worst_var = pvalues.idxmax()
                print(f"Removing variable '{worst_var}' with p-value: {max_pvalue:.4f}")
                X = X.drop(columns=[worst_var])
            else:
                # Stop the loop when all variables are significant
                break
        
        except PerfectSeparationError:
            # Handle perfect separation error
            print("Perfect separation detected; stopping backward elimination.")
            break
    
    # Drop constant column if needed
    if 'const' in X.columns:
        X = X.drop(columns=['const'])
    
    # Print summary of the final model
    if not X.empty:
        print("\nFinal Model Summary:")
        print(regressor_OLS.summary())
    else:
        print("No variables remaining after backward elimination.")
    
    return X, regressor_OLS if not X.empty else None


def drop_insignificant_features(df, target_column='target', significance_level=0.05):
    """
    Drops variables from a DataFrame when either the constant term or the variable 
    itself becomes insignificant (p-value > significance_level) after fitting 
    individual logistic regression models.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing the features and target variable.
        target_column (str): Name of the target variable column in the DataFrame.
        significance_level (float): Threshold for p-values to determine significance.
    
    Returns:
        pd.DataFrame: A DataFrame containing only the significant variables.
    """
    
    # Extract target variable (Y) and features (X)
    Y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # List to store insignificant variables
    removed_vars = []
    
    # Loop through each variable in X
    for col in X.columns:
        # Add constant to the variable
        X_single = sm.add_constant(X[[col]])
        
        try:
            # Fit the logistic regression model
            model = sm.Logit(Y.astype(float), X_single.astype(float)).fit(disp=False)
            pvalues = model.pvalues
            
            # Check if the constant term is insignificant
            if pvalues['const'] > significance_level:
                print(f"Constant term is insignificant for variable '{col}' with p-value: {pvalues['const']:.4f}")
                removed_vars.append(col)
            # Check if the variable itself is insignificant
            elif pvalues[col] > significance_level:
                print(f"Variable '{col}' is insignificant with p-value: {pvalues[col]:.4f}")
                removed_vars.append(col)
        
        except PerfectSeparationError:
            # Handle perfect separation error
            print(f"Perfect separation detected for variable '{col}'; dropping it.")
            removed_vars.append(col)
    
    # Retain only the significant variables
    significant_vars = [col for col in X.columns if col not in removed_vars]
    print(f"Significant variables retained: {significant_vars}")
    
    # Return a DataFrame with only significant variables
    return df[[target_column] + significant_vars]


def get_variable_weights(df, target_col, cols):
    """
    Calculate standardized variable weights for logistic regression.

    Parameters:
    - df (DataFrame): Input DataFrame containing the data.
    - target_col (str): The target column name for the dependent variable.

    Returns:
    - DataFrame: A DataFrame containing variable names, standardized estimates, 
                 and percentage contribution (weights).
    """
    # Separate independent and dependent variables
    df_copy = df.copy()
    independent_vars = cols
    X = df_copy[independent_vars]
    y = df_copy[target_col].values

    # Standardize the independent variables
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X_std = pd.DataFrame(X_std, columns=independent_vars)

    # Add a constant term for the intercept
    X_std = sm.add_constant(X_std, prepend=True)

    # Fit a logistic regression model using statsmodels
    logit_model = sm.Logit(y, X_std)
    result = logit_model.fit(disp=False)  # Suppress model output

    # Extract coefficients
    coefficients = result.params[independent_vars].values

    # Calculate absolute weights and percentage contributions
    variable_weights = pd.DataFrame({
        'Variable': independent_vars,
        'Standardized_Estimate': coefficients,
        'Absolute_Weight': np.abs(coefficients)
    })
    variable_weights['Weight (%)'] = round(
        100 * variable_weights['Absolute_Weight'] / variable_weights['Absolute_Weight'].sum(), 2
    )

    # Select relevant columns and return
    variable_weights = variable_weights[['Variable', 'Standardized_Estimate', 'Weight (%)']]
    return variable_weights


def get_coefs(model):
    """
    Extract coefficients, standard errors, p-values, and confidence intervals from a fitted model.

    Parameters:
    - model: A fitted statsmodels model (e.g., logistic regression).

    Returns:
    - DataFrame: A DataFrame containing variable names, coefficients, standard errors, p-values, 
                 and confidence intervals.
    """
    # Extract model parameters
    coefficients = model.params.values
    p_values = model.pvalues.values
    std_errors = model.bse.values
    variables = model.params.index

    # Create results DataFrame
    results = pd.DataFrame({
        "variable": variables,
        "estimate": coefficients,
        "Std_Error": std_errors,
        "p_value": p_values,
        "Lower_Limit": coefficients - 2 * std_errors,
        "Upper_Limit": coefficients + 2 * std_errors
    })

    # Print model summary
    print("The model summary is as follows:\n")
    print(model.summary())
    print("\n")

    return results


def score_df(scored_financial_dev, train_data, y='Class'):
    """
    Enrich the scored financial data by adding cohort year, final CRR, model CRR, 
    and calculating Loss Rate Adjusted Default Rate (LRADR).
    
    Args:
        scored_financial_dev (pd.DataFrame): Scored financial data containing scores and IDs.
        train_data (pd.DataFrame): Training data containing CRR and cohort information.
    
    Returns:
        pd.DataFrame: Updated scored financial data with additional fields.
    """
    # Step 1: Map 'COHORT_YEAR' from 'cohort_date_sas' in train_data
    scored_financial_dev["COHORT_YEAR"] = train_data[
        train_data.index.isin(scored_financial_dev.index.to_list())
    ]['cohort_date_sas'].apply(lambda x: str(x)[0:4])  # Extract year part from 'cohort_date_sas'
    
    # Step 2: Group data by 'COHORT_YEAR' and calculate sum and count of 'intodef_gid'
    cohort_stats = scored_financial_dev.groupby("COHORT_YEAR")[y].agg(["sum", "count"])
    cohort_stats.reset_index(inplace=True)  # Reset index for cleaner display
    
    # Step 3: Map 'final_crr' from train_data to scored_financial_dev
    scored_financial_dev['final_crr'] = train_data[
        train_data.index.isin(scored_financial_dev.index.to_list())
    ]['final_crr'].astype(float)
    
    # Step 4: Map 'model_crr' from train_data to scored_financial_dev
    scored_financial_dev['model_crr'] = train_data[
        train_data.index.isin(scored_financial_dev.index.to_list())
    ]['model_crr'].astype(float)
    
    # Step 5: Adjust 'final_crr' values where necessary
    scored_financial_dev["Final_CRR_MOD"] = np.where(
        (scored_financial_dev['final_crr'] - round(scored_financial_dev['final_crr']) == 0) &
        (scored_financial_dev['final_crr'] != 9) &
        (scored_financial_dev['final_crr'] != 10),
        scored_financial_dev['final_crr'] + 0.1,
        scored_financial_dev['final_crr']
    )
    
    # Step 6: Adjust 'model_crr' values where necessary
    scored_financial_dev["Model_CRR_MOD"] = np.where(
        (scored_financial_dev['model_crr'] - round(scored_financial_dev['model_crr']) == 0) &
        (scored_financial_dev['model_crr'] != 9) &
        (scored_financial_dev['model_crr'] != 10),
        scored_financial_dev['model_crr'] + 0.1,
        scored_financial_dev['model_crr']
    )
    
    # Step 7: Calculate Default Rate (DR) for each cohort year
    cohort_stats["DR"] = cohort_stats["sum"] / cohort_stats["count"]
    
    # Step 8: Calculate Loss Rate Adjusted Default Rate (LRADR) as the mean of DR
    lradr = cohort_stats["DR"].mean()
    print("LRADR: {:.2%}".format(lradr))  # Print LRADR as a percentage
    
    return scored_financial_dev


def build_model(feature_vars, train_data, test_data, oot_data, train_df, test_df, oot_df, target_col='intodef_gid'):
    """
    Build and evaluate a model using train, test, and OOT datasets.

    Parameters:
    - feature_vars (list): List of feature variable names to include in the model.
    - train_data (DataFrame): Processed training dataset.
    - test_data (DataFrame): Processed testing dataset.
    - oot_data (DataFrame): Out-of-Time (OOT) dataset for evaluation.
    - train_df (DataFrame): Original training DataFrame for scoring.
    - test_df (DataFrame): Original testing DataFrame for scoring.
    - oot_df (DataFrame): Original OOT DataFrame for scoring.
    - target_col (str): Name of the target variable. Default is 'intodef_gid'.

    Returns:
    - Tuple containing:
      - Model coefficients
      - Summary DataFrame
      - Calibration table
      - Feature weights
      - Train-Test combined KS plot
      - Train-Test combined ROC plot
      - OOT ROC plot
    """

    # Combine target column with features
    selected_features = feature_vars + [target_col]

    # Perform backward elimination to select optimal features
    X_opt, model = backward_elimination(train_data[selected_features], target_column=target_col, sl=0.05)

    # Get model coefficients
    model_coefficients = get_coefs(model)

    # Evaluate AUC-ROC and other metrics on train, test, and OOT datasets
    scored_train, train_auc, train_metrics = calculate_auc_roc(train_data,y=target_col, model_results=model_coefficients)
    scored_test, test_auc, test_metrics = calculate_auc_roc(test_data,y=target_col, model_results=model_coefficients)
    scored_oot, oot_auc, oot_metrics = calculate_auc_roc(oot_data,y=target_col, model_results=model_coefficients)

    # Combine train and test datasets for calibration analysis
    calibration_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
    scored_calibration, calib_auc, calib_metrics = calculate_auc_roc(calibration_data,y=target_col, model_results=model_coefficients)

    # Combine original DataFrames for detailed scoring
    combined_calib_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    # Perform scoring for all datasets
    scored_train_df = score_df(scored_train, train_df, y)
    scored_test_df = score_df(scored_test, test_df, y)
    scored_oot_df = score_df(scored_oot, oot_df, y)
    scored_calib_df = score_df(scored_calibration, combined_calib_df, y)

    # Evaluate performance on each dataset
    train_summary = perf_eva(train_metrics, "Train")
    test_summary = perf_eva(test_metrics, "Test")
    oot_summary = perf_eva(oot_metrics, "OOT")
    train_test_summary = perf_eva(calib_metrics, "Train+Test")

    # Clean and summarize results
    train_summary_df = summary_clean(train_summary, dataset_name="Train")
    test_summary_df = summary_clean(test_summary, dataset_name="Test")
    oot_summary_df = summary_clean(oot_summary, dataset_name="OOT")
    train_test_summary_df = summary_clean(train_test_summary, dataset_name="Train+Test")

    # Combine all summaries into a final summary DataFrame
    final_summary = pd.concat([train_summary_df, test_summary_df, oot_summary_df, train_test_summary_df], axis=0)

    # Calculate feature importance weights
    feature_weights = get_variable_weights(train_data,target_col=target_col ,cols=list(set(X_opt.columns) - set([target_col])))

    return (
        model_coefficients,
        final_summary,
        calib_metrics,
        feature_weights,
        train_test_summary['ks_plt'],
        train_test_summary['roc_plt'],
        oot_summary["roc_plt"]
    )


def dynamic_keys(features, variable_name, train_data, test_data, oot_data, train_df, test_df, oot_df, target_col='intodef_gid'):
    """
    Dynamically generates a dictionary of key-value pairs for storing model outputs.

    Parameters:
    - features (list): List of feature variables for the model.
    - variable_name (str): Name identifier for the variable set.
    - train_data (DataFrame): Processed training data.
    - test_data (DataFrame): Processed testing data.
    - oot_data (DataFrame): Processed out-of-time (OOT) data.
    - train_df (DataFrame): Original training data for scoring.
    - test_df (DataFrame): Original testing data for scoring.
    - oot_df (DataFrame): Original OOT data for scoring.
    - target_col (str): Name of the target variable. Default is 'intodef_gid'.

    Returns:
    - dict_obj (dict): Dictionary containing model results, summary, plots, and weights.
    """
    # Run the model and collect outputs
    result, final_summary, calibration_table, weights, ks_plot, roc_plot, oot_roc_plot = build_model(
        features, train_data, test_data, oot_data, train_df, test_df, oot_df, target_col
    )

    # Create a dictionary to store all outputs
    dict_obj = {}

    # Store results in the dictionary using dynamic keys
    dict_obj[f'result_{variable_name}'] = result
    dict_obj[f'final_summary_{variable_name}'] = final_summary
    dict_obj[f'calibration_table_{variable_name}'] = calibration_table
    dict_obj[f'weights_{variable_name}'] = weights
    dict_obj[f'ks_plot_{variable_name}'] = ks_plot
    dict_obj[f'roc_plot_{variable_name}'] = roc_plot
    dict_obj[f'oot_roc_plot_{variable_name}'] = oot_roc_plot

    return dict_obj

