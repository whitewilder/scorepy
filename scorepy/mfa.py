import pandas as pd
import numpy as np

def corr_drop(df, target, threshold=0.9, how='max_corr', order_list=None):
    """
    Combines two approaches for removing highly correlated variables from a dataset while preserving the target variable.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Name of the target variable (not to be dropped or used in correlation).
        threshold (float): Correlation threshold to consider variables as highly correlated. Default is 0.9.
        how (str): Method to select variables for removal:
                   - 'max_corr': Drops the variable with the highest correlation count.
                   - 'order_list': Drops variables based on their order in `order_list`.
        order_list (list, optional): List specifying the order of importance for variables. 
                                     Used when `how` is set to 'order_list'.

    Returns:
        pd.DataFrame: DataFrame with highly correlated variables removed, excluding the target variable.
    """
    if target not in df.columns:
        raise ValueError(f"The target variable '{target}' is not in the DataFrame.")
    # Create a copy of the input DataFrame to work on
    df_copy = df.copy()
    
    # Remove the target variable from the correlation analysis
    df_matrix = df_copy.drop(columns=[target], errors='ignore')

    
    if order_list is None:
        order_list = df_matrix.columns.tolist()
 


    # Variables to evaluate
    variables = df_matrix.columns.tolist()
    
    if how == 'max_corr':
        to_drop=None

        while True:
            # Calculate the correlation matrix
            corr_matrix = df_matrix.corr()
    
            # Count correlations above the threshold for each variable
            high_corr_counts = (((corr_matrix.abs() > threshold) & ((corr_matrix.abs() < 1) | (corr_matrix == -1)))).sum(axis=1)
    
            # Print the correlation matrix for debugging purposes
            #  print("\nCurrent Correlation Matrix:\n", corr_matrix.abs())
    
            # Identify variables with high correlation
            high_corr_vars = high_corr_counts[high_corr_counts > 0]
            
    
            if high_corr_vars.empty:
                # Exit loop if no variables exceed the correlation threshold
                break    
            
                # Remove the variable with the highest correlation count
            else:
                var_to_remove = high_corr_vars.idxmax()
                corr_count = high_corr_vars.max()
                print(f"Variable '{var_to_remove}' removed due to high correlation (> {threshold}) "
                      f"with {corr_count} other variable(s).")
                # Drop the variable
                variables.remove(var_to_remove)
                df_matrix = df_matrix[variables]



    elif how == 'order_list':
        if order_list is None:
            raise ValueError("'order_list' must be provided when using 'how=\'order_list\'.")
        # Identify correlated pairs
        corr_matrix = df_matrix.corr().abs()
        # Create an upper triangle matrix to avoid duplicate pairs
        upper_triangle = corr_matrix.where(
            ~pd.np.tril(pd.np.ones(corr_matrix.shape)).astype(bool)
        )
        
        # Find pairs of variables with correlation above the threshold
        correlated_pairs = [
            (col, index, value)
            for index, row in upper_triangle.iterrows()
            for col, value in row.items()
            if value > threshold
        ]
        
        # Set to track variables to drop
        to_drop = set()
        
        for col1, col2, corr_value in correlated_pairs:
            # Check which variable to drop based on order
            if order_list.index(col1) < order_list.index(col2):
                print(f"Highly correlated pair: {col1} and {col2} (correlation: {corr_value}). Dropping: {col2}")
                to_drop.add(col2)
            else:
                print(f"Highly correlated pair: {col1} and {col2} (correlation: {corr_value}). Dropping: {col1}")
                to_drop.add(col1)
    else:
        raise ValueError("Invalid value for 'how'. Use 'max_corr' or 'order_list'.")

        # Drop the selected variable
    
    if to_drop:
        final_df=df_copy.drop(columns=list(to_drop))
    else:  
       # Add the target column back to the DataFrame
        final_df = pd.concat([df_matrix, df_copy[target]], axis=1, ignore_index=False)

    return final_df



def multicollinearity_check(df, y, threshold=18, only_final_vif=True):
    """
    Checks for multicollinearity using Generalized Variance Inflation Factor (GVIF).
    Categorical columns are one-hot encoded automatically.

    Args:
        df (pd.DataFrame): Input DataFrame.
        y (str): Name of the dependent variable/target column.
        threshold (float): VIF threshold for filtering out columns. Default is 18.
        only_final_vif (bool): If True, return only the final VIF values. Default is True.

    Returns:
        tuple: 
            - A DataFrame containing GVIF and VIF values for all factors.
            - Filtered DataFrame with multicollinear columns removed.
    """
    # Drop the target variable to isolate predictors
    df_x = df.drop(y, axis=1)

    # Identify categorical columns and one-hot encode them
    onehot_list = list(df_x.select_dtypes(include=['category', 'object', 'string']).columns)
    df_1hot = pd.get_dummies(df_x, drop_first=True, dummy_na=False, prefix_sep='_')

    # Create an empty DataFrame to store GVIF results
    gvif_df = pd.DataFrame(columns=['factor', 'GVIF', 'Df', 'GVIF^(1/2DF)', 'VIF'])

    # Iterate through columns and compute GVIF
    for column_name in df_x.columns:
        if column_name in onehot_list:
            # Select all dummy-encoded columns for the categorical variable
            X1 = df_1hot.loc[:, df_1hot.columns.str.startswith(column_name)]
            X2 = df_1hot.loc[:, ~df_1hot.columns.str.startswith(column_name)]
        else:
            # For numeric variables
            X1 = df_1hot[[column_name]].values
            X2 = df_1hot.loc[:, df_1hot.columns != column_name].values

            

        # Calculate GVIF
        det_X1 = np.linalg.det(np.array(np.corrcoef(X1, rowvar=False), ndmin=2))
        det_X2 = np.linalg.det(np.corrcoef(X2, rowvar=False))
        det_combined = np.linalg.det(np.corrcoef(np.append(X1,X2,axis=1), rowvar=False))
        #         print((np.array(np.corrcoef(X1, rowvar=False), ndmin=2)[0]))
        #         print((np.corrcoef(X2, rowvar=False)[0][0]))
        # #         print(det_X2)

        gvif =  det_X1 * det_X2 /det_combined
        gvif_12df = np.power(gvif, 1 / (2 * X1.shape[1]))
        gvif_12df_sq = gvif_12df ** 2
        df_ = X1.shape[1]

        # Update results DataFrame
        new_row = {
            'factor': column_name,
            'GVIF': gvif,
            'Df': df_,
            'GVIF^(1/2DF)': gvif_12df,
            'VIF': gvif_12df_sq
        }
        gvif_df = pd.concat([gvif_df, pd.DataFrame([new_row])], ignore_index=True)

    gvif_df.set_index('factor', inplace=True)

    # Return final GVIF table or only VIF table
    if only_final_vif:
        gvif_df_final = gvif_df.drop(columns=['GVIF', 'Df', 'GVIF^(1/2DF)'])
    else:
        gvif_df_final = gvif_df

    # Filter columns exceeding threshold
    gvif_filter = gvif_df.loc[gvif_df['VIF'] > threshold]['VIF'].to_dict()
    if gvif_filter:
        for col in gvif_filter.keys():
            df_x = df_x.drop(columns=[col])

    # Return updated DataFrame
    df_m = pd.concat([df_x, df[y]], axis=1)
    return gvif_df_final, gvif_filter, df_m


def find_columns_with_high_duplicates(df, threshold=0.5):
    """
    Finds columns in a DataFrame with a high percentage of duplicate values.
  
    Args:
      df: The DataFrame to analyze.
      threshold: The minimum percentage of duplicate values for a column to be considered.
  
    Returns:
      A list of column names with a high percentage of duplicates.
    """
  
    columns_with_high_duplicates = []
    for col in df.columns:
      num_rows = len(df)
      num_unique_values = len(df[col].unique())
      duplicate_ratio = 1 - (num_unique_values / num_rows)
      if duplicate_ratio >= threshold:
        columns_with_high_duplicates.append(col)
    return columns_with_high_duplicates


def remove_high_vif_vars(df, vif_threshold=7, corr_threshold=0.9,dup_threshold=.7, target_col='intodef', how='order_list', order_list=None):
    """
    Remove variables with high Variance Inflation Factor (VIF) from a DataFrame to address multicollinearity.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing independent variables and the target column.
    - vif_threshold (float): The VIF threshold above which variables will be removed. Default is 7.
    - dup_threshold (float): The percentage of duplicate values for a column to be removed (0,1).
    - corr_threshold (float): Correlation threshold for removing variables if VIF calculations encounter issues. Default is 0.9.
    - target_col (str): The name of the target column to exclude from VIF calculations. Default is 'intodef'.
    - how (str): Method for handling correlations in case of issues. Default is 'order_list'.
    - order_list (list): List of variable preferences for correlation-based removal. Optional.

    Returns:
    - df_clean (pd.DataFrame): DataFrame after removing variables with high VIF.
    - final_vif (pd.DataFrame): Final VIF values for the remaining variables.
    """
    # Make a copy of the input DataFrame to avoid altering the original data
    df_clean = df.copy()
    variables = df_clean.columns.tolist()

    while True:
        # Stop if fewer than 3 columns remain
        if df_clean.shape[1] <= 3:
            print("DataFrame has fewer than 3 columns, stopping the process.")
            final_vif = df_clean.corr()
            break


        # print(vif_data)

        # Handle NaN or infinite VIF values
        # if vif_data['VIF'].isna().any() or np.isinf(vif_data['VIF']).any():
        if corr_threshold > 0.9:
            print("High correlation threshold detected. Adjusting corr_threshold to 0.7 to break the loop.")
            corr_threshold = 0.9
        duplist=find_columns_with_high_duplicates(df_clean.drop([target_col],axis=1), threshold=dup_threshold)
        print("Columns with high number of duplicates are dropped:", duplist)
        df_clean=df_clean.drop(duplist,axis=1)
        print("Encountered NaN or infinite VIF values. Adjusted columns based on correlation.")
        df_clean = corr_drop(df_clean, target=target_col, threshold=corr_threshold, order_list=order_list, how=how)
        variables = df_clean.columns.tolist()
        # continue


        # Calculate VIF for current variables
        vif_data, _, _ = multicollinearity_check(df_clean, y=target_col, threshold=vif_threshold, only_final_vif=True)
        vif_data["Variable"] = vif_data.index
        vif_data = vif_data.reset_index(drop=True)
     # Identify variables exceeding the VIF threshold
    
        high_vif_vars = vif_data[vif_data["VIF"] > vif_threshold]


        # Exit loop if no variables exceed the threshold
        if high_vif_vars.empty:
            break

        # Identify the variable with the highest VIF and remove it
        var_to_remove = high_vif_vars.sort_values(by="VIF", ascending=False).iloc[0]["Variable"]
        vif_to_remove = high_vif_vars.sort_values(by="VIF", ascending=False).iloc[0]["VIF"]

        if var_to_remove in variables:
            variables.remove(var_to_remove)
            df_clean = df_clean[variables]
            print(f"Removed variable: {var_to_remove} with VIF {vif_to_remove:.2f}")
        else:
            break
            
       

    # Identify and print removed variables
    removed_vars = set(df.columns) - set(variables) - {target_col}
    if removed_vars:
        print(f"Removed variables: {removed_vars}")

    # Calculate final VIF values for remaining variables
    if df_clean.shape[1] <= 3:
        final_vif = df_clean.corr()
        print("DataFrame has fewer than 3 columns, stopping the process.")
    else:
        final_vif, _, _ = multicollinearity_check(df_clean, y=target_col, threshold=vif_threshold, only_final_vif=True)
        final_vif["Variable"] = final_vif.index

    return df_clean, final_vif.reset_index(drop=True)


# Write demo function
