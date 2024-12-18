import pandas as pd
import numpy as np
import statsmodels.api as sm
def df_AR_AUC(df, Score="Score", target="target", ascending=False, method="GRADE", k=10):
    """
    Function to calculate Accuracy Ratio of CAP curve and Area Under the ROC curve.

    Inputs:
        - df: dataframe containing the Scores and default status
        - Score: field name containing the score (string)
        - target: field name containing the default status (string). Should contain 0s and 1s.
        - ascending: False signifies higher scores are associated with defaults (bool)
        - method: one of ["GRADE", "CONCORDENT"]
            - GRADE approximates AR by assuming a suitable number of bins.
            - CONCORDENT calculates AR by taking bins = unique scores.
        - k: Number of bins (used if `method="GRADE"`).

    Outputs:
        - aux02: dataframe created to be used for plotting ROC or CAP curve.
        - out: dictionary containing KS statistic, GINI, AUROC, and AR values.
    """

    possible_methods = ["GRADE", "CONCORDENT"]
    if method not in possible_methods:
        print(f"Execution Stopped. \nPlease set a method as one of the two: {possible_methods[0]} or {possible_methods[1]}")
        return np.nan, np.nan

    # Sorting and ranking
    if method == "CONCORDENT":
        k = len(df[Score].value_counts())
        df = df.sort_values(by=Score, ascending=ascending)
        df['rank'] = df[Score].rank(method="min", ascending=ascending)
        df['groups'] = df['rank']
    else:
        print("Calculating AUC assuming grades.")
        df = df.sort_values(by=Score, ascending=ascending)
        df['rank'] = df[Score].rank(method="min", ascending=ascending)
        df['groups'] = pd.cut(df['rank'], bins=k, labels=False) + 1

    df['antitarget'] = 1 - df[target]

    # Aggregations
    aux01 = pd.concat([
        df.groupby('groups')['antitarget'].sum().rename("non_def_count"),
        df.groupby('groups')[target].sum().rename("def_count"),
        df.groupby('groups')[Score].max().rename("max_score"),
        df.groupby('groups')[Score].min().rename("min_score"),
    ], axis=1).reset_index()

    aux01 = aux01.fillna(0)
    aux01['tot_count'] = aux01['non_def_count'] + aux01['def_count']

    total_nondefs = df['antitarget'].sum()
    total_defs = df[target].sum()
    total_count = aux01['tot_count'].sum()

    # Cumulative sums
    aux02 = pd.concat([
        aux01,
        aux01[['non_def_count', 'def_count', 'tot_count']].cumsum().rename(columns={
            "non_def_count": "non_def_count_cm",
            "def_count": "def_count_cm",
            "tot_count": "tot_count_cm",})
        ], axis=1)

    aux02['fraction_defs'] = aux02['def_count_cm'] / total_defs
    aux02['fraction_non_defs'] = aux02['non_def_count_cm'] / total_nondefs
    aux02['fraction_obligors'] = aux02['tot_count_cm'] / total_count
    aux02['diff'] = aux02['fraction_defs']- aux02['fraction_non_defs'] 
    

    k_bar = 0.5 * (aux02['fraction_defs'].shift(1) + aux02['fraction_defs']) * \
            (aux02['fraction_obligors'].shift(1) - aux02['fraction_obligors'])
    AR = k_bar.sum()
    AUROC = (AR + 1) / 2
    GINI = AR / (1 - 0.5)
    KS = aux02['fraction_defs'] - aux02['fraction_non_defs'].max()

    out = {
        "KS": KS,
        "GINI": GINI,
        "AUROC": AUROC,
        "AR": AR
    }

    return aux02, out



def calculate_auc(df, feature, target="intodef", ascending=True):
    """
    Calculate the Area Under the Curve (AUC) for a given feature and target variable.
    It uses a concordance-based method to measure the discriminatory power of the feature.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the feature and target variable.
    - feature (str): The column name of the feature to calculate AUC for.
    - target (str): The target variable column name. Default is 'intodef'.
    - ascending (bool): If True, sorts feature values in ascending order for AUC calculation. Default is True.

    Returns:
    - training_auc (float): The calculated AUC value for the feature and target.
    """
    # Drop missing values for the feature and target columns
    data = df[[feature, target]].dropna(axis=0)

    # Calculate AUC using the specified method
    training_auc = df_AR_AUC(df, Score="Score", target="target", ascending=False, method="CONCORDENT")[1]["AUROC"]  # Assuming the method returns AUC in the first index
    
    # Debugging Output: Log if AUC calculation fails for specific cases
    if training_auc < 0.50:  # Check if the AUC is less than the baseline
        df_AR_AUC(df, Score="Score", target="target", ascending=not ascending , method="CONCORDENT")[1]["AUROC"]

    return training_auc


def scoring(data, model_result_df, target, score_column_name):
    """
    Generate scores based on the model results.

    Inputs:
        - data: dataframe to score.
        - model_result_df: output of the model fitting (e.g., MFA).
        - target: target variable (column name).
        - score_column_name: name of the new score column.

    Outputs:
        - scored_dataset: dataframe containing the original data and the scores.
    """
    if "const" not in list(data.columns):
        data["const"] = [1] * data.shape[0]

    variable_list = list(model_result_df.variable)
    scored_dataset = data[data.columns]
    scored_dataset = sm.add_constant(scored_dataset, prepend=True)
    scored_dataset[score_column_name] = 0

    for v in variable_list:
        estimate = list(model_result_df.loc[model_result_df.variable == v, "estimate"])[0]
        scored_dataset[score_column_name] += estimate * scored_dataset[v]

    scored_dataset = scored_dataset.drop("const", axis=1)
    return scored_dataset



def calculate_auc_roc(train_data, model_results, y="intodef_gid"):
    """
    Calculate and return the AUC ROC score and Gini coefficient for model evaluation.
    
    Args:
        train_data (pd.DataFrame): Training dataset containing target and score columns.
        model_results (pd.DataFrame): Model results containing predicted scores.
    
    Returns:
        tuple: A tuple containing:
            - scored_data (pd.DataFrame): Training dataset with scores.
            - training_auc (float): Area Under ROC Curve (AUROC) score.
            - grade_auc (float): AUROC score based on GRADE method.
    """
    # Add the financial scores to the training dataset
    scored_data = scoring(train_data, model_results, y, "FINSCORE")
    
    # Calculate the AUROC using the 'CONCORDANT' method
    auc_concordant = df_AR_AUC( scored_data, Score="FINSCORE",  target=y, ascending=False,  method="CONCORDENT"    )
    
    # Calculate the AUROC using the 'GRADE' method with top 10 buckets
    auc_grade = df_AR_AUC(
        scored_data,Score="FINSCORE", target=y,   ascending=False,   method="GRADE",  k=10 )
    
    # Extract the training AUROC from the concordant calculation
    training_auc = auc_concordant[1]["AUROC"]
    
    # Extract the AUROC score from the GRADE method
    grade_auc = auc_grade[0]
    
    return scored_data, training_auc, grade_auc



