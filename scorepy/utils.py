import io
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
# from .modelling import build_model, get_variable_weights, scoring, get_coefs


def bins_to_df(bins):
    """
    Converts a dictionary of bins into a DataFrame sorted by Information Value (IV).

    Parameters:
    - bins (dict): Dictionary containing variable information with keys as variable names
      and values as dictionaries that include 'total_iv'.

    Returns:
    - DataFrame: DataFrame with columns 'Variable' and 'IV', sorted by IV in descending order.
    """
    # Extract variables and their IV values
    variables = bins.keys()
    iv_values = [bins[var]['total_iv'][0] for var in variables]

    # Create a DataFrame
    df = pd.DataFrame({'Variable': variables, 'IV': iv_values})
    df = df.sort_values(by='IV', ascending=False).reset_index(drop=True)

    return df


def is_monotonic(series):
    """
    Checks if a series is monotonic (either non-increasing or non-decreasing).

    Parameters:
    - series (pd.Series): The input series to check for monotonicity.

    Returns:
    - bool: True if the series is monotonic, False otherwise.
    """
    return all(series[i] <= series[i + 1] for i in range(len(series) - 1)) or \
           all(series[i] >= series[i + 1] for i in range(len(series) - 1))



class CapturePrint:
    """
    Context manager to capture printed output.

    Attributes:
    - output (str): Captured output as a string.
    """
    def __enter__(self):
        self.stdout = sys.stdout  # Save the original stdout
        self.stringio = io.StringIO()  # Create a string buffer
        sys.stdout = self.stringio  # Redirect stdout to the string buffer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.output = self.stringio.getvalue()  # Retrieve the captured output
        self.stringio.close()  # Close the string buffer
        sys.stdout = self.stdout  # Restore the original stdout


def filter_list(items, substring='perc'):
    """
    Filters a list to include only items containing a specific substring.

    Parameters:
    - items (list): List of strings to filter.
    - substring (str): Substring to search for in each item. Default is 'perc'.

    Returns:
    - list: Filtered list containing only items with the substring.
    """
    return [item for item in items if substring in item]


def generate_pairs(results, variable_name):
    """
    Generates grouped results for reporting or visualization.

    Parameters:
    - results (dict): Dictionary containing model outputs.
    - variable_name (str): Identifier for the variable set.

    Returns:
    - Tuple: Grouped results ([plots], [results, summary, table, weights]).
    """
    # Extract plots and results
    plots = [
        results[f'ks_plot_{variable_name}'],
        results[f'roc_plot_{variable_name}'],
        results[f'oot_roc_plot_{variable_name}']
    ]

    data = [
        results[f'result_{variable_name}'],
        results[f'final_summary_{variable_name}'],
        results[f'calibration_table_{variable_name}'],
        results[f'weights_{variable_name}']
    ]

    return plots, data


def generate_model_strings(num_models=4):
    """
    Generates formatted strings for model outputs.

    Parameters:
    - num_models (int): Number of models to generate strings for. Default is 4.
    """
    for i in range(1, num_models + 1):
        print(f"'Model {i}': generate_pairs(results=result_{i}, variable_name={i}),")
             
   
class Transformation:
    def __init__(self, df):
        """
        Initialize the Transformation class with a copy of the provided DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame to apply transformations.
        """
        self.df = df.copy()
    
    def _check_column_exists(self, col):
        """
        Check if a column exists in the DataFrame.
        
        Args:
            col (str): Column name to check.
        
        Raises:
            ValueError: If the column does not exist in the DataFrame.
        """
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
    
    def normalize_standardize(self, suffix='_normalize'):
        """
        Standardize numerical features (Z-score normalization) and add a suffix.
        
        Args:
            suffix (str): Suffix to append to column names after transformation.
        """
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        self.df[[col + suffix for col in numeric_cols]] = scaler.fit_transform(self.df[numeric_cols])
    
    def min_max_scaling(self, suffix='_minmax'):
        """
        Apply Min-Max scaling to numerical features and add a suffix.
        
        Args:
            suffix (str): Suffix to append to column names after transformation.
        """
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        scaler = MinMaxScaler()
        self.df[[col + suffix for col in numeric_cols]] = scaler.fit_transform(self.df[numeric_cols])
    
    def log_transformation(self, cols, suffix='_log'):
        """
        Apply log transformation to specified columns and add a suffix.
        
        Args:
            cols (list): List of column names to apply log transformation.
            suffix (str): Suffix to append to column names after transformation.
        """
        for col in cols:
            self._check_column_exists(col)
            self.df[col + suffix] = np.log(self.df[col] + 1)  # Add 1 to avoid log(0)
    
    def percentile_transformation(self, col, n=10, suffix='_perc'):
        """
        Transform a column into percentiles (e.g., deciles) and add a suffix.
        
        Args:
            col (str): Column to transform.
            n (int): Number of quantiles (default: 10).
            suffix (str): Suffix to append to column name after transformation.
        """
        self._check_column_exists(col)
        self.df[col + suffix] = pd.qcut(self.df[col], q=n, labels=False, duplicates="drop") + 1
    
    def ratio_terms(self, col1, col2, suffix='_ratio'):
        """
        Create interaction terms between two columns and add a suffix.
        
        Args:
            col1 (str): First column name.
            col2 (str): Second column name.
            suffix (str): Suffix to append to the new column name.
        """
        self._check_column_exists(col1)
        self._check_column_exists(col2)
        self.df[f'{col1}_{col2}{suffix}'] = self.df[col1] / self.df[col2]
    
    def normalize_financial_ratios(self, ratio_cols, industry_avg, suffix='_mill'):
        """
        Normalize financial ratios relative to industry averages and add a suffix.
        
        Args:
            ratio_cols (list): List of columns containing financial ratios.
            industry_avg (dict): Dictionary of industry averages for each column.
            suffix (str): Suffix to append to column names after transformation.
        """
        for col in ratio_cols:
            self._check_column_exists(col)
            self.df[col + suffix] = self.df[col] / industry_avg.get(col, 1)  # Avoid division by 0
    
    def winsorize(self, cols, lower_quantile, upper_quantile, suffix='_winsor'):
        """
        Winsorize columns to handle outliers by clipping values at specified quantiles.
    
        Args:
            cols (list): List of columns to winsorize.
            lower_quantile (float): Lower quantile threshold for winsorization (e.g., 0.05 for 5%).
            upper_quantile (float): Upper quantile threshold for winsorization (e.g., 0.95 for 95%).
            suffix (str): Suffix to append to column names after transformation.
        """
        for col in cols:
            self._check_column_exists(col)
            lower_bound = self.df[col].quantile(lower_quantile)
            upper_bound = self.df[col].quantile(upper_quantile)
            self.df[col + suffix] = np.clip(self.df[col], lower_bound, upper_bound)
        
    def differencing(self, col, suffix='_diff'):
        """
        Compute the difference between successive periods and add a suffix.
        
        Args:
            col (str): Column name to apply differencing.
            suffix (str): Suffix to append to column name after transformation.
        """
        self._check_column_exists(col)
        self.df[col + suffix] = self.df[col].diff().fillna(0)
    
    def encode_categorical(self, cols, suffix='_encoded'):
        """
        One-hot encode categorical variables and add a suffix.
        
        Args:
            cols (list): List of categorical columns to encode.
            suffix (str): Suffix to append to encoded column names.
        """
        encoder = OneHotEncoder(drop='first', sparse=False)
        for col in cols:
            self._check_column_exists(col)
            encoded = encoder.fit_transform(self.df[[col]])
            encoded_df = pd.DataFrame(
                encoded, 
                columns=[f'{col}{suffix}_{val}' for val in encoder.categories_[0][1:]], 
                index=self.df.index
            )
            self.df = pd.concat([self.df, encoded_df], axis=1).drop(columns=[col])
    
    def floor(self, col, min_value, suffix='_floor'):
        """
        Apply floor transformation to specified columns and add a suffix.
        
        Args:
            col (str): Column name to apply floor transformation.
            min_value (float): Minimum value to enforce.
            suffix (str): Suffix to append to column name after transformation.
        """
        self._check_column_exists(col)
        self.df[col + suffix] = np.maximum(self.df[col], min_value)
    
    def cap(self, col, max_value, suffix='_cap'):
        """
        Apply cap transformation to specified columns and add a suffix.
        
        Args:
            col (str): Column name to apply cap transformation.
            max_value (float): Maximum value to enforce.
            suffix (str): Suffix to append to column name after transformation.
        """
        self._check_column_exists(col)
        self.df[col + suffix] = np.minimum(self.df[col], max_value)
    
    def apply_transformations(
        self, normalize=False, min_max=False, log_cols=[], 
        percentile_cols={}, ratio_terms_params=[], ratio_cols=[], 
        industry_avg={}, lower_quantile=None, upper_quantile=None,  winsorize_cols=[], 
        differencing_cols=[], categorical_cols=[], floor_params={}, cap_params={} ):
        """
        Apply all transformations to the DataFrame based on specified parameters.

        Args:
            normalize (bool): Whether to apply standard normalization.
            min_max (bool): Whether to apply min-max scaling.
            log_cols (list): List of columns for log transformation.
            percentile_cols (dict): Dictionary {column: number of quantiles} for percentile transformation.
            ratio_terms_params (list): List of tuples [(col1, col2)] for ratio term creation.
            ratio_cols (list): List of ratio columns for normalization.
            industry_avg (dict): Dictionary of industry averages for ratio normalization.
            lower_quantile (float): Lower quantile threshold for winsorization.
            upper_quantile (float): Upper quantile threshold for winsorization.
            winsorize_cols (list): List of columns to winsorize.
            differencing_cols (list): List of columns to compute differencing.
            categorical_cols (list): List of categorical columns for one-hot encoding.
            floor_params (dict): Dictionary {column: min_value} for floor transformation.
            cap_params (dict): Dictionary {column: max_value} for cap transformation.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        if normalize:
            self.normalize_standardize()
        if min_max:
            self.min_max_scaling()
        if log_cols:
            self.log_transformation(log_cols)
        for col, n in percentile_cols.items():
            self.percentile_transformation(col, n)
        for col1, col2 in ratio_terms_params:
            self.ratio_terms(col1, col2)
        if ratio_cols and industry_avg:
            self.normalize_financial_ratios(ratio_cols, industry_avg)
        if lower_quantile is not None and upper_quantile is not None and winsorize_cols:
            self.winsorize(winsorize_cols, lower_quantile, upper_quantile)
        for col in differencing_cols:
            self.differencing(col)
        if categorical_cols:
            self.encode_categorical(categorical_cols)
        for col, min_value in floor_params.items():
            self.floor(col, min_value)
        for col, max_value in cap_params.items():
            self.cap(col, max_value)
        
        return self.df



