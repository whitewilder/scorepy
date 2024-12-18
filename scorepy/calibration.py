import matplotlib.pyplot as plt
import seaborn as sns


class Calibration:
    """
    This class performs True Default-Based Calibration.

    Attributes:
        input_df (pd.DataFrame): Input scored calibration dataset.
        master_scale (pd.DataFrame): Master scale dataset with CRR brackets and PD ranges.
        lradr (float): Central tendency of the model.
        moc (float): Margin of Conservatism adjustment.
        year_field (str): Column name for the year in the calibration data.
        score_field (str): Column name for model scores in the calibration data.
        target_field (str): Column name for the target variable.
        final_crr_field (str): Column name for the final CRR in the calibration data.
    """







    def __init__(self, input_df, master_scale, lradr, moc, year_field, score_field, target_field, final_crr_field):
        self.input_df = input_df
        self.master_scale = master_scale
        self.lradr = lradr
        self.moc = moc
        self.year_field = year_field
        self.score_field = score_field
        self.target_field = target_field
        self.final_crr_field = final_crr_field
        self.summary, self.params, self.delta = self._calibrate()

    def _calibrate(self):
        """
        Perform the calibration process and return outputs.

        Returns:
            tuple: Calibrated dataset, model summary, model parameters, and shift delta.
        """
        
        
        if self.final_crr_field not in self.input_df.columns:
            print("## INCORRECT FINAL CRR field")
        else:
            self.input_df["FINAL_CRR_ORIG"] = self.input_df[self.final_crr_field]

            # Step 1: First-stage model fitting
            X = self.input_df[[self.score_field]].copy()
            y = self.input_df[self.target_field]
            X = sm.add_constant(X)
            step1_model = sm.Probit(y, X).fit()

            # Step 2: Generate first step scores
            X["first_step_score"] = sum(
                step1_model.params[v] * X[v] for v in step1_model.params.index
            )

            # Linear shift to align with LRADR estimate
            delta = self._linear_shift(X, self.lradr, self.year_field)
            X["Model_PD"] = norm.cdf(X["first_step_score"] + delta)
            X["Model_PD"] = X["Model_PD"].clip(upper=1, lower=1e-4)

            # Map Model_PD to CRR using master scale
            X["pre_moc_predicted_crr"] = X["Model_PD"].apply(
                lambda x: self.master_scale.CRR[bisect.bisect(self.master_scale.Min, x) - 1]
            )

            # Apply Margin of Conservatism (MoC)
            X["post_moc_model_PD"] = X["Model_PD"] * self.moc
            X["post_moc_predicted_crr"] = X["post_moc_model_PD"].apply(
                lambda x: self.master_scale.CRR[bisect.bisect(self.master_scale.Min, x) - 1]
            )

            self.calibrated_df = X

            return X, step1_model.summary(), step1_model.params, delta

    def _average_pd(self, delta, X, lradr, year):
        """
        Compute the average PD after applying delta.

        Args:
            delta (float): The shift value to align with LRADR.
            X (pd.DataFrame): Input data.
            lradr (float): LRADR target.
            year (str): Year column name.

        Returns:
            float: Absolute difference between computed and target LRADR.
        """
        Y = X.copy()
        Y["PD"] = norm.cdf(Y["first_step_score"] + delta)
        avg_pd = Y[["PD", year]].groupby(by=year).mean().rename(columns={"PD": "Avg_PD"}).Avg_PD.mean()
        return abs(avg_pd - lradr)

    def _linear_shift(self, X, lradr, year):
        """
        Perform linear shift calculation to align with LRADR.

        Args:
            X (pd.DataFrame): Input data.
            lradr (float): LRADR target.
            year (str): Year column name.

        Returns:
            float: Calculated delta shift.
        """
        np.random.seed(123)
        result = optimize.root(lambda x: self._average_pd(x[0], X, lradr, year), [0.1])
        if result.success:
            return result.x[0]
        else:
            print("Shift calculation failed")
            return np.nan

    def get_output_data(self):
        """
        Retrieve the calibrated dataset.

        Returns:
            pd.DataFrame: Calibrated dataset.
        """
        return self.calibrated_df

    def get_summary(self):
        """
        Retrieve the model summary.

        Returns:
            str: Model summary.
        """
        return self.summary

    def get_params(self):
        """
        Retrieve model parameters.

        Returns:
            pd.Series: Model parameters.
        """
        return self.params

    def get_shift(self):
        """
        Retrieve the delta shift value.

        Returns:
            float: Delta shift.
        """
        return self.delta

    def get_distribution_plot(self, segment_field=None, num_columns=1):
        """
        Plot the CRR distribution.

        Args:
            segment_field (str, optional): Field for segmentation.
            num_columns (int, optional): Number of columns in the plot grid.
        """

        data = self.calibrated_df.copy()

        if segment_field:
            unique_segments = data[segment_field].dropna().unique()
            num_rows = -(-len(unique_segments) // num_columns)  # Ceiling division for rows
            fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 6, num_rows * 4), squeeze=False)

            for idx, segment in enumerate(unique_segments):
                row, col = divmod(idx, num_columns)
                segment_data = data[data[segment_field] == segment]
                sns.histplot(segment_data["post_moc_predicted_crr"], kde=True, ax=axes[row][col])
                axes[row][col].set_title(f"CRR Distribution - {segment}")
                axes[row][col].set_xlabel("CRR")
                axes[row][col].set_ylabel("Frequency")

            # Hide unused subplots
            for idx in range(len(unique_segments), num_rows * num_columns):
                row, col = divmod(idx, num_columns)
                axes[row][col].axis("off")

            plt.tight_layout()
        else:
            plt.figure(figsize=(8, 6))
            sns.histplot(data["post_moc_predicted_crr"], kde=True)
            plt.title("CRR Distribution")
            plt.xlabel("CRR")
            plt.ylabel("Frequency")

        plt.show()

    def get_backtesting_plot_across_years(self, segment_field=None, segment=None):
        """
        Generate backtesting plot across years.

        Args:
            segment_field (str, optional): Field for segmentation.
            segment (str, optional): Specific segment value.
        """
        # Implementation here
        pass

    def get_backtesting_plot_by_grade(self):
        """
        Generate backtesting plot by grade.
        """
        # Implementation here
        pass

    def get_scatter_plot(self):
        """
        Generate scatter plot for visualizing calibration results.
        """
        # Implementation here
        pass
