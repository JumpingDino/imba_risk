import numpy as np
import pandas as pd
from IPython.display import display

class SISOAnalysis:
    def __init__(self, input_df, option='equal_volume'):
        """
        Initializes the SISOAnalysis class with the given input dataframe and option.

        Args:
            input_df (pd.DataFrame): The input dataframe containing the necessary columns.
             - primary_key: The primary key of the record.
             - y_true: The true target variable.
            - champ_prob: The probability of the champion model.
            - champ_rat: The rating of the champion model.
            - chall_prob: The probability of the challenger model.

            It's not expected to have the chall_rat column, because it will be calculated 
            through processing time based on the option selected.
            option (str): The option for building the aggregate tables. Default is 'equal_volume'.

        Raises:
            AssertionError: If the input dataframe does not contain the necessary columns.
        """
        self.option=option
        
        #grant the input_df contains all the necessary columns
        necessary_cols = ['primary_key', 'y_true', 'champ_prob', 'champ_rat', 'chall_prob']
        assert all(col in input_df.columns for col in necessary_cols), f"Missing columns in input_df: {set(necessary_cols) - set(input_df.columns)}"

        self.input_df = input_df.copy()
        self.input_df['index'] = 1
        self.input_df['index'] = self.input_df['index'].cumsum()

        self.champion_volume = self.input_df['champ_rat'].value_counts().sort_index()
        self.champion_threshold = self.champion_volume.cumsum()
        self.champion_agg = self._build_agg_champ()

        self.challenger_agg = self._build_agg_chall()
        self.challenger_volume = self.input_df['chall_rat'].value_counts().sort_index()
        self.challenger_threshold = self.challenger_volume.cumsum()
        

    def _build_agg_chall(self) -> pd.DataFrame:
        """
        Builds the aggregate table for the challenger model.

        Returns:
            pd.DataFrame: A dataframe containing the aggregate statistics for the challenger model.
            - count_pk: The count of primary keys in each challenger rating group.
            - count_default: The count of defaults in each challenger rating group.
            - default_rate: The default rate in each challenger rating group.
            - cumsum_count_pk: The cumulative sum of primary keys up to each challenger rating group.
            - cumsum_count_default: The cumulative sum of defaults up to each challenger rating group.
            - cumsum_default_rate: The cumulative default rate up to each challenger rating group.
        """
        if self.option == 'equal_volume':
            _tmp = self.input_df[['primary_key', 'index', 'chall_prob', "champ_rat", "y_true"]].copy()
            _tmp.sort_values(by='chall_prob', ascending=True, inplace=True)
            _tmp['chall_rat'] = pd.cut(_tmp['index'],
                                    bins=[0] + list(self.champion_threshold.values), 
                                    labels=self.champion_threshold.index)

        if 'chall_rat' not in self.input_df.columns:
            print("Adding challenger rating to input_df")
            self.input_df = self.input_df.merge(
                _tmp[['primary_key', 'chall_rat']], on='primary_key', how='left'
            )

        challenger_agg = _tmp.groupby(['chall_rat'], observed=False).agg(
            count_pk=('primary_key', 'count'), 
            count_default=('y_true', 'sum'),
            default_rate = ('y_true', 'mean')
            ).sort_index()

        challenger_agg['cumsum_count_pk'] = challenger_agg['count_pk'].cumsum()
        challenger_agg['cumsum_count_default'] = challenger_agg['count_default'].cumsum()
        challenger_agg['cumsum_default_rate'] = challenger_agg['cumsum_count_default']/challenger_agg['cumsum_count_pk']
        return challenger_agg


    def _build_agg_champ(self) -> pd.DataFrame:
        """
        Builds the aggregate table for the champion model."
        """
        champion_agg = self.input_df.groupby(['champ_rat']).agg(
            count_pk=('primary_key', 'count'), 
            count_default=('y_true', 'sum'),
            default_rate = ('y_true', 'mean')
            ).sort_index()
        
        champion_agg['cumsum_count_pk'] = champion_agg['count_pk'].cumsum()
        champion_agg['cumsum_count_default'] = champion_agg['count_default'].cumsum()
        champion_agg['cumsum_default_rate'] = champion_agg['cumsum_count_default']/champion_agg['cumsum_count_pk']
        return champion_agg
    
    def build_agg_tables(self):
        return {'challenger': self.challenger_agg, 'champion': self.champion_agg}
    
    def build_siso_volume(self):
        siso_volume = pd.crosstab(self.input_df['champ_rat'], 
                                  self.input_df['chall_rat'],
                                  margins=True)
        return siso_volume
    
    def build_siso_risk(self):
        siso_risk = pd.pivot_table(self.input_df, 
                                   values='y_true', 
                                   index='champ_rat', 
                                   columns='chall_rat', 
                                   aggfunc='mean', 
                                   observed=False)
        return siso_risk
    

if __name__ == "__main__":

    # Create the sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    pk = [f'customer-{str(i)}' for i in range(1,251)]
    # Create the champion_pred_prob array with random probabilities
    champion_pred_prob = np.random.uniform(0, 1, 250)
    champion_rating = pd.cut(champion_pred_prob, bins=4, labels=["A", "B", "C", "D"])

    challenger_pred_prob = sigmoid(np.linspace(-6, 6, 250)) + np.random.uniform(-0.2, 0.2, 250)
    challenger_pred_prob = np.clip(challenger_pred_prob, 0, 1)  # Ensure probabilities are between 0 and 1

    # Generate a linear space and apply the sigmoid function
    x = np.linspace(-6, 6, 250)
    y_true = np.round(sigmoid(x)).astype(int)

    #generate the input_df
    input_df = pd.DataFrame(
            [
                pk,
                y_true,
                champion_pred_prob,
                champion_rating,
                challenger_pred_prob,
            ],
            index = ['primary_key', 'y_true', 'champ_prob', 'champ_rat', 'chall_prob']
        ).T

    # Create an instance of SISOAnalysis
    siso_analysis = SISOAnalysis(input_df)

    # Build aggregate tables
    agg_tables = siso_analysis.build_agg_tables()
    print("Aggregate Tables:")
    display(agg_tables['challenger'])
    display(agg_tables['champion'])	

    # Build SISO volume table
    siso_volume = siso_analysis.build_siso_volume()
    print("\nSISO Volume Table:")
    print(siso_volume)

    # Build SISO risk table
    siso_risk = siso_analysis.build_siso_risk()
    print("\nSISO Risk Table:")
    print(siso_risk)