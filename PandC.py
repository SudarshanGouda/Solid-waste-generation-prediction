import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import StandardScaler


class SolidWastePrediction():
    '''
        A class to represent a prediction of Solid Waste.

        ...

        Attributes
        ----------
        model file : file
                    predictive model saved after the experiment
                In this case it is the 'Neural Network Regression' model

        Methods
            -------
            load_clean_data(data_file):

            # take a data file (*.txt) and preprocess it
                Import the text file, it processes , clean and standardize the file required for prediction
            Parameters

        predicted_vallue():

                Processed data will be predicted.

        predicted_outputs():

             Processed data will be predicted and concated with Original Value.
        pass'''

    def __init__(self, model_files):

        """
        Constructs all the necessary attributes for the person object.

        Parameters
        ----------
            model file : file
                predictive model saved after the experiment
            In this case it is the 'Neural Network Regression' model

        """
        # read the 'model' files which were saved
        with open(model_files, 'rb') as model_file:
            self.classification = pickle.load(model_file)

    def load_clean_data(self, df):
        # take a dataframe file
        """
            Import the csv file and it process and clean and standardize the file required for prediction
        Parameters
        ----------
        data_file : in .txt format

        Returns
        -------
        cleaned and processed file required for prediction

        '''
        """

        # store the data in a new variable for later use
        self.df_with_predictions = df.copy()

        ## One Hot encoding
        df['Type of Residence_Apartment'] = df['TypeofResidence'].apply(lambda x: 1 if x == 'Type of Residence_Apartment' else 0)
        df['Type of Residence_Gated Community'] = df['TypeofResidence'].apply(lambda x: 1 if x == 'Type of Residence_Gated Community' else 0)
        df['Type of Residence_Independent house'] = df['TypeofResidence'].apply(lambda x: 1 if x == 'Type of Residence_Independent house' else 0)
        df['Type of Residence_Small Community(2-3 Story)'] = df['TypeofResidence'].apply(lambda x: 1 if x == 'Type of Residence_Small Community(2-3 Story)' else 0)

        df['Mode of Disposal_Door to door collection'] = df['ModeofDisposal'].apply(lambda x: 1 if x == 'Mode of Disposal_Door to door collection' else 0)
        df['Mode of Disposal_Mobile Collection Vehicle'] = df['ModeofDisposal'].apply(lambda x: 1 if x == 'Mode of Disposal_Mobile Collection Vehicle' else 0)
        df['Mode of Disposal_Nearby dump yard'] = df['ModeofDisposal'].apply(lambda x: 1 if x == 'Mode of Disposal_Nearby dump yard' else 0)
        df['Mode of Disposal_Open area dumping'] = df['ModeofDisposal'].apply(lambda x: 1 if x == 'Mode of Disposal_Open area dumping' else 0)

        df['Frequency of Disposal_Alternate day'] = df['FrequencyofDisposal'].apply(lambda x: 1 if x == 'Frequency of Disposal_Alternate day' else 0)
        df['Frequency of Disposal_Daily'] = df['FrequencyofDisposal'].apply(lambda x: 1 if x == 'Frequency of Disposal_Daily' else 0)
        df['Frequency of Disposal_Once in a week'] = df['FrequencyofDisposal'].apply(lambda x: 1 if x == 'Frequency of Disposal_Once in a week' else 0)
        df['Frequency of Disposal_Twice a Week'] = df['FrequencyofDisposal'].apply(lambda x: 1 if x == 'Frequency of Disposal_Twice a Week' else 0)

        ## Droping the unwanted Column
        df.drop(['TypeofResidence', 'ModeofDisposal', 'FrequencyofDisposal'], axis=1, inplace=True)

        # re-order the columns in df
        Column_names = ['Numberofpeople', 'AnnualIncome', 'NumberofBinsintheVicinity',
       'Type of Residence_Apartment',
        'Type of Residence_Gated Community',
       'Type of Residence_Independent house',
       'Type of Residence_Small Community(2-3 Story)',
       'Mode of Disposal_Door to door collection',
       'Mode of Disposal_Mobile Collection Vehicle',
       'Mode of Disposal_Nearby dump yard',
       'Mode of Disposal_Open area dumping',
       'Frequency of Disposal_Alternate day',
        'Frequency of Disposal_Daily',
       'Frequency of Disposal_Once in a week',
       'Frequency of Disposal_Twice a Week']

        df = df[Column_names]

        self.preprocessed_data = df.copy()

        sc = StandardScaler()

        self.data = sc.fit_transform(df)


    def predicted_vallue(self):
        """
            Processed data will be predicted.
        ----------

        Returns
        -------
        Predicted values
        """
        if (self.data is not None):
            pred = self.classification.predict(self.data)[:, 1]
            return pred

    # predict the outputs and
    # add columns with these values at the end of the new data

    def predicted_outputs(self):
        """
            Processed data will be predicted and concated with Original Value.
        ----------

        Returns
        -------
        Predicted values
        """
        if (self.data is not None):
            self.prediction = self.classification.predict(self.data)
            self.preprocessed_data['Prediction'] = self.prediction
            return self.preprocessed_data

    def predicted_array_outputs(self):
        """
            Processed data will be predicted and concated with Original Value.
        ----------

        Returns
        -------
        Predicted values
        """
        if (self.new_arr is not None):
            self.prediction = self.classification.predict(self.new_arr)
            return self.prediction
