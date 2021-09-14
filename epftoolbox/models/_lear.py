"""
Classes and functions to implement the LEAR model for electricity price forecasting
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
import pandas as pd
from statsmodels.robust import mad
import os

from sklearn.linear_model import LassoLarsIC, Lasso
from epftoolbox.data import scaling
from epftoolbox.data import read_data
from epftoolbox.evaluation import MAE, sMAPE

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class LEAR(object):
    """
    New version of the class used to build a LEAR model, recalibrate it, and predict DA electricity prices.

    New features:
    - User can now supply the lags (in days) used for creating new lagged exogenous features SEPARATELY.
      Previously all exog. features used d-1 and d-2. NOTE: the original documentation wrongly stated d-1 and d-7!
    - When the original X features get split/reshaped into 24 hourly features, the hourly features now get checked
      for being constant. Constant features get removed, and their original index recorded.
      This is needed because during calibration all X features get scaled by MAD or std, and
      if the scaling factor is 0, it leads to an error. This error is very common for solar data (eg. solar power
      is always 0 at midnight)!
    - Order of the newly built hourly and lagged features is changed to be grouped by feature:lag:hour, instead of
      hour:lag:feature. This makes it easier to track the coefficients of the different features when analyzing the
      results. This is less efficient wrt memory when building the array, but the practical difference is marginal.
    - Not recalibrating in every forecast loop (daily) is now more convenient. Previously all the methods were there,
      but had to be called manually in the forecast loop. In addition training data arrays would be constructed, even
      though only test data is needed.
    - changed variable names to Python naming conventions (eg. predDatesTrain -> pred_dates_train)

    An example on how to use this class is provided :ref:`here<learex2>`.
    
    Parameters
    ----------
    calibration_window : int, optional
        Calibration window (in days) for the LEAR model.

    lags: list[int] or list[list[int]], optional
        The number of lags for the exogenous features. If list of ints, then the same
        lags are applied for all exogenous features. If list of lists of ints then
        the number of lists must match the number of exogenous features.
        
    """

    def __init__(self, calibration_window=364 * 3, lags=[1, 2]):
        # Calibration window in hours
        self.calibration_window = calibration_window
        self.lags = lags

    # Ignore convergence warnings from scikit-learn LASSO module
    @ignore_warnings(category=ConvergenceWarning)
    def recalibrate(self, X_train, Y_train):
        """Function to recalibrate the LEAR model. 
        
        It uses a training (Xtrain, Ytrain) pair for recalibration
        
        Parameters
        ----------
        X_train : numpy.array
            Input in training dataset. It should be of size *[n,m]* where *n* is the number of days
            in the training dataset and *m* the number of input features
        
        Y_train : numpy.array
            Output in training dataset. It should be of size *[n,24]* where *n* is the number of days 
            in the training dataset and 24 are the 24 prices of each day
                
        Returns
        -------
        numpy.array
            The prediction of day-ahead prices after recalibrating the model        
        
        """

        # # Applying Invariant, aka asinh-median transformation to the prices
        [Y_train], self.scalerY = scaling([Y_train], 'Invariant')

        # # Rescaling all inputs except dummies (7 last features)
        [Xtrain_no_dummies], self.scalerX = scaling([X_train[:, :-7]], 'Invariant')
        X_train[:, :-7] = Xtrain_no_dummies

        self.models = {}
        for h in range(24):
            # Estimating lambda hyperparameter using LARS
            param_model = LassoLarsIC(criterion='aic', max_iter=2500)
            param = param_model.fit(X_train, Y_train[:, h]).alpha_

            # Re-calibrating LEAR using standard LASSO estimation technique
            model = Lasso(max_iter=2500, alpha=param)
            model.fit(X_train, Y_train[:, h])

            self.models[h] = model

    def predict(self, X):
        """Function that makes a prediction using some given inputs.
        
        Parameters
        ----------
        X : numpy.array
            Input of the model.
        
        Returns
        -------
        numpy.array
            An array containing the predictions.
        """

        # Predefining predicted prices
        Yp = np.zeros(24)

        # # Rescaling all inputs except dummies (7 last features)
        X_no_dummies = self.scalerX.transform(X[:, :-7])
        X[:, :-7] = X_no_dummies

        # Predicting the current date using a recalibrated LEAR
        for h in range(24):
            # Predicting test dataset and saving
            Yp[h] = self.models[h].predict(X)

        Yp = self.scalerY.inverse_transform(Yp.reshape(1, -1))

        return Yp

    def recalibrate_predict(self, X_train, Y_train, X_test):
        """Function that first recalibrates the LEAR model and then makes a prediction.

        The function receives the training dataset, and trains the LEAR model. Then, using
        the inputs of the test dataset, it makes a new prediction.
        
        Parameters
        ----------
        X_train : numpy.array
            Input of the training dataset.
        X_test : numpy.array
            Input of the test dataset.
        Y_train : numpy.array
            Output of the training dataset.
        
        Returns
        -------
        numpy.array
            An array containing the predictions in the test dataset.
        """

        self.recalibrate(X_train=X_train, Y_train=Y_train)

        Yp = self.predict(X=X_test)

        return Yp

    def _build_and_split_XYs(self, df_train, df_test=None, date_test=None, calibration=True):
        """
        Internal function that generates the X,Y arrays for training and testing based on pandas dataframes

        Parameters
        ----------
        df_train : pandas.DataFrame
            Pandas dataframe containing the training data

        df_test : pandas.DataFrame
            Pandas dataframe containing the test data

        date_test : datetime, optional
            If given, then the test dataset is only built for that date

        calibration : bool, optional
            If True, only test array is built

        Returns
        -------
        list[numpy.array]
            [X_train, Y_train, X_test] as the list containing the (X,Y) input/output pairs for training,
            and the input for testing
        """

        # 96 prices + n_exogenous * (24 * 3 exogeneous) + 7 weekday dummies
        # Price lags: D-1, D-2, D-3, D-7
        # Exogeneous inputs lags: D, D-1, D-7

        # Checking that the first index in the dataframes corresponds with the hour 00:00
        if df_train.index[0].hour != 0 or df_test.index[0].hour != 0:
            print('Problem with the index')

        # Defining the number of Exogenous inputs
        n_exogenous_inputs = len(df_train.columns) - 1

        # Checking if lags are validly defined
        assert (isinstance(self.lags, list))
        # If there are no lagged values
        if len(self.lags) == 0:
            lags = []
            n_features = 96 + 7
        # If list of ints given repeat it for all exog. features
        elif isinstance(self.lags[0], int):
            lags = []
            for i in range(n_exogenous_inputs):
                lags.append(self.lags)
            n_features = 96 + 7 + n_exogenous_inputs * (len(self.lags) + 1) * 24

        # If list of ints given repeat it for all exog. features
        elif isinstance(self.lags[0], list):
            assert (len(self.lags) == n_exogenous_inputs)
            lags = self.lags
            # Calculate number of new features
            n_new_exog_feat = 0
            for feat_lags in lags:
                n_new_exog_feat += (len(feat_lags) + 1) * 24
            n_features = 96 + 7 + n_new_exog_feat
        else:
            raise ValueError("Something is wrong with the list of lags parameter.")

        # Extracting the predicted dates for testing and training. We leave the first week of data
        # out of the prediction as we the maximum lag can be one week

        # We define the potential time indexes that have to be forecasted in training
        # and testing
        if calibration:
            index_train = df_train.loc[df_train.index[0] + pd.Timedelta(weeks=1):].index

        # For testing, the test dataset is different depending on whether a specific test
        # dataset is provided
        if date_test is None:
            index_test = df_test.loc[df_test.index[0] + pd.Timedelta(weeks=1):].index
        else:
            index_test = df_test.loc[date_test:date_test + pd.Timedelta(hours=23)].index
        # We extract the prediction dates/days.
        if calibration:
            pred_dates_train = index_train.round('1H')[::24]
        pred_dates_test = index_test.round('1H')[::24]

        # We create two dataframe to build XY.
        # These dataframes have as indices the first hour of the day (00:00)
        # and the columns represent the 23 possible horizons/dates along a day
        if calibration:
            index_train = pd.DataFrame(index=pred_dates_train, columns=['h' + str(hour) for hour in range(24)])
        index_test = pd.DataFrame(index=pred_dates_test, columns=['h' + str(hour) for hour in range(24)])
        for hour in range(24):
            if calibration:
                index_train.loc[:, 'h' + str(hour)] = index_train.index + pd.Timedelta(hours=hour)
            index_test.loc[:, 'h' + str(hour)] = index_test.index + pd.Timedelta(hours=hour)

        # Preallocating in memory the X and Y arrays
        if calibration:
            X_train = np.zeros([index_train.shape[0], n_features])
            Y_train = np.zeros([index_train.shape[0], 24])
        else:
            X_train = None
            Y_train = None
        X_test = np.zeros([index_test.shape[0], n_features])

        # Variable keeping track of nr of features built so far
        feature_index = 0

        # Adding the historical prices during days D-1, D-2, D-3, and D-7
        # For each possible past day where prices can be included
        for past_day in [1, 2, 3, 7]:
            # For each hour of a day
            for hour in range(24):
                # We define the corresponding past time indexs using the auxiliary dataframses
                if calibration:
                    past_index_train = pd.to_datetime(index_train.loc[:, 'h' + str(hour)].values) - \
                                     pd.Timedelta(hours=24 * past_day)
                past_index_test = pd.to_datetime(index_test.loc[:, 'h' + str(hour)].values) - \
                                pd.Timedelta(hours=24 * past_day)

                # We include the historical prices at day D-past_day and hour "h"
                if calibration:
                    X_train[:, feature_index] = df_train.loc[past_index_train, 'Price']
                X_test[:, feature_index] = df_test.loc[past_index_test, 'Price']
                feature_index += 1

        # Adding the lagged version of the exogenous inputs
        # For each of the exogenous input
        for exog in range(n_exogenous_inputs):
            # For each possible past day where exogenous inputs can be included
            # lag 0 is the original feature!
            feature_lags = [0] + lags[exog]
            for lag in feature_lags:
                # For each hour of a day
                for hour in range(24):
                    # Defining the corresponding past time indices using the auxiliary dataframes
                    if calibration:
                        past_index_train = pd.to_datetime(index_train.loc[:, 'h' + str(hour)].values) - \
                                         pd.Timedelta(hours=24 * lag)
                    past_index_test = pd.to_datetime(index_test.loc[:, 'h' + str(hour)].values) - \
                                    pd.Timedelta(hours=24 * lag)

                    # Including the exogenous input at day D-past_day and hour "h"
                    if calibration:
                        X_train[:, feature_index] = df_train.loc[past_index_train, 'Exogenous ' + str(exog + 1)]
                    X_test[:, feature_index] = df_test.loc[past_index_test, 'Exogenous ' + str(exog + 1)]
                    feature_index += 1

        # Adding the dummy variables that depend on the day of the week. Monday is 0 and Sunday is 6
        # For each day of the week
        for dayofweek in range(7):
            if calibration:
                X_train[index_train.index.dayofweek == dayofweek, feature_index] = 1
            X_test[index_test.index.dayofweek == dayofweek, feature_index] = 1
            feature_index += 1
        # Extracting the predicted values Y
        for hour in range(24):
            # Defining time index at hour h
            if calibration:
                future_index_train = pd.to_datetime(index_train.loc[:, 'h' + str(hour)].values)
                # Extracting Y value based on time indices
                Y_train[:, hour] = df_train.loc[future_index_train, 'Price']

        # Remove features with 0 MAD
        # Only check features for being constant when calibrating
        if calibration:
            self.const_features = np.where(mad(X_train[:, :-7], axis=0) == 0)[0]
            X_train = np.concatenate([np.delete(X_train[:, :-7], self.const_features, axis=1), X_train[:, -7:]], axis=1)
        X_test = np.concatenate([np.delete(X_test[:, :-7], self.const_features, axis=1), X_test[:, -7:]], axis=1)

        return X_train, Y_train, X_test

    def recalibrate_and_forecast_next_day(self, df, next_day_date):
        """Easy-to-use interface for daily recalibration and forecasting of the LEAR model.
        
        The function receives a pandas dataframe and a date. Usually, the data should
        correspond with the date of the next-day when using for daily recalibration.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe of historical data containing prices and *N* exogenous inputs. 
            The index of the dataframe should be dates with hourly frequency. The columns 
            should have the following names ``['Price', 'Exogenous 1', 'Exogenous 2', ...., 'Exogenous N']``.

        next_day_date : datetime
            Date of the day-ahead.
        
        Returns
        -------
        numpy.array
            The prediction of day-ahead prices.
        """

        # We define the new training dataset and test datasets 
        df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]
        # Limiting the training dataset to the calibration window
        df_train = df_train.iloc[-self.calibration_window * 24:]

        # We define the test dataset as the next day (they day of interest) plus the last two weeks
        # in order to be able to build the necessary input features. 
        df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]

        # Generating X,Y pairs for predicting prices
        X_train, Y_train, X_test, = self._build_and_split_XYs(
            df_train=df_train, df_test=df_test, date_test=next_day_date)

        # Recalibrating the LEAR model and extracting the prediction
        Yp = self.recalibrate_predict(X_train=X_train, Y_train=Y_train, X_test=X_test)

        return Yp

    def forecast_next_day(self, df, next_day_date):
        """
        Easy-to-use interface forecasting with a pre-calibrated LEAR model.
        To be used when the model is not recalibrated every day.

        The function receives a pandas dataframe and a date.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe of historical data containing prices and *N* exogenous inputs.
            The index of the dataframe should be dates with hourly frequency. The columns
            should have the following names ``['Price', 'Exogenous 1', 'Exogenous 2', ...., 'Exogenous N']``.

        calibration_window : int
            Calibration window (in days) for the LEAR model.

        next_day_date : datetime
            Date of the day-ahead.

        Returns
        -------
        numpy.array
            The prediction of day-ahead prices.
        """

        # We define the new training dataset and test datasets
        df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]
        # Limiting the training dataset to the calibration window
        df_train = df_train.iloc[-self.calibration_window * 24:]

        # We define the test dataset as the next day (they day of interest) plus the last two weeks
        # in order to be able to build the necessary input features.
        df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]

        # Generating X,Y pairs for predicting prices
        _, _, X_test, = self._build_and_split_XYs(
            df_train=df_train, df_test=df_test, date_test=next_day_date, calibration=False)

        # Recalibrating the LEAR model and extracting the prediction
        Yp = self.predict(X=X_test)

        return Yp

def evaluate_lear_in_test_dataset(path_datasets_folder=os.path.join('.', 'datasets'),
                              path_recalibration_folder=os.path.join('.', 'experimental_files'),
                              dataset='PJM', years_test=2, calibration_window=364 * 3,
                              begin_test_date=None, end_test_date=None):
    """Function for easy evaluation of the LEAR model in a test dataset using daily recalibration.

    The test dataset is defined by a market name and the test dates dates. The function
    generates the test and training datasets, and evaluates a LEAR model considering daily recalibration.

    An example on how to use this function is provided :ref:`here<learex1>`.

    Parameters
    ----------
    path_datasets_folder : str, optional
        path where the datasets are stored or, if they do not exist yet,
        the path where the datasets are to be stored.

    path_recalibration_folder : str, optional
        path to save the files of the experiment dataset.

    dataset : str, optional
        Name of the dataset/market under study. If it is one one of the standard markets,
        i.e. ``"PJM"``, ``"NP"``, ``"BE"``, ``"FR"``, or ``"DE"``, the dataset is automatically downloaded. If the name
        is different, a dataset with a csv format should be place in the ``path_datasets_folder``.

    years_test : int, optional
        Number of years (a year is 364 days) in the test dataset. It is only used if
        the arguments ``begin_test_date`` and ``end_test_date`` are not provided.

    calibration_window : int, optional
        Number of days used in the training dataset for recalibration.

    begin_test_date : datetime/str, optional
        Optional parameter to select the test dataset. Used in combination with the argument
        ``end_test_date``. If either of them is not provided, the test dataset is built using the
        ``years_test`` argument. ``begin_test_date`` should either be a string with the following
        format ``"%d/%m/%Y %H:%M"``, or a datetime object.

    end_test_date : datetime/str, optional
        Optional parameter to select the test dataset. Used in combination with the argument
        ``begin_test_date``. If either of them is not provided, the test dataset is built using the
        ``years_test`` argument. ``end_test_date`` should either be a string with the following
        format ``"%d/%m/%Y %H:%M"``, or a datetime object.

    Returns
    -------
    pandas.DataFrame
        A dataframe with all the predictions in the test dataset. The dataframe is also written to path_recalibration_folder.
    """

    # Checking if provided directory for recalibration exists and if not create it
    if not os.path.exists(path_recalibration_folder):
        os.makedirs(path_recalibration_folder)

    # Defining train and testing data
    df_train, df_test = read_data(dataset=dataset, years_test=years_test, path=path_datasets_folder,
                                  begin_test_date=begin_test_date, end_test_date=end_test_date)

    # Defining unique name to save the forecast
    forecast_file_name = 'LEAR_forecast' + '_dat' + str(dataset) + '_YT' + str(years_test) + \
                         '_CW' + str(calibration_window) + '.csv'

    forecast_file_path = os.path.join(path_recalibration_folder, forecast_file_name)

    # Defining empty forecast array and the real values to be predicted in a more friendly format
    forecast = pd.DataFrame(index=df_test.index[::24], columns=['h' + str(k) for k in range(24)])
    real_values = df_test.loc[:, ['Price']].values.reshape(-1, 24)
    real_values = pd.DataFrame(real_values, index=forecast.index, columns=forecast.columns)

    forecast_dates = forecast.index

    model = LEAR(calibration_window=calibration_window)

    # For loop over the recalibration dates
    for date in forecast_dates:
        # For simulation purposes, we assume that the available data is
        # the data up to current date where the prices of current date are not known
        data_available = pd.concat([df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0)

        # We set the real prices for current date to NaN in the dataframe of available data
        data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

        # Recalibrating the model with the most up-to-date available data and making a prediction
        # for the next day
        Yp = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=date,
                                                     calibration_window=calibration_window)
        # Saving the current prediction
        forecast.loc[date, :] = Yp

        # Computing metrics up-to-current-date
        mae = np.mean(MAE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values))
        smape = np.mean(sMAPE(forecast.loc[:date].values.squeeze(), real_values.loc[:date].values)) * 100

        # Pringint information
        print('{} - sMAPE: {:.2f}%  |  MAE: {:.3f}'.format(str(date)[:10], smape, mae))

        # Saving forecast
        forecast.to_csv(forecast_file_path)

    return forecast
