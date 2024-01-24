import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split

from typing import Tuple, Union, List

from challenge import utils


class DelayModel:

    def __init__(self):
        self._base_model = self.__load_model("latam_base_model.pkl")
        self._model = self.__load_model("latam_model.pkl")
    def preprocess(
            self,
            data: pd.DataFrame,
            target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )
        if target_column is not None:
            threshold_in_minutes = 15
            data['min_diff'] = data.apply(utils.get_min_diff, axis=1)
            data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
            data['high_season'] = data['Fecha-I'].apply(utils.is_high_season)
            data['period_day'] = data['Fecha-I'].apply(utils.get_period_day)

            target = data[target_column] if target_column is not None else data["delay"]

            x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
            self._base_model.fit(x_train, y_train)
            if self._model is None:
                self.__save_model(self._base_model, "latam_base_model.pkl")
            feature_important = self._base_model.get_booster().get_score(importance_type='weight')
            top_10_features = [k for k, v in sorted(feature_important.items(), key=lambda item: item[1], reverse=True)][
                              :10]


            return features[top_10_features], target.to_frame()
        else:
            feature_important = self._base_model.get_booster().get_score(importance_type='weight')
            top_10_features = [k for k, v in sorted(feature_important.items(), key=lambda item: item[1], reverse=True)][
                              :10]
            valid_features = [feature for feature in top_10_features if feature in features.columns]
            return features[valid_features]

    def fit(
            self,
            features: pd.DataFrame,
            target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        target = target.squeeze()
        x_train2, x_test2, y_train2, y_test2 = train_test_split(features, target, test_size=0.33,
                                                                random_state=42)
        scale = utils.get_data_balance(target)
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        self._model.fit(features, target)
        self.__save_model(self._model, "latam_model.pkl")

    def predict(
            self,
            features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        predicted_target = self._model.predict(
            features
        )
        return [int(value) for value in list(predicted_target)]

    def __save_model(self, model, model_pkl_file) -> None:
        pickle.dump(model, open(f"models/{model_pkl_file}", 'wb'))

    def __load_model(self, model_pkl_file):
        if os.path.exists(f"models/{model_pkl_file}"):
            loaded_model = pickle.load(open(f"models/{model_pkl_file}", 'rb'))
            #when(loaded_model).predict(ANY).thenReturn(np.array([0]))
            return loaded_model
        elif "base" in model_pkl_file:
            return xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        else:
            return None
