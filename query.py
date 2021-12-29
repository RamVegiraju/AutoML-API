import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

class autoML:
    def __init__(self, data_path, target_column, model_type, metric_type):
        self.data_path = data_path
        self.target_column = target_column
        self.model_type = model_type
        self.metric_type = metric_type
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.RMSE = None
        self.MAE = None


    def preprocess(self):
        df = pd.read_csv(self.data_path)
        columns = df.columns
        if self.target_column not in columns:
            raise ValueError("Target column not present in dataset")
        X = df.drop([self.target_column], axis = 1) 
        y = df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def train(self):
        model_types = ["linear_regression", "random_forest"]
        if self.model_type not in model_types:
            raise ValueError(f"Enter a model type that is supported from the following list: {model_types}")

        if self.model_type == "linear_regression":
            #Model Creation
            lm = LinearRegression()
            lm.fit(self.X_train,self.y_train)
            #Model prediction
            Y_Pred = lm.predict(self.X_test)
            MAE = metrics.mean_absolute_error(self.y_test, Y_Pred) 
            RMSE = np.sqrt(metrics.mean_squared_error(self.y_test, Y_Pred))

        elif self.model_type == "random_forest":
            regressor = RandomForestRegressor(n_estimators=40) 
            regressor.fit(self.X_train, self.y_train) 
            Y_Pred = regressor.predict(self.X_test)
            MAE = metrics.mean_absolute_error(self.y_test, Y_Pred) 
            RMSE = np.sqrt(metrics.mean_squared_error(self.y_test, Y_Pred))

        self.MAE = MAE
        self.RMSE = RMSE

    def evaluate(self):
        metric_types = ["MAE", "RMSE"]
        if self.metric_type not in metric_types:
            raise ValueError(f"Enter a metric that is supported from the following list: {metric_types}")

        if self.metric_type == "MAE":
            return f"{self.metric_type}: {self.MAE}"
        return f"{self.metric_type}: {self.RMSE}"

    def logModelInfo(self):
        modelDetails = {
            "ModelType": self.model_type,
            "MAE": self.MAE,
            "RMSE": self.RMSE
        }
        return modelDetails


if __name__ == '__main__':
    auto_model = autoML("petrol.csv", "Petrol_Consumption", "random_forest", "RMSE")
    auto_model.preprocess()
    auto_model.train()
    print(auto_model.evaluate())
    print(auto_model.logModelInfo())