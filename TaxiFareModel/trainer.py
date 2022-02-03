from TaxiFareModel.encoders import TimeFeaturesEncoder,DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data
from TaxiFareModel.data import clean_data
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import set_config; set_config(display='diagram')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from memoized_property import memoized_property
import mlflow
import joblib
from  mlflow.tracking import MlflowClient



class Trainer():

    
    EXPERIMENT_NAME = "[DE] [Berlin] [adelinaionescu] TaxiFare v3"
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.MLFLOW_URI = "https://mlflow.lewagon.co/"

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', GradientBoostingRegressor(n_estimators=100))
            ])
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        pipeline = self.set_pipeline()
        pipeline.fit(self.X, self.y)
        return pipeline

    def evaluate(self, X, y):
        """evaluates the pipeline on df_test and return the RMSE"""
        pipeline = self.run()
        y_pred = pipeline.predict(X)
        rmse = compute_rmse(y_pred, y)
        self.mlflow_log_metric("rmse", rmse)
        self.mlflow_log_param("model", "GBR")
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
    
    def save_model(self):
        """ Save the trained model into a model.joblib file """
        pipeline = self.set_pipeline()
        saved_model = joblib.dump(pipeline, "pipeline.joblib")
        return saved_model




if __name__ == "__main__":
    df = get_data()
    new_data = clean_data(df)
    y = new_data["fare_amount"]
    X = new_data.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    trainer = Trainer(X_train, y_train)
    trainer.run()
    trainer.evaluate(X_test, y_test)
    trainer.save_model()
    print('TODO')
