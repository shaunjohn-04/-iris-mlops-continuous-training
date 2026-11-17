import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("iris-classification")

MODEL_NAME = "IrisRandomForestModel"


def train_model():

    print("Loading Iris dataset...")

    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)

    with mlflow.start_run():

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)

        mlflow.log_metric("accuracy", accuracy)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        client = MlflowClient()

        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            model_info.registered_model_version
        )

    print("Model successfully registered in MLflow!")

    print("Model Version:", model_info.registered_model_version)


if __name__ == "__main__":
    train_model()