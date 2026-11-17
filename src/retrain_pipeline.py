import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.monitor_drift import check_data_drift


# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("iris-continuous-retraining")

MODEL_NAME = "IrisRandomForestModel"

ACCURACY_THRESHOLD = 0.90


def train_and_register_model():

    print("Initiating retraining process...")

    # Load dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    with mlflow.start_run(run_name="automated_retrain_run"):

        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluate model
        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f"Candidate Model Accuracy: {accuracy:.4f}"
        )

        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("trigger", "data_drift")

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Quality gate
        if accuracy >= ACCURACY_THRESHOLD:

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )

            # Promote candidate using champion alias
            client = MlflowClient()

            client.set_registered_model_alias(
                MODEL_NAME,
                "champion",
                model_info.registered_model_version
            )

            print("Model successfully registered!")

            print(
                "New Model Version:",
                model_info.registered_model_version
            )

            print("Champion alias updated!")

        else:

            print(
                "Accuracy below threshold. Promotion skipped."
            )

    return accuracy


def run_pipeline():

    print("Checking production data for drift...")

    drift_detected = check_data_drift()

    if drift_detected:

        print(
            "Data drift confirmed! Triggering retraining pipeline..."
        )

        train_and_register_model()

    else:

        print(
            "No significant drift detected. "
            "Model operating within normal bounds."
        )


if __name__ == "__main__":

    run_pipeline()