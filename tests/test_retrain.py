import mlflow

from src.monitor_drift import check_data_drift

from src.retrain_pipeline import (
    train_and_register_model,
    run_pipeline,
    MODEL_NAME
)


mlflow.set_tracking_uri("sqlite:///mlflow.db")


def test_data_drift_detection():

    drift_status = check_data_drift()

    assert isinstance(drift_status, bool)

    assert drift_status is True


def test_train_and_register_model_execution():

    train_and_register_model()

    experiment = mlflow.get_experiment_by_name(
        "iris-continuous-retraining"
    )

    assert experiment is not None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    assert len(runs) > 0

    latest_run = runs.iloc[0]

    assert latest_run["metrics.accuracy"] >= 0.90


def test_full_retrain_pipeline_trigger(capsys):

    run_pipeline()

    captured = capsys.readouterr()

    assert "Checking production data for drift..." in captured.out

    assert (
        "Data drift confirmed! Triggering retraining pipeline..."
        in captured.out
    )