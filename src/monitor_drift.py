from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from evidently import Report
from evidently.presets import DataDriftPreset
from pathlib import Path


def check_data_drift():

    print("Loading reference data...")

    iris = load_iris(as_frame=True)

    # Reference dataset
    reference = iris.frame.sample(
        n=100,
        random_state=42
    )

    # Simulated production dataset
    current = iris.frame.sample(
        n=100,
        random_state=99
    )

    # Artificially introduce drift
    current["sepal length (cm)"] *= 1.5
    current["petal length (cm)"] *= 1.5

    print("Checking data drift...")

    # Generate Evidently report
    try:

        report = Report(
            metrics=[
                DataDriftPreset()
            ]
        )

        snapshot = report.run(
            reference_data=reference,
            current_data=current
        )

        report_path = (
            Path(__file__).resolve().parents[1]
            / "drift_report.html"
        )

        snapshot.save_html(str(report_path))

        print("Drift HTML report generated!")

    except Exception as e:

        print("HTML report generation warning:", e)

    # KS statistical test
    drifted_columns = 0

    feature_cols = iris.feature_names

    for col in feature_cols:

        p_value = ks_2samp(
            reference[col],
            current[col]
        ).pvalue

        print(f"{col} : p-value = {p_value:.5f}")

        if p_value < 0.05:
            drifted_columns += 1

    # Drift threshold: 50% of features
    dataset_drift = drifted_columns >= (
        len(feature_cols) / 2
    )

    print("Drifted Features:", drifted_columns)
    print("Dataset Drift Detected:", dataset_drift)

    return dataset_drift


if __name__ == "__main__":

    check_data_drift()