import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


DATA_PATH = "data/breast_cancer.csv"
MODEL_NAME = "BreastCancerModel"


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    return accuracy, precision, recall, f1


def main():

    # -----------------------------
    # 1. Load dataset
    # -----------------------------

    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    # -----------------------------
    # 2. Train-test split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # 3. Create MLflow experiment
    # -----------------------------

    mlflow.set_experiment("Breast Cancer Classification")

    # -----------------------------
    # 4. Define models
    # -----------------------------

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    # -----------------------------
    # 5. Track best model
    # -----------------------------

    best_model = None
    best_model_name = ""
    best_accuracy = 0
    best_run_id = ""

    # -----------------------------
    # 6. Train all models
    # -----------------------------

    for name, model in models.items():

        with mlflow.start_run(run_name=name) as run:

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            accuracy, precision, recall, f1 = evaluate_model(
                model,
                X_test,
                y_test
            )

            # Print results
            print(f"\n{name}")
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")

            # Log parameter
            mlflow.log_param("model", name)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Log model artifact
            mlflow.sklearn.log_model(
                model,
                name="model"
            )

            # Check whether this is the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name
                best_run_id = run.info.run_id
    # -----------------------------
    # 7. Register ONLY best model
    # -----------------------------


    joblib.dump(best_model, "model.pkl")
    print("Best model saved to model.pkl")

    model_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    # -----------------------------
    # 8. Print final result
    # -----------------------------

    print("\n==============================")
    print("BEST MODEL")
    print("==============================")

    print(f"Model       : {best_model_name}")
    print(f"Accuracy    : {best_accuracy:.4f}")
    print(f"Run ID      : {best_run_id}")
    print(f"Model Name  : {MODEL_NAME}")
    print(f"Model Version: {registered_model.version}")


if __name__ == "__main__":
    main()