import json
from pathlib import Path

from gold_model import (
    load_and_prepare_data,
    train_random_forest,
    evaluate_model,
    save_model,
    MODEL_PATH,
)


METRICS_PATH = Path("artifacts") / "metrics.json"


def main():
    gold, X, y, X_train, X_test, y_train, y_test = load_and_prepare_data()

    model = train_random_forest(X_train, y_train)
    save_model(model, MODEL_PATH)

    metrics_dict = evaluate_model(model, X_test, y_test)
    metrics_to_save = {
        "r2": float(metrics_dict["r2"]),
        "mae": float(metrics_dict["mae"]),
        "rmse": float(metrics_dict["rmse"]),
        "rows": int(len(gold)),
        "features": int(X.shape[1]),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics_to_save, indent=2), encoding="utf-8")

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(
        "Test metrics -> "
        f"R2: {metrics_to_save['r2']:.4f}, "
        f"MAE: {metrics_to_save['mae']:.4f}, "
        f"RMSE: {metrics_to_save['rmse']:.4f}"
    )


if __name__ == "__main__":
    main()
