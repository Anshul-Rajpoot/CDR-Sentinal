"""Train/evaluate the model and create the required submission file."""
from __future__ import annotations

from src.pipeline import load_bundle, predict_test


def main() -> None:
    bundle = load_bundle(force_retrain=True)
    print("\nValidation results")
    print("-" * 60)
    for row in bundle.model_comparison:
        print(
            f'{row["model"]:<20} '
            f'MAE={row["mae"]:>8.3f}  '
            f'RMSE={row["rmse"]:>8.3f}  '
            f'R2={row["r2"]:>7.3f}'
        )

    output = predict_test(bundle)
    output_path = __import__("pathlib").Path(__file__).resolve().parent / "submission.csv"
    output.to_csv(output_path, index=False)
    print(f"\nSaved {len(output)} predictions to {output_path.name}")
    print("Columns:", ", ".join(output.columns))


if __name__ == "__main__":
    main()
