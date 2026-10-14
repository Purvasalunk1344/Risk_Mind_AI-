import argparse
import json

import pandas as pd

from config import MODELS
from macros.score_patient import score_patient


def score_dataframe(df, model_name="logistic", threshold=0.20):
    """Score a dataframe of patients and attach probability / band labels."""
    rows = []
    for _, row in df.iterrows():
        rows.append(score_patient(row.to_dict(), model_name=model_name, threshold=threshold))
    out = pd.DataFrame(rows)
    if "risk" in df.columns:
        out["risk"] = df["risk"].astype(float)
        out["risk_band"] = pd.cut(out["risk"], [-1, 0.2, 0.5, 2], labels=["Low", "Medium", "High"], right=False)
    return out


def main():
    parser = argparse.ArgumentParser(description="Deploy the champion heart-risk model.")
    parser.add_argument("--model", default="logistic", help="Champion model name in the models folder")
    parser.add_argument("--threshold", type=float, default=0.20, help="High-risk flagging threshold")
    parser.add_argument("--input", help="CSV file with patient rows to score; if omitted, a JSON example is shown")
    parser.add_argument("--output", help="Optional CSV output path for scored records")
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input, keep_default_na=False, na_values=[""])
        scored = score_dataframe(df, model_name=args.model, threshold=args.threshold)
        if args.output:
            scored.to_csv(args.output, index=False)
            print(f"Saved scored rows to {args.output}")
        else:
            print(scored.to_string(index=False))
        return

    example = {
        "AgeAtStart": 58,
        "Sex": "Female",
        "Systolic": 145,
        "Diastolic": 88,
        "Cholesterol": 210,
        "Weight": 160,
        "Height": 66,
        "Smoking": 12,
    }
    result = score_patient(example, model_name=args.model, threshold=args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
