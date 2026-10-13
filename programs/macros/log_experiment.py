from datetime import datetime
import pandas as pd
from config import MON


def log_experiment(model_name, params, metrics, dataset="valid"):
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "model": model_name, "dataset": dataset,
           "params": str(params), **{k: round(v, 4) for k, v in metrics.items()}}
    f = MON / "experiment_log.csv"
    df = pd.DataFrame([row])
    df.to_csv(f, mode="a", header=not f.exists(), index=False)