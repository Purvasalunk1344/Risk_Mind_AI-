from pathlib import Path
import pandas as pd

ROOT = Path("/workspaces/myfolder/heart_risk_project")
SEED = 2026

RAW = ROOT / "data/raw"
PROC = ROOT / "data/processed"
MODELS = ROOT / "models"
MON = ROOT / "monitoring"
PLOTS = ROOT / "output/plots"
for p in (RAW, PROC, MODELS, MON, PLOTS):
    p.mkdir(parents=True, exist_ok=True)

SASHELP_HEART = "/opt/sas/viya/home/SASFoundation/sashelp/heart.sas7bdat"

def load_raw() -> pd.DataFrame:
    """Read SASHELP.HEART exported by 00_export_raw.sas."""
    f = RAW / "heart_raw.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found. Run programs/00_export_raw.sas in SAS first.")
    return pd.read_csv(f)