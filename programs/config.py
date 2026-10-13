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
    """Read SASHELP.HEART straight from the SAS install."""
    df = pd.read_sas(SASHELP_HEART, encoding="latin1")
    for c in df.select_dtypes("object"):
        df[c] = df[c].astype(str).str.strip().replace({"": None, "nan": None})
    return df