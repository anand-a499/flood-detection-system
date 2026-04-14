
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "flood_risk_model.joblib"

rng = np.random.default_rng(42)
n = 3000
rainfall = rng.uniform(0,300,n)
river = rng.uniform(0,12,n)
soil = rng.uniform(0,100,n)
drain = rng.uniform(0,100,n)
elev = rng.uniform(0,100,n)
prev24 = rng.uniform(0,300,n)

score = (
    0.32*rainfall +
    18*river +
    0.22*soil -
    0.20*drain +
    0.16*elev +
    0.24*prev24 +
    rng.normal(0,18,n)
)

y = np.digitize(score, bins=[90, 155, 235])  # 0 low,1 medium,2 high,3 critical
X = np.column_stack([rainfall, river, soil, drain, elev, prev24])

model = RandomForestClassifier(n_estimators=220, max_depth=10, random_state=42)
model.fit(X, y)
joblib.dump(model, MODEL_PATH)
print(f"saved to {MODEL_PATH}")
