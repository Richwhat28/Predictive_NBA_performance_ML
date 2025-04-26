import pandas as pd
import numpy as np
import joblib
import pathlib
from sklearn.linear_model  import LinearRegression
from sklearn.preprocessing import StandardScaler

# — your existing helpers — 
def find_csv(filename: str) -> pathlib.Path:
    for candidate in (
        pathlib.Path(filename),
        pathlib.Path("Datasets") / filename
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename}")

def load_season_averages() -> pd.DataFrame:
    folder = pathlib.Path("Datasets") / "NBA dataset 23-24 (NEW)"
    if not folder.exists():
        raise FileNotFoundError(f"{folder} not found.")
    dfs = []
    for f in folder.glob("*.csv"):
        df = pd.read_csv(f)
        if "Player" in df.columns:
            df = df.rename(columns={"Player": "PLAYER_NAME"})
        dfs.append(df)
    season_df = pd.concat(dfs, ignore_index=True)
    return season_df.rename(columns={
        "MP":   "MP_per_game",
        "PTS":  "PTS_per_game",
        "AST":  "AST_per_game",
        "TRB":  "TRB_per_game",
        "eFG%": "eFG_pct"
    })[["PLAYER_NAME","MP_per_game","PTS_per_game","AST_per_game","TRB_per_game","eFG_pct"]]

# — Load and prepare your game_player_data exactly as in your prediction script —
shots_csv   = find_csv("23-24 NBA dataset.csv")
ranking_csv = find_csv("23-24 NBA RANKING.csv")
nba_df     = pd.read_csv(shots_csv)
ranking_df = pd.read_csv(ranking_csv)
# … (all your dtype enforcement, grouping into first_half & second_half, merging, feature engineering)
# at the end you should have a DataFrame `game_player_data` with:
#   X = game_player_data[feature_cols]
#   y = game_player_data["second_half_points"]

feature_cols = [
    'first_half_attempts','first_half_made','first_half_points','first_half_fg_pct',
    'avg_shot_distance','pct_3pt_attempts','home_flag','opponent_difficulty',
    'MP_per_game','PTS_per_game','AST_per_game','TRB_per_game','eFG_pct'
]

X = game_player_data[feature_cols]
y = game_player_data["second_half_points"]

# — Fit scaler and model —
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

model = LinearRegression().fit(X_scaled, y)

# — Dump to disk —
joblib.dump(model,  'linear_performance_predictor.pkl')
joblib.dump(scaler, 'linear_scaler.pkl')

print("✅ Saved linear_performance_predictor.pkl and linear_scaler.pkl")
