import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix 
from pathlib import Path    
import json

played = pd.read_csv("superlig_2025_26_matches_played.csv")
full   = pd.read_csv("superlig_2025_26.csv")


for df in [played, full]:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["time"] = df["time"].astype(str).str.strip().str.replace(";", ":", regex=False)
    df["hour"] = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour.fillna(0).astype(int)
    df["day_num"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

label_map = {"L":0, "D":1, "W": 2}
played["target"] = played["result"].astype(str).str.strip().str.upper().map(label_map)


teams = pd.concat([played["team"],played["opponent"], full["team"], full["opponent"]]).dropna().unique()
team_codes = {t:i for i,t in enumerate(sorted(teams))}


