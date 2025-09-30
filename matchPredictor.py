import pandas as pd 

matches = pd.read_csv("superlig_2025_26_matches_played.csv")


matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

matches["time"] = matches["time"].astype(str).str.strip().str.replace(";", ":", regex=False)
matches["hour"] = pd.to_datetime(matches["time"], format="%H:%M", errors="coerce").dt.hour.astype("Int64")

matches["day_num"] = matches["date"].dt.dayofweek.astype("Int64")

label_map = {"L":0, "D":1, "W": 2}
matches["target"] = (
    matches["result"].astype(str).str.strip().str.upper().map(label_map).astype("Int64"))

from sklearn.ensemble import RandomForestClassifier 

