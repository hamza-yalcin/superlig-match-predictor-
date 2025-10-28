import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


played = pd.read_csv("superlig_2025_26_matches_played.csv")
full   = pd.read_csv("superlig_2025_26.csv")


def base_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["time"] = df["time"].astype(str).str.strip().str.replace(";", ":", regex=False)
    df["hour"] = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour.fillna(0).astype(int)
    df["day_num"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["home"] = df["venue"].astype(str).str.contains("Stadyumu", case=False, na=False).astype(int)
    return df

played = base_clean(played)
full   = base_clean(full)


label_map = {"L":0, "D":1, "W":2}
played["result"] = played["result"].astype(str).str.strip().str.upper()
played["target"] = played["result"].map(label_map)


for c in ["gf","ga"]:
    if c in played.columns:
        played[c] = pd.to_numeric(played[c], errors="coerce")


teams = pd.concat([played["team"], played["opponent"], full["team"], full["opponent"]]).dropna().unique()
team_codes = {t:i for i,t in enumerate(sorted(teams))}

for df in [played, full]:
    df["team_code"] = df["team"].map(team_codes)
    df["opp_code"]  = df["opponent"].map(team_codes)


def add_form_block(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    g["gf_roll"]   = g["gf"].rolling(3, closed="left").mean()
    g["ga_roll"]   = g["ga"].rolling(3, closed="left").mean()
    g["points"]    = g["target"].map({0:0, 1:1, 2:3})
    g["form_roll"] = g["points"].rolling(3, closed="left").mean()
    return g

blocks = []
for team, g in played.groupby("team", sort=False):
    g = g.copy()
    g["team"] = team  
    blocks.append(add_form_block(g))
played = pd.concat(blocks, ignore_index=True)


ppg = (
    played.assign(points=played["target"].map({0:0,1:1,2:3}))
          .groupby("team", as_index=False)["points"].mean()
          .rename(columns={"points":"ppg"})
)

played = played.merge(ppg, on="team", how="left")
played = played.merge(
    ppg.rename(columns={"team":"opponent","ppg":"opp_ppg"}),
    on="opponent", how="left"
)

ppg_mean = ppg["ppg"].mean()
for c in ["ppg","opp_ppg"]:
    played[c] = played[c].fillna(ppg_mean)


need_cols = ["target","gf_roll","ga_roll","form_roll","ppg","opp_ppg"]
train_df = played.dropna(subset=need_cols).copy()

features = ["team_code","opp_code","home","hour","day_num","month",
            "gf_roll","ga_roll","form_roll","ppg","opp_ppg"]

X_all = train_df[features]
y_all = train_df["target"].astype(int)


train_df = train_df.sort_values("date")
cut = int(len(train_df) * 0.75) if len(train_df) > 0 else 0
tr, va = train_df.iloc[:cut], train_df.iloc[cut:]

X_tr, y_tr = tr[features], tr["target"].astype(int)
X_va, y_va = va[features], va["target"].astype(int)

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=6,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

if len(tr) > 0 and len(va) > 0:
    rf.fit(X_tr, y_tr)
    va_preds = rf.predict(X_va)
    print("Validation Accuracy:", round(accuracy_score(y_va, va_preds), 3))
    print("Confusion Matrix (rows=actual, cols=pred; 0=L,1=D,2=W):\n",
          confusion_matrix(y_va, va_preds, labels=[0,1,2]))
else:
    rf.fit(X_all, y_all)
    print("Not enough data for time-split validation; trained on all played matches.")


rf.fit(X_all, y_all)


last_form = (
    train_df.groupby("team", as_index=False)[["gf_roll","ga_roll","form_roll"]]
            .last()
    if len(train_df) > 0
    else played.groupby("team", as_index=False)[["gf_roll","ga_roll","form_roll"]].last()
)

future = full.copy()
future = future.merge(last_form, on="team", how="left")
future = future.merge(ppg, on="team", how="left")
future = future.merge(
    ppg.rename(columns={"team":"opponent","ppg":"opp_ppg"}),
    on="opponent", how="left"
)

for c in ["gf_roll","ga_roll","form_roll"]:
    future[c] = future[c].fillna(train_df[c].mean() if len(train_df) else played[c].mean())
for c in ["ppg","opp_ppg"]:
    future[c] = future[c].fillna(ppg_mean)

X_future = future[features]
future_preds = rf.predict(X_future)
future_proba = rf.predict_proba(X_future)

inv_label = {0:"L", 1:"D", 2:"W"}
future["predicted_result"] = [inv_label[p] for p in future_preds]

proba_df = pd.DataFrame(
    future_proba,
    columns=["predicted_L","predicted_D","predicted_W"],
    index=future.index
)
future = pd.concat([future, proba_df], axis=1)

cols_out = ["date","time","team","opponent","predicted_result",
            "predicted_L","predicted_D","predicted_W"]
future[cols_out].to_csv("season_preds.csv", index=False)

if len(tr) > 0 and len(va) > 0:
    cm = confusion_matrix(y_va, va_preds, labels=[0,1,2])
    pd.DataFrame(cm, index=["Actual_L","Actual_D","Actual_W"],
                    columns=["Pred_L","Pred_D","Pred_W"]).to_csv("confusion_matrix.csv")

pd.DataFrame({"feature": features, "importance": rf.feature_importances_}) \
  .sort_values("importance", ascending=False) \
  .to_csv("feature_importances.csv", index=False)

print("Saved: season_preds.csv", "(and confusion_matrix.csv)" if len(tr)>0 and len(va)>0 else "", "feature_importances.csv")
