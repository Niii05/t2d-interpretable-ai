# train.py
import argparse, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import optuna
from utils import ensure_features

def build_preprocessor(df):
    cat_cols = [c for c in ["AgeGroup","BMICategory"] if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols + ["Outcome"]]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    return pre, num_cols, cat_cols

def grid_search_models(X_train, y_train, pre):
    rf = Pipeline([("pre", pre),
                   ("clf", RandomForestClassifier(random_state=42))])
    rf_grid = {"clf__n_estimators":[200,400],
               "clf__max_depth":[None,8,12],
               "clf__min_samples_split":[2,5,10]}

    lr = Pipeline([("pre", pre),
                   ("clf", LogisticRegression(max_iter=2000, n_jobs=None))])
    lr_grid = {"clf__C":[0.1,1,3],
               "clf__penalty":["l2"],
               "clf__solver":["lbfgs"]}

    xgb = Pipeline([("pre", pre),
                    ("clf", XGBClassifier(
                        random_state=42, n_estimators=400, tree_method="hist", eval_metric="logloss"))])
    xgb_grid = {"clf__max_depth":[3,5,7],
                "clf__learning_rate":[0.03,0.1],
                "clf__subsample":[0.8,1.0],
                "clf__colsample_bytree":[0.8,1.0]}

    results = []
    for name, pipe, grid in [("RandomForest", rf, rf_grid),
                             ("LogReg", lr, lr_grid),
                             ("XGBoost", xgb, xgb_grid)]:
        gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring="roc_auc", n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        results.append((name, gs.best_score_, gs.best_params_, gs.best_estimator_))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[0], results

def optuna_xgb(X_train, y_train, pre, n_trials=25):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "random_state": 42, "eval_metric": "logloss", "tree_method": "hist",
        }
        pipe = Pipeline([("pre", pre), ("clf", XGBClassifier(**params))])
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        aucs = []
        for tr, va in cv.split(X_train, y_train):
            pipe.fit(X_train.iloc[tr], y_train.iloc[tr])
            p = pipe.predict_proba(X_train.iloc[va])[:,1]
            aucs.append(roc_auc_score(y_train.iloc[va], p))
        return float(np.mean(aucs))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_pipe = Pipeline([("pre", pre), ("clf", XGBClassifier(**best_params))])
    best_pipe.fit(X_train, y_train)
    return best_pipe, best_params, study.best_value

def main(csv_path, use_optuna=False, trials=25, outcome_col="Outcome"):
    df = pd.read_csv(csv_path)

    # Drop unwanted columns
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "Is_High_Risk" in df.columns:
        df = df.rename(columns={"Is_High_Risk": "IsHighRisk"})

    df = ensure_features(df)

    X = df.drop(columns=[outcome_col])
    y = df[outcome_col].astype(int)

    pre, num_cols, cat_cols = build_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if use_optuna:
        best_estimator, best_params, best_auc = optuna_xgb(X_train, y_train, pre, n_trials=trials)
        model = best_estimator
        print("Optuna XGB best AUC (CV):", round(best_auc, 4))
        print("Best params:", best_params)
    else:
        (name, best_cv_auc, best_params, model), all_results = grid_search_models(X_train, y_train, pre)
        print("Best by GridSearchCV:", name, "CV AUC:", round(best_cv_auc, 4))
        print("Params:", best_params)

    # Evaluate
    proba = model.predict_proba(X_test)[:,1]
    preds = (proba >= 0.5).astype(int)
    print("\nTest ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    print(classification_report(y_test, preds, digits=4))

    # Save artifacts
    import os
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")
    with open("artifacts/feature_names.json","w") as f:
         json.dump(list(X.columns), f)
    print("\nSaved: artifacts/model.pkl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="ai_final_dataset.csv")
    ap.add_argument("--optuna", type=int, default=0)
    ap.add_argument("--trials", type=int, default=25)
    args = ap.parse_args()
    main(args.csv, bool(args.optuna), args.trials)
