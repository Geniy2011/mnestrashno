from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score


from .data import CLIENT_FEATURES

CLIENT_FEATURE_COLUMNS = ["branch_id", *CLIENT_FEATURES]
CLIENT_CATEGORICAL = ["branch_id", "region_code"]
CLIENT_NUMERIC = [c for c in CLIENT_FEATURE_COLUMNS if c not in CLIENT_CATEGORICAL]

from .data import CREDIT_FEATURES

CREDIT_FEATURE_COLUMNS = [*CREDIT_FEATURES]
CREDIT_CATEGORICAL = []
CREDIT_NUMERIC = [c for c in CREDIT_FEATURE_COLUMNS if c not in CREDIT_CATEGORICAL]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m bank_ctf", description="Green Bank Pipeline CTF participant CLI")
    sub = p.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--task", choices=["inflow", "risk"], required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    pred = sub.add_parser("predict")
    pred.add_argument("--task", choices=["inflow", "risk"], required=True)
    pred.add_argument("--model-dir", required=True)
    pred.add_argument("--data-dir", required=True)
    pred.add_argument("--output", required=True)
    exp = sub.add_parser("experiment")
    exp.add_argument("--config", required=True)
    exp.add_argument("--data-dir", required=True)
    exp.add_argument("--runs-dir", required=True)
    exp.add_argument("--run-id", required=True)
    return p


def read_config(path: str | Path) -> dict[str, Any]:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config {path}: {exc}") from exc
    if cfg.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")
    return cfg


def make_inflow_model(cfg: dict[str, Any]) -> Pipeline:
    model_spec = cfg.get("inflow_model", cfg.get("model", {}))
    params = dict(model_spec.get("params", {}))
    seed = int(cfg.get("seed", 0))
    kind = model_spec.get("kind", "hist_gradient_boosting")
    if kind == "hist_gradient_boosting":
        allowed = {"max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization", "min_samples_leaf"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unsupported inflow params: {sorted(unknown)}")
        estimator = HistGradientBoostingRegressor(random_state=seed, **params)
    elif kind == "ridge":
        allowed = {"alpha"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unsupported inflow params: {sorted(unknown)}")
        estimator = Ridge(random_state=seed, **params)
    elif kind == "catboost":
        start_cat_idx = len(CLIENT_NUMERIC)
        cat_indices = list(range(start_cat_idx, start_cat_idx + len(CLIENT_CATEGORICAL)))
        allowed = {"iterations", "learning_rate", "depth", "l2_leaf_reg", "border_count"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unsupported catboost params: {sorted(unknown)}")
        estimator = CatBoostRegressor(random_state=seed, verbose=0, cat_features=cat_indices, **params)
    else:
        raise ValueError(f"unsupported inflow model kind: {kind}")
    
    
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), CLIENT_NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CLIENT_CATEGORICAL),
    ])
    if kind == "catboost":
        preprocessor = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), CLIENT_NUMERIC),
            ("cat", SimpleImputer(strategy="constant", fill_value="missing"), CLIENT_CATEGORICAL),
        ])
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])

def make_risk_model(cfg: dict[str, Any]) -> Pipeline | CatBoostClassifier:
    model_spec = cfg.get("risk_model", cfg.get("model_risk", {}))
    params = dict(model_spec.get("params", {}))
    seed = int(cfg.get("seed", 0))
    kind = model_spec.get("kind", "catboost")

    if kind == "catboost":
        allowed = {"iterations", "learning_rate", "depth", "l2_leaf_reg", "border_count"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unsupported catboost_classifier params: {sorted(unknown)}")
        
        return CatBoostClassifier(
            random_state=seed,
            verbose=0,
            loss_function="Logloss",
            eval_metric="AUC",
            **params
        )
        
    elif kind == "logistic_regression":
        allowed = {"C", "penalty", "solver", "max_iter"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unsupported logistic_regression params: {sorted(unknown)}")
        
        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), CREDIT_NUMERIC),
        ])
        
        estimator = LogisticRegression(random_state=seed, **params)
        
        return Pipeline([("preprocess", preprocessor), ("model", estimator)])
        
    else:
        raise ValueError(f"unsupported risk model kind: {kind}")


def require_columns(df: pd.DataFrame, columns: list[str], file_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{file_name} missing columns: {missing}")


def train_inflow(config_path: str, data_dir: str, output_dir: str) -> None:
    cfg = read_config(config_path)
    if cfg.get("task") != "inflow":
        raise ValueError("train --task inflow requires config task='inflow'")
    data = Path(data_dir)
    train = pd.read_csv(data / "clients_train.csv")
    require_columns(train, ["client_id", "inflow_next_30d", *CLIENT_FEATURE_COLUMNS], "clients_train.csv")
    if not train["client_id"].is_unique:
        raise ValueError("clients_train.csv has duplicate client_id")
    model = make_inflow_model(cfg)
    model.fit(train[CLIENT_FEATURE_COLUMNS], train["inflow_next_30d"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": CLIENT_FEATURE_COLUMNS, "config": cfg}, out / "inflow_model.joblib")
    manifest = {"task": "inflow", "model_file": "inflow_model.joblib", "feature_columns": CLIENT_FEATURE_COLUMNS, "effective_config": cfg}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def predict_inflow(model_dir: str, data_dir: str, output: str) -> None:
    bundle = joblib.load(Path(model_dir) / "inflow_model.joblib")
    clients = pd.read_csv(Path(data_dir) / "clients_test.csv")
    require_columns(clients, ["client_id", *bundle["feature_columns"]], "clients_test.csv")
    if not clients["client_id"].is_unique:
        raise ValueError("clients_test.csv has duplicate client_id")
    preds = np.maximum(0, np.asarray(bundle["model"].predict(clients[bundle["feature_columns"]]), dtype=float))
    if not np.isfinite(preds).all():
        raise ValueError("model produced non-finite predictions")
    out = pd.DataFrame({"client_id": clients["client_id"], "predicted_inflow": preds})
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

def train_risk(config_path: str, data_dir: str, output_dir: str) -> None:

    cfg = read_config(config_path)
    data = Path(data_dir)

    clients_train = pd.read_csv(data / "clients_train.csv")
    require_columns(clients_train, ["client_id", "inflow_next_30d", *CLIENT_FEATURE_COLUMNS], "clients_train.csv")
    
    credit_train = pd.read_csv(data / "credit_train.csv")
    
    require_columns(credit_train, ["application_id", "client_id", "overdue_next_60d", *CREDIT_FEATURE_COLUMNS], "credit_train.csv")
    
    folds = pd.read_csv(data / "folds_group.csv")

    clients_train = pd.merge(clients_train, folds, on='client_id', how='inner')
    unique_folds = sorted(clients_train["fold_id"].unique())
    if "inner_folds" in cfg:
        unique_folds = unique_folds[:int(cfg["inner_folds"])]
    oof_predictions = np.zeros(len(clients_train))
   
    for fold in unique_folds:

        train_idx = clients_train["fold_id"] != fold
        val_idx = clients_train["fold_id"] == fold
        
        model = make_inflow_model(cfg)
        model.fit(
            clients_train.loc[train_idx, CLIENT_FEATURE_COLUMNS], 
            clients_train.loc[train_idx, "inflow_next_30d"]
        )
        
        fold_preds = model.predict(clients_train.loc[val_idx, CLIENT_FEATURE_COLUMNS])
        oof_predictions[val_idx] = fold_preds
    
    clients_train["predicted_inflow"] = np.maximum(0, oof_predictions)
   
    inflow_lookup = clients_train[["client_id", "predicted_inflow"]]
    credit_train = pd.merge(credit_train, inflow_lookup, on="client_id", how="inner")

    risk_model = make_risk_model(cfg)

    
    risk_model.fit(credit_train[[*CREDIT_FEATURE_COLUMNS,"predicted_inflow"]], credit_train["overdue_next_60d"])
    final_inflow_model = make_inflow_model(cfg)
    final_inflow_model.fit(clients_train[CLIENT_FEATURE_COLUMNS], clients_train["inflow_next_30d"])
   
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = {
        "inflow_model": final_inflow_model,
        "risk_model": risk_model,
        "feature_columns_inflow": CLIENT_FEATURE_COLUMNS,
        "feature_columns_risk": [*CREDIT_FEATURE_COLUMNS,"predicted_inflow"],
        "config": cfg
    }
    joblib.dump(bundle, out / "risk_bundle.joblib")
    
    manifest = {"task": "risk", "model_file": "risk_bundle.joblib", "effective_config": cfg}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

def predict_risk(model_dir: str, data_dir: str, output: str) -> None:
    bundle = joblib.load(Path(model_dir) / "risk_bundle.joblib")
    data = Path(data_dir)
    
    clients = pd.read_csv(data / "clients_test.csv")
    credit = pd.read_csv(data / "credit_test.csv")
    require_columns(clients, ["client_id", *bundle["feature_columns_inflow"]], "clients_test.csv")
   
    credit_base_features = [c for c in bundle["feature_columns_risk"] if c != "predicted_inflow"]
    require_columns(credit, ["application_id", "client_id", *credit_base_features], "credit_test.csv")
    
    if not clients["client_id"].is_unique:
        raise ValueError("clients_test.csv has duplicate client_id")
        
    clients["predicted_inflow"] = np.maximum(
        0, 
        np.asarray(bundle["inflow_model"].predict(clients[bundle["feature_columns_inflow"]]), dtype=float)
    )

    inflow_lookup = clients[["client_id", "predicted_inflow"]]
    credit = pd.merge(credit, inflow_lookup, on="client_id", how="inner")
    
    risk_preds = np.asarray(bundle["risk_model"].predict_proba(credit[bundle["feature_columns_risk"]])[:, 1], dtype=float)
    
    if not np.isfinite(risk_preds).all():
        raise ValueError("model produced non-finite predictions")
        
    credit["probability_overdue"] = risk_preds
    
    out_risk = pd.DataFrame({
        "application_id": credit["application_id"], 
        "probability_overdue": credit["probability_overdue"]
    })

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_risk.to_csv(path, index=False)

def run_experiment(config_path: str, data_dir: str, runs_dir: str, run_id: str) -> None:
    cfg = read_config(config_path)
    task = cfg.get("task")
    
    if task not in ["inflow", "risk"]:
        raise ValueError(f"unsupported experiment task: {task}")

    runs_folder = Path(runs_dir)
    runs_folder.mkdir(parents=True, exist_ok=True)
    runs_file = runs_folder / "runs.jsonl"
    
    if runs_file.is_file():
        with open(runs_file, mode="r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    run_entry = json.loads(line)
                    if run_entry.get("run_id") == run_id:
                        raise ValueError(f"run_id '{run_id}' already exists")

    data = Path(data_dir)
    clients_train = pd.read_csv(data / "clients_train.csv")
    folds = pd.read_csv(data / "folds_random.csv")
    if cfg.get("split_version") == "group_v1":
        folds = pd.read_csv(data / "folds_group.csv")
    clients_train = pd.merge(clients_train, folds, on='client_id', how='inner')
    unique_folds = sorted(clients_train["fold_id"].unique())

    if task == "inflow":
        oof_preds = np.zeros(len(clients_train))
        for fold in unique_folds:
            train_idx = clients_train["fold_id"] != fold
            val_idx = clients_train["fold_id"] == fold
            
            model = make_inflow_model(cfg)
            model.fit(clients_train.loc[train_idx, CLIENT_FEATURE_COLUMNS], clients_train.loc[train_idx, "inflow_next_30d"])
            oof_preds[val_idx] = model.predict(clients_train.loc[val_idx, CLIENT_FEATURE_COLUMNS])
            
        oof_preds = np.maximum(0, oof_preds)
        metric_name = "MAE"
        metric_value = float(mean_absolute_error(clients_train["inflow_next_30d"], oof_preds))
        
    elif task == "risk":  
            
        credit_train = pd.read_csv(data / "credit_train.csv")
        require_columns(credit_train, ["application_id", "client_id", "overdue_next_60d", *CREDIT_FEATURE_COLUMNS], "credit_train.csv")
        credit_train = pd.merge(credit_train, folds, on='client_id', how='inner')
        outer_risk_preds = []
        outer_risk_targets = []
        
        if "inner_folds" in cfg:
            unique_folds = unique_folds[:int(cfg["inner_folds"])]
        
        for outer_fold in unique_folds:
            
            c_train_mask = clients_train["fold_id"] != outer_fold
            c_val_mask = clients_train["fold_id"] == outer_fold
           
            cr_train_mask = credit_train["fold_id"] != outer_fold
            cr_val_mask = credit_train["fold_id"] == outer_fold
            
            c_train = clients_train[c_train_mask].copy()
            c_val = clients_train[c_val_mask].copy()
            cr_train = credit_train[cr_train_mask].copy()
            cr_val = credit_train[cr_val_mask].copy()

            inner_folds = sorted(c_train["fold_id"].unique())
            inner_oof_inflow = np.zeros(len(c_train))
            
            for inner_fold in inner_folds:
                sub_train_mask = c_train["fold_id"] != inner_fold
                sub_val_mask = c_train["fold_id"] == inner_fold
                
                inner_inflow_model = make_inflow_model(cfg)
                inner_inflow_model.fit(
                    c_train.loc[sub_train_mask, CLIENT_FEATURE_COLUMNS],
                    c_train.loc[sub_train_mask, "inflow_next_30d"]
                )
                inner_oof_inflow[sub_val_mask] = inner_inflow_model.predict(
                    c_train.loc[sub_val_mask, CLIENT_FEATURE_COLUMNS]
                )
            
            c_train["predicted_inflow"] = np.maximum(0, inner_oof_inflow)
            
            outer_inflow_model = make_inflow_model(cfg)
            outer_inflow_model.fit(c_train[CLIENT_FEATURE_COLUMNS], c_train["inflow_next_30d"])
            
            c_val["predicted_inflow"] = np.maximum(0, outer_inflow_model.predict(c_val[CLIENT_FEATURE_COLUMNS]))
            
            cr_train = pd.merge(cr_train, c_train[["client_id", "predicted_inflow"]], on="client_id", how="inner")
            cr_val = pd.merge(cr_val, c_val[["client_id", "predicted_inflow"]], on="client_id", how="inner")
            
            feature_cols_risk = [*CREDIT_FEATURE_COLUMNS, "predicted_inflow"]
            outer_risk_model = make_risk_model(cfg)
            outer_risk_model.fit(cr_train[feature_cols_risk], cr_train["overdue_next_60d"])
            
            fold_risk_preds = outer_risk_model.predict_proba(cr_val[feature_cols_risk])[:, 1]
            
            outer_risk_preds.extend(fold_risk_preds)
            outer_risk_targets.extend(cr_val["overdue_next_60d"].tolist())
            
        metric_name = "AUC"
        metric_value = float(roc_auc_score(outer_risk_targets, outer_risk_preds))

    log_entry = {
        "run_id": run_id,
        "task": task,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "effective_config": cfg
    }
    
    with open(runs_file, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    print(f"Experiment {run_id} finished successfully. {metric_name}: {metric_value:.4f}")

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train" and args.task == "inflow":
            train_inflow(args.config, args.data_dir, args.output_dir)
            return 0
        if args.command == "train" and args.task == "risk":
            train_risk(args.config, args.data_dir, args.output_dir)
            return 0
        if args.command == "predict" and args.task == "inflow":
            predict_inflow(args.model_dir, args.data_dir, args.output)
            return 0
        if args.command == "predict" and args.task == "risk":
            predict_risk(args.model_dir, args.data_dir, args.output)
            return 0
        if args.command == "experiment":
            run_experiment(args.config, args.data_dir, args.runs_dir, args.run_id)
            return 0

        print(
            f"TODO: implement {args.command} --task {getattr(args, 'task', '<from config>')} beyond the stage-1 inflow baseline.",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
