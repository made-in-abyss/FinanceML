
##   _____ . _____    |                        ______  ._____
##   |     | |    \   |       /\     |\    |  /        |
##   |____ | | __ /   |      /  \    | \   | |         |
##   |     | |   \    |     /____\   |  \  | |         |_____
##   |     | |    \   |    /      \  |   \ | |         |
##   |     | |     \  |__ /        \ |    \|  \______  |_____
##   github.com/made-in-abyss
##   THIS IS THE TRAINING SECTION, ONLY CONFIGURE IF YOU'RE SMART
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from run import token
import yfinance as yf
import pandas as pd
archetypes = {
    #conservative, filters noisy trades
    "sniper": {
        "n_estimators": 100, "learning_rate": 0.02, "max_depth": 3,
        "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 8,
        "reg_lambda": 3.0, "tree_method": "hist"
    },
    #all-in, rush
    "berserker": {
        "n_estimators": 200, "learning_rate": 0.06, "max_depth": 6,
        "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1,
        "reg_lambda": 0.1, "tree_method": "hist"
    },
    #very safe, but very stupid
    "tank": {
        "n_estimators": 120, "learning_rate": 0.015, "max_depth": 2,
        "subsample": 0.6, "colsample_bytree": 0.6, "min_child_weight": 10,
        "reg_lambda": 5.0, "tree_method": "hist"
    },
    #learn patiently
    "stoic_monk": {
        "n_estimators": 300, "learning_rate": 0.01, "max_depth": 4,
        "subsample": 0.85, "colsample_bytree": 0.8, "min_child_weight": 5,
        "reg_lambda": 2.0, "tree_method": "hist"
    },
    #chaotic data
    "chaos_master": {
        "n_estimators": 150, "learning_rate": 0.035, "max_depth": 5,
        "subsample": 0.65, "colsample_bytree": 0.5, "min_child_weight": 3,
        "reg_lambda": 1.0, "tree_method": "hist"
    },
    #fast to change
    "flash_scalper": {
        "n_estimators": 60, "learning_rate": 0.08, "max_depth": 3,
        "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 2,
        "reg_lambda": 0.5, "tree_method": "hist"
    },
    #deep thinker
    "deep_hunter": {
        "n_estimators": 180, "learning_rate": 0.03, "max_depth": 7,
        "subsample": 0.75, "colsample_bytree": 0.85, "min_child_weight": 4,
        "reg_lambda": 1.5, "tree_method": "hist"
    },
    #equality
    "equalizer": {
        "n_estimators": 100, "learning_rate": 0.03, "max_depth": 4,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
        "reg_lambda": 2.0, "tree_method": "hist"
    },
    #feature spammer (i am using this)
    "feature_abuser": {
        "n_estimators": 150, "learning_rate": 0.03, "max_depth": 5,
        "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1,
        "reg_lambda": 0.5, "tree_method": "hist"
    },
    #paper certificate
    "grandmaster": {
        "n_estimators": 220, "learning_rate": 0.025, "max_depth": 4,
        "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 4,
        "reg_lambda": 1.8, "tree_method": "hist"
    },
    #wtf
    "wtf": {
        "n_estimators": 500, "learning_rate": 0.075, "max_depth": 5,
        "subsample": 1, "colsample_bytree": 1, "min_child_weight": 7,
        "reg_lambda": 3.0, "tree_method": "hist"
    }
}
STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA",
    "INTC", "PFE", "BA", "NKE", "PYPL", "DIS", "WBA",
    "COST", "WMT", "JNJ", "PG", "KO", "PEP", "MCD",
    "XOM", "CAT", "DE", "FCX", "UNP",
    "JPM", "BAC", "GS", "V", "BLK"
]
def get_model_preset(preset_name="equalizer"):
    params = archetypes.get(preset_name, archetypes["equalizer"])
    return XGBClassifier(
        **params,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
def InitModel(name, type):
    model = get_model_preset(type)
    raw_data = yf.download(STOCKS, period="max",group_by="ticker",threads=True)
    for i in STOCKS:
        dt = raw_data[i].dropna(how="all").copy()
        if isinstance(dt.columns, pd.MultiIndex):
            dt.columns = dt.columns.get_level_values(0)
        token(False,True,model,dt)
    print("Finished training the model")
    print(f"Now named: {name}.json")
    model.save_model(f"{name}.json")
    return model
def FetchModel(name):
    model = XGBClassifier()  
    model.load_model(f"{name}.json")
    return model