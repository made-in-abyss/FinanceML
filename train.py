
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
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=2.0,
    random_state=42,
    eval_metric="logloss"
)
PRETRAIN_50 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "CRM",
    "INTC", "PFE","BA", "DIS", "PYPL", "SNAP", "WBA", "NKE", "PARA", "RIVN",
    "JNJ","PG","KO", "PEP", "WMT", "COST", "MCD", "UNH", "BMY", "CL",
    "XOM", "CVX","CAT", "DE", "GE", "FCX", "NUE", "F", "D", "VZ",
    "JPM", "BAC","GS", "MS", "BLK", "SCHW", "C", "AXP", "SPG", "AMT"
]
PRETRAIN_30 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "CRM",
    "JNJ","PG","KO", "PEP", "WMT", "COST", "MCD", "UNH", "BMY", "CL",
    "JPM", "BAC","GS", "MS", "BLK", "SCHW", "C", "AXP", "SPG", "AMT"
]
PRETRAIN_40 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "CRM",
    "JNJ", "PG", "KO", "PEP", "WMT", "COST", "MCD", "UNH", "BMY", "CL",
    "JPM", "BAC", "GS", "MS", "BLK", "SCHW", "C", "AXP", "SPG", "AMT",
    "BRK-B", "V", "MA", "LLY", "TSM", "ASML", "HD", "UNP", "ABT", "LIN"
]
def InitModel(name):
    for i in PRETRAIN_50:
        token(i,False,True,model)
    print("Finished training the model")
    print(f"Now named: {name}")
    model.save_model(f"{name}.json")
    return model
def FetchModel(name):
    model.load_model(f"{name}.json")
    return model