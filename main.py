
##   _____ . _____    |                        ______  ._____
##   |     | |    \   |       /\     |\    |  /        |
##   |____ | | __ /   |      /  \    | \   | |         |
##   |     | |   \    |     /____\   |  \  | |         |_____
##   |     | |    \   |    /      \  |   \ | |         |
##   |     | |     \  |__ /        \ |    \|  \______  |_____
##   github.com/made-in-abyss
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import yfinance as yf
import pandas as pd
model = XGBClassifier(
    n_estimators=50, 
    max_depth=3, 
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)
booster = None
tt_accu = 0
def fetch(pretick):
    if pretick != False:
        ticker = pretick
    else:
        ticker = str(input("input your ticker: "))
    try:
        df = yf.download(ticker, period="max", interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        print(Exception)
        return
def token(pretick,output_mode,train):
    dt = fetch(pretick)
    if dt is None or dt.empty or len(dt) < 50:
        print(f"skipped {pretick}")
        return
    global booster
    global tt_accu
    daily = dt["Close"].pct_change(1)
    dt["median_5d"] = daily.rolling(5).median()
    dt["median_10d"] = daily.rolling(10).median()
    dt["volatility_5d"] = daily.rolling(5).std()
    dt["volatility_10d"] = daily.rolling(10).std()
    ema_9   = dt["Close"].ewm(span=9, adjust=False).mean()
    ema_21  = dt["Close"].ewm(span=21, adjust=False).mean()
    ema_55  = dt["Close"].ewm(span=55, adjust=False).mean()
    ema_100 = dt["Close"].ewm(span=100, adjust=False).mean()

    dt["dist_ema_9"]   = (dt["Close"] /ema_9) -1.0
    dt["dist_ema_21"]  = (dt["Close"] /ema_21) -1.0
    dt["dist_ema_55"]  = (dt["Close"] /ema_55) -1.0
    dt["dist_ema_100"] = (dt["Close"] /ema_100) -1.0

    dt["ema_cross_fast"] = (ema_9 /ema_21) -1.0
    dt["ema_cross_slow"] = (ema_21 /ema_55) -1.0

    delta = dt["Close"].diff()
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    rs = gain / (loss+1e-9)
    dt["rsi"] = 100 - (100 /(1 + rs))
    dt["future_return_5d"] = dt["Close"].shift(-5) /dt["Close"] - 1.0
    dt["target"] =(dt["future_return_5d"] >0).astype(int)
    dt_clean = dt.dropna()
    features = ["median_5d","median_10d","volatility_5d","volatility_10d","rsi",
                "dist_ema_9","dist_ema_21","dist_ema_55","dist_ema_100",
                "ema_cross_fast","ema_cross_slow"] #you can try to include more features idk
    X = dt_clean[features]
    y = dt_clean["target"]
    split_point = int(len(dt_clean)*0.8) #80% train, 20% test
    X_train = X.iloc[:split_point] 
    y_train = y.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_test = y.iloc[split_point:]
    if train == True:
        model.fit(X_train, y_train, xgb_model=booster)
        booster =model.get_booster()
        model.n_estimators += 50
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    tt_accu += accuracy*100
    if (output_mode == True):
        print(f"accuracy: {accuracy*100}%")
        counts = pd.Series(y_pred).value_counts().to_dict()
        print(f"breakdown: {counts.get(1, 0)} UP vs {counts.get(0, 0)} DOWN")
        print("care:")
        for feat,imp in zip(features, model.feature_importances_):
            print(f"{feat:15s}:{imp * 100}%")

PRETRAIN_50 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "CRM",
    "INTC", "PFE","BA", "DIS", "PYPL", "SNAP", "WBA", "NKE", "PARA", "RIVN",
    "JNJ","PG","KO", "PEP", "WMT", "COST", "MCD", "UNH", "BMY", "CL",
    "XOM", "CVX","CAT", "DE", "GE", "FCX", "NUE", "F", "D", "VZ",
    "JPM", "BAC","GS", "MS", "BLK", "SCHW", "C", "AXP", "SPG", "AMT"
]#ai gen list lol
PRETRAIN_30 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "CRM",
    "JNJ","PG","KO", "PEP", "WMT", "COST", "MCD", "UNH", "BMY", "CL",
    "JPM", "BAC","GS", "MS", "BLK", "SCHW", "C", "AXP", "SPG", "AMT"
] #what is wba, woman basketball association?
PRETRAIN_40 = [
    # 1. Mega-Cap Tech & Growth
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "CRM",
    # 2. Defensive & Consumer Staples
    "JNJ", "PG", "KO", "PEP", "WMT", "COST", "MCD", "UNH", "BMY", "CL",
    # 3. Financials, Banking & Real Estate
    "JPM", "BAC", "GS", "MS", "BLK", "SCHW", "C", "AXP", "SPG", "AMT",
    # 4. Prestigious Blue-Chip Moats & Global Leaders
    "BRK-B", "V", "MA", "LLY", "TSM", "ASML", "HD", "UNP", "ABT", "LIN"
]#again, AI selected
for i in PRETRAIN_40:
    print(i)
    token(i,True,True)
print(f"total accuracy: {tt_accu/40}% ")
while(input):
    token(False,True,False)