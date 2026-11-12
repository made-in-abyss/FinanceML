
##   _____ . _____    |                        ______  ._____
##   |     | |    \   |       /\     |\    |  /        |
##   |____ | | __ /   |      /  \    | \   | |         |
##   |     | |   \    |     /____\   |  \  | |         |_____
##   |     | |    \   |    /      \  |   \ | |         |
##   |     | |     \  |__ /        \ |    \|  \______  |_____
##   github.com/made-in-abyss
from train import InitModel
from train import FetchModel
from run import token
archetypes = {
    1: "sniper",
    2: "berserker",
    3: "tank",
    4: "stoic_monk",
    5: "chaos_master",
    6: "flash_scalper",
    7: "deep_hunter",
    8: "equalizer",
    9: "feature_abuser",
    10: "grandmaster",
    11: "wtf",
}
description = {
    "sniper": "very filtering, hates noise. (TESTED: BAD)",
    "berserker": "takes trades recklessly, non linear. (TESTED: BAD)",
    "tank": "safe but stupid.",
    "stoic_monk": "bigger capacity and learns good, but slow. (2ND BEST)",
    "chaos_master": "high randomness in data, builds crazy trees.",
    "flash_scalper": "fast but too adaptive",
    "deep_hunter": "goes deeper.",
    "equalizer": "the default model, all rounded. (TESTED: DECENT) (DEFAULT)",
    "feature_abuser": "spams features. (3RD BEST)",
    "grandmaster": "textbook warrior. (1ST BEST)",
    "wtf": "i dont know",
}

t = int(input("1 to initialize a new model.\n2 to load your own model.\ntype: "))
if t == 1:
    agree = 2
    type = 0
    while(agree != 1):
        name = str(input("what is your model name?: ")).strip(' ').strip('/')
        agree = int(input(f"final name: {name}.json\n 1 yes, 2 no: "))
    try:
        for i in range(1,12):
            print(f"{i}: {archetypes[i]} - {description[archetypes[i]]}")
        while(type<=0 or type>=12):
            type = int(input("please select your model preset (to add your own, check train.py): "))
        model = InitModel(name,archetypes[type])
    except Exception:
        print("something's wrong idk.")
        exit(0)
elif t == 2:
    agree = 2
    while(agree != 1):
        name = str(input("what is your model name?: ")).strip(' ').strip('/')
        agree = int(input(f"final name: {name}.json\n 1 yes, 2 no: "))
    try:
        model = FetchModel(name)
    except Exception:
        print("please type correct name, or else train it.")
        exit(0)
while(True):
    token(True,False,model, dt=None)