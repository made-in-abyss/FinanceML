
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

t = int(input("1 to initialize a new model.\n2 to load your own model.\ntype:"))
if t == 1:
    name = str(input("give the model a name: "))
    name.strip(' ').strip('/')
    print(f"final name: {name}")
    try:
        model = InitModel(name)
    except Exception:
        print("something's wrong idk.")
        exit(0)
elif t == 2:
    name = str(input("what is your model name?: "))
    name.strip(' ').strip('/')
    print(f"final name: {name}")
    try:
        model = FetchModel(name)
    except Exception:
        print("please type correct name, or else train it.")
        exit(0)
while(True):
    token(False,True,False,False,model)