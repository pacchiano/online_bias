import pickle
import sys

sys.path.append(
    "/private/home/apacchiano/OnlineBias"
)

with open("fnr_dump.p", 'rb') as f:
    x = pickle.load(f)

print("FPR + Norm")
print(x[-1])
