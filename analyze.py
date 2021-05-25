import pickle
import sys

sys.path.append(
    "/private/home/apacchiano/OnlineBias"
)

with open("fnr_dump.p", 'rb') as f:
    x = pickle.load(f)

#print("FPR + Norm")
#print(x[-1])

train = x[0]
for exp in train:
    print("Train FPR")
    print([y.fpr for y in exp])
    print("FnR")
    print([y.fnr for y in exp])
    print("Weight Norm")
    print([y.weight_norm for y in exp])

test = x[1]
for exp in test:
    print("Test FPR")
    print([y.fpr for y in exp])
    print("FnR")
    print([y.fnr for y in exp])
    print("Weight Norm")
    print([y.weight_norm for y in exp])
