import pickle
import sys

sys.path.append(
    "/private/home/apacchiano/OnlineBias"
)

with open("fnr_dump.p", 'rb') as f:
    x = pickle.load(f)

#print("FPR + Norm")
#print(x[-1])

# print(len(x[0]))
exp = x[-1]
# print(exp)
print("FPR")
print([y.fpr for y in exp])
print("FnR")
print([y.fnr for y in exp])
print("Weight Norm")
print([y.weight_norm for y in exp])
# for l in x:
#     for y in l:
#         print(y.weight_norm)
#


