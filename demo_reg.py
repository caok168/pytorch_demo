import torch
import numpy as np
import re

# data
# net
# loss
# optimiter
# training
# test

data = []
ff = open("./datasets/housing.data").readlines()
for item in ff:
    out = re.sub(r"\s{2,}", " ", item).strip()
    print(out)
    data.append(out.split(" "))
data = np.array(data).astype(np.float)
print(data.shape)

Y = data[:-1]
X = data[:, 0:-1]

X_train = X[0:496, ...]
Y_train = X[0:496, ...]
X_test = X[496:, ...]
Y_test = X[496:, ...]

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
