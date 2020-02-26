import numpy as np

# Open data files:
train = open("train.csv")
train = train.read()
train = train.split("\n")
train.pop(0)

test = open("test.csv")
test = test.read()
test = test.split("\n")
test.pop(0)

# Init parameters:
d = 10  # dimension of data points
n = len(train)  # number of data points
m = 5  # number of iterations
w = np.zeros(d)
eta = 0.0000001  # learning rate

# Do gradient descent:
for i in range(m):
    grad = np.zeros(d)
    MSE = 0
    for vec in train:
        temp = np.fromstring(vec, dtype=float, sep=',')
        y_i = temp[1]
        x_i = temp[2:d+2]
        y_pred = np.dot(w, x_i)
        res = y_i - y_pred
        MSE += res**2 / n
        grad -= 2 / n * res * x_i
    w -= eta * grad
    print("MSE i=", i, ":", MSE)

print("Final w: ", w)

# Predict y on test data:
for vec in test:
    temp = np.fromstring(vec, dtype=float, sep=',')
    x_i = temp[1:d+1]
    y_pred = np.dot(w, x_i)
    print(y_pred)
    # TODO: write y_pred's to output file
