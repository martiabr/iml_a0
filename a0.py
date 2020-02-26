import numpy as np
#from sklearn.metrics import mean_squared_error

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
d = 10
n = len(train)
w = np.ones(d)
diff = 10
threshold = 1
eta = 0.5

while diff > threshold:
    grad = np.zeros(d)
    for vec in train:
        temp = np.fromstring(vec, dtype=float, sep=',')
        y_i = temp[1]
        x_i = temp[2:d+2]
        #print(y_i, np.dot(w, x_i))
        res = y_i - np.dot(w, x_i)
        print(res)
        grad += res * x_i
        #print(grad)

    grad = 2*grad
    #print(grad)
    wold = w
    w = w - eta*grad
    diff = np.linalg.norm(w - wold)

j = 0
y_pred = np.zeros(n)
y = np.zeros(n)

for vec in train:
    temp = np.fromstring(vec, dtype=float, sep=',')
    y[j] = temp[1]
    x_i = temp[2:d+2]
    y_pred[j] = np.dot(w,x_i)
    j = j + 1

#RMSE = mean_squared_error(y, y_pred)**0.5


