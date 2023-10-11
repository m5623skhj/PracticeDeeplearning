import sys, os
import numpy as np
from PIL import Image
import pickle

sys.path.append(os.getcwd())
from mnist import load_mnist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def img_show(img):
    image = Image.fromarray(np.uint8(img))
    image.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, X):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['b1'], network['b2'], network['b3']

    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    A3 = np.dot(Z2, W3) + B3
    Y = softmax(A3)

    return Y

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_bat = x[i:i+batch_size]
    y_bat = predict(network, x_bat)
    p = np.argmax(y_bat, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy : " + str(float(accuracy_cnt) / len(x)))