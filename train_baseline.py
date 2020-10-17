import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, GRUCell, CuDNNLSTM, BatchNormalization, RNN
from sklearn import preprocessing
from numpy import array
from input_data_baseline import preprocess_data, load_dow_price_data
from utils import calculate_laplacian, evaluation, get_trend, avg_relative_error, get_vague_trend
from tensorflow.keras.layers import AbstractRNNCell
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid, tanh
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
from configparser import SafeConfigParser

config_file_addr = "./config.ini"
config = SafeConfigParser()
config.read(config_file_addr)


def main(data_addr, adj_addr, s_index, lr, n_neurons, seq_len, n_epochs, batch_size, n_off):
    # hyperperameter
    data, adj = load_dow_price_data(data_addr, adj_addr)
    labels = data[:, s_index]
    assert(data.shape[1] == adj.shape[0])

    assert(n_off >= 0)
    if n_off != 0:
        data = data[:-n_off]
        labels = labels[n_off:]

    train_rate = 0.8
    pre_len = 1
    time_len = data.shape[0]
    n_gcn_nodes = data.shape[1]

    X_train, y_train, X_test, y_test, pre_y_test = preprocess_data(
        data, labels, time_len, train_rate, seq_len, pre_len)

    X_train = X_train[:, :, s_index]
    X_test = X_test[:, :, s_index]
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    cell = GRUCell(n_neurons)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model = Sequential()
    model.add(RNN(cell, input_shape=(seq_len, 1)))
    model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer="adam")
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.fit(X_train, y_train, epochs=n_epochs, validation_split=0.1,
              batch_size=batch_size, verbose=2)

    # evaluate
    result = model.predict(X_test, batch_size=batch_size, verbose=0)

    actual_trend = get_trend(pre_y_test, y_test)
    predicted_trend = get_trend(pre_y_test, result)
    accuracy = accuracy_score(actual_trend, predicted_trend)

    actual_vague_trend = get_vague_trend(pre_y_test, y_test, th)
    predicted_vague_trend = get_vague_trend(pre_y_test, result, th)

    count = 0
    for i in range(len(actual_vague_trend)):
        if abs(actual_vague_trend[i]) == abs(predicted_vague_trend[i]):
            count += 1
        elif actual_vague_trend[i] < 0 and predicted_vague_trend[i] < 0:
            count += 1
    accuracy = count/len(actual_vague_trend)

    print("***********************")
    print("accuracy: ", accuracy)

    r2 = r2_score(y_test, result)
    print("r2: ", r2)
    rmse = sqrt(mean_squared_error(y_test, result))
    print("rmse: ", rmse)
    mae = mean_absolute_error(y_test, result)
    print("mae: ", mae)
    re = avg_relative_error(y_test, result)
    print("re: ", re)

    with open("result_baseline.txt", "a") as f:
        f.write("s_index: " + str(s_index))
        f.write("\n")
        f.write("accuracy: " + str(accuracy))
        f.write("\n")
        f.write("r2: " + str(r2))
        f.write("\n")
        f.write("rmse: " + str(rmse))
        f.write("\n")
        f.write("mae: " + str(mae))
        f.write("\n")
        f.write("re: " + str(re))
        f.write("\n")
        f.write("\n")


data_addr = config["hyper"]["data_addr"]
adj_addr = config["hyper"]["adj_addr"]
s_index = int(config["hyper"]["s_index"])
lr = float(config["hyper"]["lr"])
n_neurons = int(config["hyper"]["n_neurons"])
seq_len = int(config["hyper"]["seq_len"])
n_epochs = int(config["hyper"]["n_epochs"])
batch_size = int(config["hyper"]["batch_size"])
n_off = int(config["hyper"]["n_off"])
th = float(config["hyper"]["th"])

main(data_addr, adj_addr, s_index, lr, n_neurons,
     seq_len, n_epochs, batch_size, n_off)
