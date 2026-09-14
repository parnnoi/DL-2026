import numpy as np
import pandas as pd
import asciichartpy

def MSE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def part_one():
    def training_gradient_descent(X, y, weights, learning_rate, epochs):
        history = []

        for _ in range(epochs):
            y_pred = X @ weights
            error = y_pred - y
            loss = MSE(y, y_pred)

            gradient = X.T @ error * (1 / len(y))

            weights -= learning_rate * gradient
            history.append(loss)

        return weights, history

    def prediction(X_test, y_test, weights, data_min, data_max):
        y_pred = X_test @ weights
        y_unnorm = y_pred * (data_max[-1] - data_min[-1]) + data_min[-1]

        y_true = y_test * (data_max[-1] - data_min[-1]) + data_min[-1]
        loss = MSE(y_true, y_unnorm)
        return y_unnorm, loss
    data = pd.read_csv("src/Lab4/data/insurance_charges.csv")
    
    np.set_printoptions(precision=5, suppress=True)

    data = data.to_numpy()
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    divider = data_max - data_min
    if np.any(divider == 0):
        divider[divider == 0] = 1  # Avoid division by zero
    data_norm = (data - data_min) / divider

    weights = np.zeros(data_norm.shape[1] - 1)  # Initialize weights for features (excluding target)
    learning_rate = 0.01
    epochs = 100

    weights, history = training_gradient_descent(data_norm[:, :-1], data_norm[:, -1], weights, learning_rate, epochs)
    print(f"{' Part I: Regression Model ':-^100}")
    print(f"{'Q1':->5}")
    print(f"Training completed. Final weights: {weights}, Final Loss: {history[-1]:.10f}\nHyperParameters: Learning Rate = {learning_rate}, Epochs = {epochs}")
    downsampled_history = history[::5]
    print("\nTraining Loss History every 5 epochs:")
    print(asciichartpy.plot(downsampled_history, {'height': 30, 'format': '{:8.5f}'}))
    for i, loss in enumerate(history):
        print(f"Epoch {i+1}/{epochs}, Loss: {loss:.10f}")

    learning_rates = [1, 0.5, 0.1, 0.01, 0.005, 0.001]

    print(f"\n\n{'Q2':->5}")
    for l in learning_rates:
        weight_l, history_l = training_gradient_descent(data_norm[:, :-1], data_norm[:, -1], np.zeros(data_norm.shape[1] - 1), l, epochs)
        print(f"Training completed. Final weights: {weight_l}, Final Loss: {history_l[-1]:.10f}\nHyperParameters: Learning Rate = {l}, Epochs = {epochs}\n")

    print(f"\n\n{'Q3':->5}")
    data_mocking = np.array([[31, 43, 28, 50],
                             [0, 1, 1, 0],
                             [22, 19, 33, 28]]).T  # Mocking a single data point for prediction

    data_mocking = (data_mocking - data_min[:-1]) / divider[:-1]  # Normalize mocking data

    mock_pred, mock_loss = prediction(data_mocking, np.zeros(data_mocking.shape[0]), weights, data_min, data_max)
    data_mocking = np.concatenate((data_mocking, mock_pred.reshape(-1, 1)), axis=1)
    print(f"Mocking data with predictions by weight {weights}:\n", data_mocking)

def part_two():
    def training_gradient_descent(X, y, weights, learning_rate, bias, epochs):
        history = []

        for _ in range(epochs):
            y_pred = X @ weights + bias
            error = y_pred - y
            loss = MSE(y, y_pred)

            gradient = X.T @ error * (1 / len(y))

            weights -= learning_rate * gradient
            history.append(loss)

        return weights, history

    def prediction(X_test, y_test, weights, bias, data_min, data_max):
        y_pred = (X_test @ weights + bias)
        y_unnorm = y_pred * (data_max[-1] - data_min[-1]) + data_min[-1]

        y_true = y_test * (data_max[-1] - data_min[-1]) + data_min[-1]
        loss = MSE(y_true, y_unnorm)
        return y_unnorm, loss
    
    mapping = {"Iris-setosa": 0, "Iris-versicolor": 1}
    data = pd.read_csv("src/Lab4/data/iris_2class_train.csv")
        
    np.set_printoptions(precision=5, suppress=True)

    data = data.to_numpy()
    data[:, -1] = np.array([mapping[x] for x in data[:, -1]])
    data = data[:, 1:] #cut the ID out off the training

    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    divider = data_max - data_min
    if np.any(divider == 0):
        divider[divider == 0] = 1  # Avoid division by zero
    data_norm = (data - data_min) / divider

    weights = np.zeros(data_norm.shape[1] - 1)  # Initialize weights for features (excluding target)
    learning_rate = 0.1
    epochs = 100
    bias = 0

    features = data_norm[:, :-1].astype(np.float64)
    label = data_norm[:, -1].astype(np.float64)
    weights, loss = training_gradient_descent(features, label, weights, learning_rate, bias, epochs)
    print(weights)

    for i, l in enumerate(loss):
        print(f"Epoch {i+1}/{epochs}, Loss: {l:.10f}")

    ########################################################

    test = pd.read_csv("src/Lab4/data/iris_2class_test.csv")

    test = test.to_numpy()
    test[:, -1] = np.array([mapping[x] for x in test[:, -1]])
    test = test[:, 1:] #cut the ID out off the training
    test_min = test.min(axis=0)
    test_max = test.max(axis=0)
    divider = test_max - test_min
    if np.any(divider == 0):
        divider[divider == 0] = 1  # Avoid division by zero
    test_norm = (test - test_min) / divider

    features = test_norm[:, :-1].astype(np.float64)

    test_pred, _ = prediction(features, np.zeros(features.shape[0]), weights, bias, test_min, test_max)
    print(test_pred)
    test_pred = np.array([1 if i > 0.5 else 0 for i in test_pred])
    print(test_pred)
    print(test_norm[:, -1].astype(np.int16))

def part_three():
    def training_stochastic_gradient_descent(X, y, weights, learning_rate, bias, epochs, window_sizes = 1):
        history = []

        for _ in range(epochs):
            #Random X and y
            indexes = np.random.permutation(len(y))
            x_rand = X[indexes]
            y_rand = y[indexes]
            y_predict = np.zeros(len(y))

            for i in range(0, len(y), window_sizes):
                x_sto = x_rand[i:i + window_sizes]
                y_sto = y_rand[i:i + window_sizes]

                y_pred = x_sto @ weights + bias
                y_predict[i:i + window_sizes] = y_pred
                error = y_pred - y_sto

                gradient = x_sto.T @ error * (1 / len(y_sto))

                weights -= learning_rate * gradient
            loss = MSE(y_rand, y_predict)
            history.append(loss)

        return weights, history

    def prediction(X_test, weights, bias):
        y_pred = (X_test @ weights + bias)
        y_unnorm = y_pred * (data_max[-1] - data_min[-1]) + data_min[-1]
        return y_unnorm

    mapping = {"Iris-setosa": 0, "Iris-versicolor": 1}
    data = pd.read_csv("src/Lab4/data/iris_2class_train.csv")
        
    np.set_printoptions(precision=5, suppress=True)

    data = data.to_numpy()
    data[:, -1] = np.array([mapping[x] for x in data[:, -1]])
    data = data[:, 1:] #cut the ID out off the training

    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    divider = data_max - data_min
    if np.any(divider == 0):
        divider[divider == 0] = 1  # Avoid division by zero
    data_norm = (data - data_min) / divider

    weights = np.zeros(data_norm.shape[1] - 1)  # Initialize weights for features (excluding target)
    learning_rate = 0.001
    epochs = 1000
    bias = 0

    features = data_norm[:, :-1].astype(np.float64)
    label = data_norm[:, -1].astype(np.float64)
    weights, loss = training_stochastic_gradient_descent(features, label, weights, learning_rate, bias, epochs, 1)
    print(weights)

    ########################################################
    
    test = pd.read_csv("src/Lab4/data/iris_2class_test.csv")

    test = test.to_numpy()
    test[:, -1] = np.array([mapping[x] for x in test[:, -1]])
    test = test[:, 1:] #cut the ID out off the training
    test_min = test.min(axis=0)
    test_max = test.max(axis=0)
    divider = test_max - test_min
    if np.any(divider == 0):
        divider[divider == 0] = 1  # Avoid division by zero
    test_norm = (test - test_min) / divider

    features = test_norm[:, :-1].astype(np.float64)

    test_pred = prediction(features, weights, bias)
    print(f"Predicted before sigmoid\n{test_pred}")
    test_pred = np.array([1 if i > 0.5 else 0 for i in test_pred])
    print(f"Predicted after sigmoid\n{test_pred}")
    print(f"Actual target labels\n{test_norm[:, -1].astype(np.int16)}")

if __name__ == "__main__":
    part_one()
    # part_two()
    # part_three()