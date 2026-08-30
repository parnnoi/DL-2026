import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

weight = [1, -1] # Weight Declaration
bias = -0.03 # Bias Declaration

def generate_data(num, filename, mutation_rate, generate_when_exists:bool=False):
    isExists = os.path.exists(filename)

    #automaticcally regenerate if num was changed when it already declaired
    if isExists:
        df = pd.read_csv(filename)
        length = len(df)
        isExists = length == num

    if not isExists or generate_when_exists:
        x = np.random.rand(num).round(2)
        y = np.random.rand(num).round(2)
        z = np.zeros(num)

        for i in range(num):
            target_x = x[i]
            targey_y = y[i]
            z[i] = 0 if target_x < targey_y else 1 #actual value

            #calculate the mutation rate
            mutaion = (max(target_x, targey_y) - math.fabs(target_x - targey_y)) / max(target_x, targey_y) * mutation_rate #if point near line = higher mutation rate
            if np.random.rand() < mutaion:
                z[i] = np.random.randint(0, 2) #mutate to be random value

        df = pd.DataFrame({
                            "x": x,
                            "y": y,
                            "z": z
                            })
        df.to_csv("Lab1/data.csv", index=False)

def data_plot(filename):
    df = pd.read_csv(filename)
    data1 = df[df["z"] == 0]
    data2 = df[df["z"] == 1]

    plt.scatter(data1["x"], data1["y"], label="Data 1")
    plt.scatter(data2["x"], data2["y"], label="Data 2")

    plt.plot([0, 1], [0, 1], label="x = y", color="black", linestyle="--")

    #if x = 0, y=?
    y_expected_begin = -(0*weight[0]+bias) / weight[1]
    #if x = 1, y=?
    y_expected_end = -(1*weight[0]+bias) / weight[1]
    plt.plot([0, 1], [y_expected_begin, y_expected_end], label="Prediction", color="black", linestyle="--")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data Distribution")
    # plt.legend()
    plt.grid(True)

    plt.show()

def data_plot_prediction(original, prediction):
    df_original = pd.read_csv(original)
    df_prediction = pd.read_csv(prediction)

    # Original data
    data1_original = df_original[df_original["z"] == 0]
    data2_original = df_original[df_original["z"] == 1]

    # Plot original points as circles
    plt.scatter(
        data1_original["x"],
        data1_original["y"],
        label="Original Data x1",
        alpha=0.5,
        marker="o"
    )

    plt.scatter(
        data2_original["x"],
        data2_original["y"],
        label="Original Data x2",
        alpha=0.5,
        marker="o"
    )

    # Find wrong predictions
    wrong_prediction = df_original["z"] != df_prediction["z"]

    wrong_data = df_original[wrong_prediction]

    # Draw squares around wrong predictions
    plt.scatter(
        wrong_data["x"],
        wrong_data["y"],
        marker="s",
        facecolors="none",
        edgecolors="red",
        s=150,
        linewidths=2,
        label="Wrong Prediction"
    )

    # x = y line
    plt.plot(
        [0, 1],
        [0, 1],
        label="actual line x1 = x2",
        color="black",
        linestyle="-"
    )

    # Prediction line
    y_expected_begin = -(0 * weight[0] + bias) / weight[1]
    y_expected_end = -(1 * weight[0] + bias) / weight[1]

    plt.plot(
        [0, 1],
        [y_expected_begin, y_expected_end],
        label="Prediction",
        color="black",
        linestyle="--"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data Distribution with Predictions")
    plt.legend()
    plt.grid(True)

    plt.show()

def classifier(filename):
    df = pd.read_csv(filename)

    data = df[['x', 'y']].to_numpy()

    new_z = weight[0] * data[:, 0] + weight[1] * data[:, 1] + bias
    new_z = np.where(new_z < 0, 0, 1) #Finalize the prediction
    df['z'] = new_z
    df.to_csv("Lab1/classify.csv", index=False)

def numpy_confusion_matrix(y_true, y_pred, num_classes=None):
    # Determine the number of classes
    if num_classes is None:
        num_classes = max(max(y_true), max(y_pred)) + 1

    tp = true_positive = np.sum((y_true == 1) & (y_pred == 1))
    tn = true_negative = np.sum((y_true == 0) & (y_pred == 0))
    fp = false_positive = np.sum((y_true == 0) & (y_pred == 1))
    fn = false_negative = np.sum((y_true == 1) & (y_pred == 0))

    cm = np.array([ [tn, fp],
                    [fn, tp]])

    # Create an easy-to-read table
    table = pd.DataFrame(
        cm,
        index=[f"Actual {"Positive" if i==0 else "Negative"}" for i in range(num_classes)],
        columns=[f"Predicted {"Positive" if i==0 else "Negative"}" for i in range(num_classes)]
    )

    print("\nConfusion Matrix:")
    print(table)

    accuracy = (tp + tn) / np.sum(cm)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    print(f"\nTotal Predictions: {np.sum(cm)}")
    print(f"Correct Predictions: {np.trace(cm)}")
    print(f"Incorrect Predictions: {np.sum(cm) - np.trace(cm)}")

    return cm

generate_data(1000, 'Lab1/data.csv', 0.15, generate_when_exists=False)
# data_plot('Lab1/data.csv')
classifier('Lab1/data.csv')
# data_plot('Lab1/classify.csv')
cm = numpy_confusion_matrix(pd.read_csv('Lab1/data.csv')['z'].to_numpy().astype(int), pd.read_csv('Lab1/classify.csv')['z'].to_numpy().astype(int), num_classes=2)
data_plot_prediction('Lab1/data.csv', 'Lab1/classify.csv')
