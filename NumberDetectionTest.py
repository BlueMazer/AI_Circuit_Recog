import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

# File path to the dataset
file_path = '/Users/rayanraad/PycharmProjects/NumberDetectionTest /digit-recognizer/train.csv'

# Load data
if os.path.exists(file_path):
    data = pd.read_csv(file_path)
else:
    print("File not found. Please check the file path.")
    raise SystemExit

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# Split data into development and training sets
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.0

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0
_, m_train = X_train.shape


def he_initialization(num_units, num_features):
    """
    Initialize weights using He initialization.

    Parameters:
    - num_units: Number of units in the current layer
    - num_features: Number of units in the previous layer

    Returns:
    - W: Initialized weight matrix
    - b: Initialized bias vector (optional, initialized to zero)
    """
    stddev = np.sqrt(2. / num_features)
    W = np.random.randn(num_units, num_features) * stddev
    b = np.zeros((num_units, 1))
    return W, b


def init_params(num_classes):
    W1, b1 = he_initialization(128, 784)  # Example: 128 units in first hidden layer
    W2, b2 = he_initialization(num_classes, 128)  # Example: 128 units in hidden layer, num_classes output units
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0, Z)


def softmax(Z):
    exp_Z = np.exp(Z)
    return exp_Z / (np.sum(exp_Z, axis=0, keepdims=True) + 1e-8)


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y, num_classes):
    one_hot_Y = np.zeros((Y.size, num_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def deriv_ReLU(Z):
    return Z > 0


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    num_classes = W2.shape[0]
    one_hot_Y = one_hot(Y, num_classes)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, dW2, db1, db2


def update_params(W1, b1, W2, b2, dW1, dW2, db1, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, axis=0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha, initial_alpha, num_classes):
    W1, b1, W2, b2 = init_params(num_classes)
    alpha = initial_alpha

    for i in range(iterations):
        # Making a decreasing learning rate
        if i % 1000 == 0 and i != 0:
            alpha *= 0.95  # Decrease learning rate by 5%

        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, dW2, db1, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, dW2, db1, db2, alpha)
        if (i % 100) == 0:
            predictions = get_predictions(A2)
            print(f"Iteration: {i}")
            print(f"Learning Rate: {alpha}")
            print(f"Accuracy: {get_accuracy(predictions, Y)}")
    return W1, b1, W2, b2


# Train the neural network
num_classes = 10  # Number of classes (digits 0-9)
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 10000, 0.1, 0.1, num_classes)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)


def test_prediction(index, X, Y, W1, b1, W2, b2):
    current_image = X[:, index, None]
    prediction = make_predictions(X[:, index, None], W1, b1, W2, b2)
    label = Y[index]
    print(f"Prediction: {prediction}")
    print(f"Label: {label}")

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


def list_incorrect_predictions(X, Y, W1, b1, W2, b2):
    predictions = make_predictions(X, W1, b1, W2, b2)
    incorrect_indices = np.where(predictions != Y)[0]

    if len(incorrect_indices) == 0:
        print("All predictions are correct!")
        return

    num_errors_to_show = int(input("How many errors would you like to see? "))
    num_errors_to_show = min(num_errors_to_show, len(incorrect_indices))  # Ensure we do not exceed the number of errors

    print(f"Number of incorrect predictions: {len(incorrect_indices)}")

    # Ensure the error directory exists
    output_dir = 'incorrect_predictions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index in incorrect_indices[:num_errors_to_show]:
        current_image = X[:, index, None]
        prediction = predictions[index]
        label = Y[index]

        print(f"Index: {index}")
        print(f"Prediction: {prediction}")
        print(f"Label: {label}")

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')

        # Save the image with the predicted number as part of the filename
        filename = os.path.join(output_dir, f'predicted_{prediction}_actual_{label}_{index}.png')
        plt.savefig(filename)
        plt.close()  # Close the plot to free memory


# Check incorrect predictions
print("Checking incorrect predictions on training set:")
list_incorrect_predictions(X_train, Y_train, W1, b1, W2, b2)

print("Checking incorrect predictions on development set:")
list_incorrect_predictions(X_dev, Y_dev, W1, b1, W2, b2)

# Checking accuracy on the testing set
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print(f"Accuracy on development set: {get_accuracy(dev_predictions, Y_dev)}")

# HIGHEST ACCURACY: 0.9896341463414634 with Iteration = 10000 and Initial Alpha = 0.1 and .95 decrease for learning rate per 1000
# Highest Accuracy on development set: 0.978