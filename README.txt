Hand-Drawn Circuit Symbol Recognition

This project implements a neural network model from scratch to recognize hand-drawn circuit symbols. The model is designed to classify images of digits (0-9) using a custom dataset and advanced machine learning techniques.

Features

Dataset Creation: The model is trained on a dataset of hand-drawn circuit symbols, processed and normalized for optimal performance.
Model Implementation: The neural network uses He Initialization, ReLU activation, and softmax for output predictions. Gradient descent with learning rate decay is implemented for training without using TensorFlow initially.
High Accuracy: Achieved a maximum training accuracy of 98.96% and a testing accuracy of 97.8%, surpassing typical MNIST dataset benchmarks.

Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib

How to Use

1. Dataset Preparation: Ensure the dataset is correctly placed at the specified file path in the code (`/digit-recognizer/train.csv`).
2. Training: Run the script to train the neural network. The model parameters will be initialized and optimized over multiple iterations.
3. Testing: Use the `test_prediction()` function to visualize and test individual predictions.
4. Error Analysis: The `list_incorrect_predictions()` function lists and saves images of incorrect predictions for further analysis.

Code Overview

- Data Loading and Preprocessing: The dataset is loaded from a CSV file, shuffled, and split into training and development sets. Images are normalized to improve model performance.
- Model Initialization: Weights are initialized using He Initialization to ensure effective training.
- Forward Propagation: The network performs forward propagation with ReLU activation and softmax output.
- Backward Propagation: Gradients are calculated and used to update weights and biases.
- Gradient Descent: The model is trained using gradient descent with learning rate decay to ensure convergence.
- Accuracy Evaluation: Accuracy is printed at regular intervals during training, and final accuracy is reported on the development set.

Example Usage

To visualize a prediction, use the `test_prediction()` function with the desired index (Code Input) :

	test_prediction(index=0, X=X_train, Y=Y_train, W1=W1, b1=b1, W2=W2, b2=b2)

To analyze incorrect predictions, use (Code Input) :

	list_incorrect_predictions(X_dev, Y_dev, W1, b1, W2, b2)


Results

- Training Set: Maximum accuracy of 98.96% achieved after 10,000 iterations.
- Development Set: Accuracy of 97.8% on the development set, demonstrating strong generalization.

Future Work

- Further optimization of hyperparameters.
- Expansion of the dataset with additional circuit symbols.
- Integration of more advanced deep learning frameworks like TensorFlow or PyTorch for model refinement.

License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

