import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# -----------------------------
# Neural Network implementation
# -----------------------------

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_name="relu", learning_rate=0.01, seed=42):
        """
        layer_sizes: list like [input_dim, h1, h2, ..., output_dim]
        activation_name: "relu" or "sigmoid" for all hidden layers
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1   # number of weight layers
        self.learning_rate = learning_rate
        self.activation_name = activation_name.lower()

        rng = np.random.default_rng(seed)
        self.params = {}

        # He or Xavier initialization depending on activation
        for l in range(1, len(layer_sizes)):
            in_dim = layer_sizes[l - 1]
            out_dim = layer_sizes[l]
            if self.activation_name == "relu" and l != self.num_layers:  # hidden layers
                # He initialization
                std = np.sqrt(2.0 / in_dim)
            else:
                # Xavier/glorot initialization
                std = np.sqrt(1.0 / in_dim)

            self.params[f"W{l}"] = rng.normal(0.0, std, size=(in_dim, out_dim))
            self.params[f"b{l}"] = np.zeros((1, out_dim))

    # --------- activations ----------

    def _activation(self, Z):
        if self.activation_name == "relu":
            return np.maximum(0, Z)
        elif self.activation_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-Z))
        else:
            raise ValueError("Unsupported activation (use 'relu' or 'sigmoid')")

    def _activation_derivative(self, A):
        """
        Derivative wrt Z, expressed as function of A=f(Z) for convenience.
        NOTE: This will be used in backprop, which your team will fill in.
        """
        if self.activation_name == "relu":
            return (A > 0).astype(A.dtype)
        elif self.activation_name == "sigmoid":
            return A * (1.0 - A)
        else:
            raise ValueError("Unsupported activation (use 'relu' or 'sigmoid')")

    def _softmax(self, Z):
        # Z: (N, C)
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)  # numerical stability
        expZ = np.exp(Z_shift)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    # --------- forward & loss ----------

    def forward(self, X):
        """
        Forward pass through all layers.
        Returns:
            probs: softmax probabilities for last layer
            cache: list of (Z_l, A_l) for each layer (for backprop)
        """
        A = X
        cache = [("input", A)]  # store input as layer 0 activation

        # Hidden layers
        for l in range(1, self.num_layers):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            Z = A @ W + b     # (N, in_dim) @ (in_dim, out_dim) + (1, out_dim)
            A = self._activation(Z)
            cache.append((Z, A))

        # Output layer (softmax)
        W_L = self.params[f"W{self.num_layers}"]
        b_L = self.params[f"b{self.num_layers}"]
        Z_L = A @ W_L + b_L
        probs = self._softmax(Z_L)
        cache.append((Z_L, probs))

        return probs, cache

    def compute_loss(self, probs, y_one_hot):
        """
        Categorical cross-entropy loss.
        probs: (N, C)
        y_one_hot: (N, C)
        """
        eps = 1e-15
        N = y_one_hot.shape[0]
        log_likelihood = -np.log(probs + eps) * y_one_hot
        loss = np.sum(log_likelihood) / N
        return loss

    # --------- backprop (TO BE DONE BY YOUR TEAM) ----------

    def backward(self, X, y_one_hot, cache):
        """
        Compute gradients of loss w.r.t. all weights and biases.

        X: (N, D) input
        y_one_hot: (N, C) targets
        cache: list of (Z_l, A_l) from forward()

        Should return a dict grads such that:
            grads["dW1"], grads["db1"], ..., grads[f"dW{L}"], grads[f"db{L}"]

        *** TODO: IMPLEMENT THIS FUNCTION ***
        """
        raise NotImplementedError("Backpropagation is left for the team to implement.")

    # --------- training & prediction ----------

    def train(self, X_train, y_train, X_val, y_val,
              num_epochs=50, batch_size=32, verbose=True):
        """
        Mini-batch gradient descent training loop.
        (Will only run once backward() is implemented.)
        """
        N = X_train.shape[0]
        num_batches = int(np.ceil(N / batch_size))
        rng = np.random.default_rng()

        for epoch in range(1, num_epochs + 1):
            # shuffle indices
            indices = np.arange(N)
            rng.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0.0

            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, N)
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                # forward
                probs, cache = self.forward(X_batch)
                loss = self.compute_loss(probs, y_batch)
                epoch_loss += loss

                # backward (YOU need to implement backward())
                grads = self.backward(X_batch, y_batch, cache)

                # parameter update: W_l -= lr * dW_l, b_l -= lr * db_l
                for l in range(1, self.num_layers + 1):
                    self.params[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
                    self.params[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]

            epoch_loss /= num_batches

            if verbose:
                train_acc = self.accuracy(X_train, np.argmax(y_train, axis=1))
                val_acc = self.accuracy(X_val, np.argmax(y_val, axis=1))
                print(f"Epoch {epoch:3d} | loss={epoch_loss:.4f} "
                      f"| train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

    def predict(self, X):
        """
        Predict class indices for each sample in X.
        """
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y_true_int):
        """
        Convenience function for accuracy.
        y_true_int: integer labels (N,)
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true_int)


# -----------------------------
# Main script
# -----------------------------

def main():
    # 1) Load an image dataset (digits: 8x8 grayscale images, 10 classes)
    digits = load_digits()
    X = digits.images      # (N, 8, 8)
    y = digits.target      # (N,)
    class_names = digits.target_names

    # 2) Preprocessing / cleaning using sklearn
    # Flatten images: (N, 8, 8) -> (N, 64)
    X = X.reshape(X.shape[0], -1).astype(np.float32)

    # Split into train / val / test using sklearn
    # 20% test, then from remaining 80% take 25% as val => 60/20/20
    X_temp, X_test, y_temp, y_test_int = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train_int, y_val_int = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    # Scale features to [0,1] using MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # One-hot encode labels with sklearn
    enc = OneHotEncoder(sparse=False)
    y_train = enc.fit_transform(y_train_int.reshape(-1, 1))
    y_val = enc.transform(y_val_int.reshape(-1, 1))
    y_test = enc.transform(y_test_int.reshape(-1, 1))

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    # 4) Ask user for network architecture
    print("Input dimension:", input_dim)
    print("Number of classes:", output_dim)

    num_hidden = int(input("Enter number of hidden layers: "))
    hidden_sizes = []
    for i in range(num_hidden):
        h = int(input(f"Enter number of neurons in hidden layer {i + 1}: "))
        hidden_sizes.append(h)

    activation = input("Enter activation function for hidden layers ('relu' or 'sigmoid'): ").strip().lower()
    if activation not in ("relu", "sigmoid"):
        raise ValueError("Activation must be 'relu' or 'sigmoid'")

    layer_sizes = [input_dim] + hidden_sizes + [output_dim]
    print("Network architecture:", layer_sizes)

    # 5) Initialize neural network
    nn = NeuralNetwork(layer_sizes, activation_name=activation, learning_rate=0.01)

    # 6) Train network (this will work after backprop is implemented)
    # nn.train(X_train, y_train, X_val, y_val,
    #          num_epochs=50, batch_size=32, verbose=True)

    # 7) Evaluate on test set (once trained)
    # test_acc = nn.accuracy(X_test, y_test_int)
    # print("Test accuracy:", test_acc)

    # 8) predict() function for a single test sample
    def predict_sample(idx):
        """
        Takes an index into X_test and prints the predicted class.
        """
        x_sample = X_test[idx].reshape(1, -1)
        pred_class = nn.predict(x_sample)[0]
        print(f"Predicted class: {pred_class} (label name: {class_names[pred_class]})")
        print(f"True class: {y_test_int[idx]}")

    # Example usage (will be meaningful after training):
    # predict_sample(0)


if __name__ == "__main__":
    main()