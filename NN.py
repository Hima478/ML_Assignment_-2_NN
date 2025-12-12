import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class NeuralNetwork:
    def __init__(self, layer_sizes, activation_name="relu", learning_rate=0.01, seed=42):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.learning_rate = learning_rate
        self.activation_name = activation_name.lower()

        rng = np.random.default_rng(seed)
        self.params = {}

        for l in range(1, len(layer_sizes)):
            in_dim = layer_sizes[l - 1]
            out_dim = layer_sizes[l]
            if self.activation_name == "relu" and l != self.num_layers:
                std = np.sqrt(2.0 / in_dim)
            else:
                std = np.sqrt(1.0 / in_dim)

            self.params[f"W{l}"] = rng.normal(0.0, std, size=(in_dim, out_dim))
            self.params[f"b{l}"] = np.zeros((1, out_dim))


    def _activation(self, Z):
        if self.activation_name == "relu":
            return np.maximum(0, Z)
        elif self.activation_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-Z))
        else:
            raise ValueError("Unsupported activation (use 'relu' or 'sigmoid')")

    def _activation_derivative(self, A):
        if self.activation_name == "relu":
            return (A > 0).astype(A.dtype)
        elif self.activation_name == "sigmoid":
            return A * (1.0 - A)
        else:
            raise ValueError("Unsupported activation (use 'relu' or 'sigmoid')")

    def _softmax(self, Z):
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z_shift)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X):
        A = X
        cache = [("input", A)]

        for l in range(1, self.num_layers):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            Z = A @ W + b
            A = self._activation(Z)
            cache.append((Z, A))

        W_L = self.params[f"W{self.num_layers}"]
        b_L = self.params[f"b{self.num_layers}"]
        Z_L = A @ W_L + b_L
        probs = self._softmax(Z_L)
        cache.append((Z_L, probs))

        return probs, cache

    def compute_loss(self, probs, y_one_hot):
        eps = 1e-15
        N = y_one_hot.shape[0]
        log_likelihood = -np.log(probs + eps) * y_one_hot
        loss = np.sum(log_likelihood) / N
        return loss


    def backward(self, X, y_one_hot, cache):
        grads = {}
        deltas = {}
        for j in range(self.num_layers, 0, -1):
            if j == self.num_layers:
                delta = cache[j][1] - y_one_hot
                deltas[j] = delta
            else:
                Wk = self.params[f"W{j+1}"]
                deltak = deltas[j+1]
                outputj = cache[j][1]
                delta = (deltak @ Wk.T) * self._activation_derivative(outputj)
                deltas[j] = delta
            A_prev = cache[j-1][1]
            grads[f"dW{j}"] = (A_prev.T @ delta) / X.shape[0]
            grads[f"db{j}"] = np.sum(delta, axis=0, keepdims=True) / X.shape[0]
        return grads

    def train(self, X_train, y_train, X_val, y_val,
              num_epochs=50, batch_size=32, verbose=True):
        N = X_train.shape[0]
        num_batches = int(np.ceil(N / batch_size))
        rng = np.random.default_rng()

        for epoch in range(1, num_epochs + 1):
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

                probs, cache = self.forward(X_batch)
                loss = self.compute_loss(probs, y_batch)
                epoch_loss += loss

                grads = self.backward(X_batch, y_batch, cache)

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
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y_true_int):
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true_int)


def main():
    digits = load_digits()
    X = digits.images
    y = digits.target
    class_names = digits.target_names

    X = X.reshape(X.shape[0], -1).astype(np.float32)

    X_temp, X_test, y_temp, y_test_int = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train_int, y_val_int = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    enc = OneHotEncoder(sparse_output=False)
    y_train = enc.fit_transform(y_train_int.reshape(-1, 1))
    y_val = enc.transform(y_val_int.reshape(-1, 1))
    y_test = enc.transform(y_test_int.reshape(-1, 1))

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

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

    nn = NeuralNetwork(layer_sizes, activation_name=activation, learning_rate=0.01)

    nn.train(X_train, y_train, X_val, y_val,
            num_epochs=50, batch_size=32, verbose=True)

    test_acc = nn.accuracy(X_test, y_test_int)
    print("Test accuracy:", test_acc)

    def predict_sample(idx):
        x_sample = X_test[idx].reshape(1, -1)
        pred_class = nn.predict(x_sample)[0]
        print(f"Predicted class: {pred_class} (label name: {class_names[pred_class]})")
        print(f"True class: {y_test_int[idx]}")

    predict_sample(0)


if __name__ == "__main__":
    main()