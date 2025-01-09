import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_dim, learning_rate=0.01, epochs=1000):
        """
        Initialise le perceptron.

        Args:
            input_dim (int): Nombre de caractéristiques des données d'entrée.
            learning_rate (float): Taux d'apprentissage (par défaut : 0.01).
            epochs (int): Nombre d'itérations d'entraînement (par défaut : 1000).
        """
        self.weights = np.random.randn(input_dim)  # Poids initialisés aléatoirement
        self.bias = 0.0  # Biais initialisé à zéro
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        """
        Fonction d'activation (Heaviside).

        Args:
            x (float): Entrée.

        Returns:
            int: 1 si x >= 0, sinon 0.
        """
        return 1 if x >= 0 else 0

    def predict(self, X):
        """
        Prédit les classes pour les données.

        Args:
            X (np.array): Données d'entrée (n_samples, n_features).

        Returns:
            np.array: Prédictions (0 ou 1).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(x) for x in linear_output])

    def fit(self, X, y):
        """
        Entraîne le perceptron.

        Args:
            X (np.array): Données d'entrée (n_samples, n_features).
            y (np.array): Étiquettes de classe (0 ou 1).
        """
        for epoch in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.activation(np.dot(xi, self.weights) + self.bias)
                error = target - prediction
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error
            if np.all(self.predict(X) == y):  # Arrêt si toutes les prédictions sont correctes
                print(f"Entraînement terminé après {epoch + 1} itérations")
                return
        print(f"Entraînement terminé après {self.epochs} itérations")

def generate_data(num_samples=100):
    """
    Génère des données pour tester le perceptron.

    Args:
        num_samples (int): Nombre d'échantillons par classe (par défaut : 100).

    Returns:
        X (np.array): Données d'entrée (2 * num_samples, 2).
        y (np.array): Étiquettes de classe (0 ou 1).
    """
    np.random.seed(42)
    X1 = np.random.randn(num_samples, 2) + np.array([2, 2])  # Classe 1
    X2 = np.random.randn(num_samples, 2) + np.array([-2, -2])  # Classe 0
    X = np.vstack((X1, X2))
    y = np.array([1] * num_samples + [0] * num_samples)
    return X, y

# Tester le perceptron
X, y = generate_data()
perceptron = Perceptron(input_dim=2, learning_rate=0.1, epochs=1000)
perceptron.fit(X, y)

# Prédictions
predictions = perceptron.predict(X)

# Visualisation
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", label="Véritables étiquettes", alpha=0.6)
plt.title("Données factices et séparation avec Perceptron")
plt.xlabel("Caractéristique 1")
plt.ylabel("Caractéristique 2")

# Tracer la frontière de décision
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
Z = perceptron.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap="bwr")

plt.legend()
plt.show()

# Évaluation du modèle
accuracy = np.mean(predictions == y)
print(f"Précision : {accuracy * 100:.2f}%")
