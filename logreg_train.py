import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm  # Importer tqdm pour la barre de progression

def sigmoid(z):
    """La fonction sigmoïde pour la classification binaire."""
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, weights, learning_rate=0.01, epochs=10000, lambda_reg=0.0001, tolerance=1e-6, progress_bar=None):
    """Descente de gradient batch pour optimiser les poids du modèle."""
    m = len(y)
    cost_history = []

    for epoch in range(epochs):
        # Mélanger les données pour chaque itération
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Calcul des prédictions et des gradients
        z = np.dot(X_shuffled, weights[1:]) + weights[0]
        predictions = sigmoid(z)

        # Calcul du gradient
        gradient = np.zeros(len(weights))
        gradient[0] = (1 / m) * np.sum(predictions - y_shuffled).item()  # .item() pour extraire la valeur
        gradient[1:] = (1 / m) * X_shuffled.T @ (predictions - y_shuffled)

        # Ajout de la régularisation L2
        gradient[1:] += (lambda_reg / m) * weights[1:]

        # Mise à jour des poids
        weights -= learning_rate * gradient

        # Calcul du coût et suivi de l'historique
        z = np.dot(X, weights[1:]) + weights[0]
        predictions = sigmoid(z)
        cost = (-1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions)) + (lambda_reg / (2 * m)) * np.sum(weights[1:] ** 2)
        cost_history.append(cost)

        # Early stopping si le coût ne change plus de manière significative
        if epoch > 0 and np.abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"Early stopping à l'époque {epoch}, variation du coût inférieure à {tolerance}.")
            break

        # Mettre à jour la barre de progression à chaque époque
        if progress_bar:
            progress_bar.update(1)

    return weights, cost_history

def stochastic_gradient_descent(X, y, weights, learning_rate=0.001, epochs=1000, lambda_reg=0.001, batch_size=128, progress_bar=None):
    """Fonction de gradient descendant stochastique pour entraîner le modèle."""
    m = len(y)
    cost_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Traitement de chaque exemple individuellement
        for i in range(m):
            xi = X_shuffled[i:i+1]
            yi = y_shuffled[i:i+1]

            # Calcul de la prédiction
            z = np.dot(xi, weights[1:]) + weights[0]
            prediction = sigmoid(z)

            # Calcul du gradient
            gradient = np.zeros(len(weights))
            gradient[0] = prediction.item() - yi.item()  # .item() pour extraire la valeur
            gradient[1:] = xi.flatten().T * (prediction - yi).item()  # .item() pour extraire la valeur

            gradient[1:] += (lambda_reg / m) * weights[1:]

            # Mise à jour des poids
            weights -= learning_rate * gradient
        
        # Mise à jour de la barre de progression après chaque époque
        if progress_bar.n < progress_bar.total:
            progress_bar.update(1)

    progress_bar.n = progress_bar.total
    progress_bar.last_print_n = progress_bar.n
    progress_bar.update(0)

    return weights, cost_history

def mini_batch_gradient_descent(X, y, weights, learning_rate=0.001, epochs=20000, lambda_reg=0.001, batch_size=128, progress_bar=None):
    """Fonction de descente de gradient par mini-lots pour entraîner le modèle."""
    m = len(y)
    cost_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]

            # Calcul de la prédiction
            z = np.dot(xi, weights[1:]) + weights[0]
            predictions = sigmoid(z)

            # Calcul du gradient
            gradient_dw = np.zeros(len(weights))
            gradient_dw[0] = np.mean(predictions - yi).item()  # .item() pour extraire la valeur
            gradient_dw[1:] = (1 / batch_size) * xi.T @ (predictions - yi)

            gradient_dw[1:] += (lambda_reg / m) * weights[1:]

            # Mise à jour des poids
            weights -= learning_rate * gradient_dw

        # Calcul du coût (pour suivi de la descente du gradient)
        z = np.dot(X, weights[1:]) + weights[0]
        predictions = sigmoid(z)
        cost = (-1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions)) + (lambda_reg / (2 * m)) * np.sum(weights[1:] ** 2)
        cost_history.append(cost)

        # Mettre à jour la barre de progression à chaque époque
        if progress_bar:
            progress_bar.update(1)

    return weights, cost_history


def train(X, y, classes, learning_rate=0.0001, epochs=500, lambda_reg=0.001, method='batch', batch_size=128):
    """Entraînement du modèle en choisissant la méthode de descente de gradient."""
    all_weights = {}

    if method == 'batch':
        total_iterations = epochs
    elif method == 'sgd':
        total_iterations = epochs
    elif method == 'mini_batch':
        total_iterations = epochs * (len(y) // batch_size)
    else:
        raise ValueError("Méthode non reconnue. Choisir 'batch', 'sgd', ou 'mini_batch'.")

    # Créer la barre de progression (avec couleurs et emoji)
    progress_bar = tqdm(total=total_iterations, desc="Entraînement 🚀 ", ncols=100, unit="itération", leave=True, bar_format="{l_bar}{bar}", colour="green")

    for cls in classes:
        binary_y = (y == cls).astype(int)
        weights = np.random.randn(X.shape[1] + 1) * 0.01

        # Choisir la méthode de descente de gradient
        if method == 'batch':
            weights, _ = gradient_descent(X, binary_y, weights, learning_rate, epochs, lambda_reg, progress_bar=progress_bar)
        elif method == 'sgd':
            weights, _ = stochastic_gradient_descent(X, binary_y, weights, learning_rate, epochs, lambda_reg, progress_bar=progress_bar)
        elif method == 'mini_batch':
            weights, _ = mini_batch_gradient_descent(X, binary_y, weights, learning_rate, epochs, lambda_reg, batch_size, progress_bar=progress_bar)
        else:
            raise ValueError("Méthode non reconnue. Choisir 'batch', 'sgd', ou 'mini_batch'.")

        all_weights[cls] = weights

    progress_bar.update(total_iterations - progress_bar.n)
    progress_bar.close()

    return all_weights

def save_weights(weights, filename):
    """Sauvegarde des poids dans un fichier."""
    with open(filename, 'w') as f:
        for cls, w in weights.items():
            f.write(f"{cls}:{','.join(map(str, w))}\n")

def preprocess_data(file_path):
    """Charge et nettoie les données (normalisation et sélection des colonnes pertinentes)."""
    data = pd.read_csv(file_path)

    # Sélection des colonnes pertinentes (seules les colonnes numériques sont prises en compte)
    features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
                'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 
                'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
    
    X = data[features].values
    y = data['Hogwarts House'].values
    
    # Remplacer les NaN par la moyenne de chaque colonne
    X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

    # Normalisation Min-Max des données
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min)
    
    return X, y, list(data['Hogwarts House'].unique())


def main():
    # Charger les données d'entraînement
    file_path = "dataset_train.csv" 
    X, y, classes = preprocess_data(file_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['batch', 'sgd', 'mini_batch'], default='batch', help="Méthode de descente de gradient à utiliser")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Taux d'apprentissage")
    parser.add_argument('--lambda_reg', type=float, default=0.0001, help="Facteur de régularisation L2")
    args = parser.parse_args()

    # Entraîner le modèle avec la méthode choisie
    weights = train(X, y, classes, learning_rate=args.learning_rate, lambda_reg=args.lambda_reg, method=args.method)

    # Sauvegarder les poids dans un fichier
    save_weights(weights, 'weight.txt')
    print(f"Poids sauvegardés dans weight.txt avec la méthode {args.method}.")

if __name__ == "__main__":
    main()
