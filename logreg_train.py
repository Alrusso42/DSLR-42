import pandas as pd
import argparse
from tqdm import tqdm
from utils import gradient_descent, stochastic_gradient_descent, mini_batch_gradient_descent, sigmoid, preprocess_training_data, save_weights
import numpy as np


def train(X, y, classes, learning_rate=0.0001, epochs=10000, lambda_reg=0.001, method='batch', batch_size=128):
    """Entraînement du modèle en choisissant la méthode de descente de gradient."""
    all_weights = {}

    np.random.seed(42)

    classes = sorted(classes)

    base_weights = np.random.randn(X.shape[1] + 1) * 0.01

    for cls in classes:
        binary_y = (y == cls).astype(int)
        weights = base_weights.copy()

        if method == 'batch':
            weights, _ = gradient_descent(X, binary_y, weights)
        elif method == 'sgd':
            weights, _ = stochastic_gradient_descent(X, binary_y, weights)
        elif method == 'mini_batch':
            weights, _ = mini_batch_gradient_descent(X, binary_y, weights)
        else:
            raise ValueError("Méthode non reconnue. Choisir 'batch', 'sgd', ou 'mini_batch'.")

        all_weights[cls] = weights

    return all_weights



def main():
    # Charger les données d'entraînement
    file_path = "dataset_train.csv" 
    X, y, classes = preprocess_training_data(file_path)

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
