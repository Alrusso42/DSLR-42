import pandas as pd
import argparse
from tqdm import tqdm
from utils import gradient_descent, stochastic_gradient_descent, mini_batch_gradient_descent, sigmoid, preprocess_training_data, save_weights
import numpy as np
import matplotlib.pyplot as plt


def train(X, y, classes, learning_rate=0.0001, epochs=10000, lambda_reg=0.001, method='batch', batch_size=128):
    all_weights = {}
    all_cost_histories = {}

    np.random.seed(42)
    classes = sorted(classes)
    base_weights = np.random.randn(X.shape[1] + 1) * 0.01

    for cls in classes:
        binary_y = (y == cls).astype(int)
        weights = base_weights.copy()

        if method == 'batch':
            weights, cost_history = gradient_descent(X, binary_y, weights)
        elif method == 'sgd':
            weights, cost_history = stochastic_gradient_descent(X, binary_y, weights)
        elif method == 'mini_batch':
            weights, cost_history = mini_batch_gradient_descent(X, binary_y, weights)
        else:
            raise ValueError("Méthode non reconnue. Choisir 'batch', 'sgd', ou 'mini_batch'.")

        all_weights[cls] = weights
        all_cost_histories[cls] = cost_history

    # Moyenne des losses sur toutes les classes (all_cost_histories est un dict)
    avg_cost = np.mean([cost for cost in all_cost_histories.values()], axis=0)

    plt.figure(figsize=(8,5))
    plt.plot(avg_cost, label='Loss moyen')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Courbe de Loss durant l'entraînement ({method})")
    plt.legend()
    plt.grid(True)
    plt.show()

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
