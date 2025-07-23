import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from utils import preprocess_test_data, sigmoid

def softmax(x):
    """Calcul de la fonction softmax pour obtenir des probabilités entre 0 et 1."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def predict(X, weights, classes):
    """Prédiction pour chaque échantillon avec les poids appris."""
    predictions = []

    for x in X:
        scores = []

        for cls, w in weights.items():
            score = np.dot(x, w[1:]) + w[0]
            scores.append(score)
        
        # Appliquer Softmax pour obtenir des probabilités
        probs = softmax(np.array(scores))

        # Choisir la classe avec la probabilité la plus élevée
        predicted_class = classes[np.argmax(probs)]
        predictions.append(predicted_class)

    return predictions


def load_weights(file_path):
    """Charge les poids depuis un fichier."""
    weights = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                cls, w_str = line.strip().split(":")
                w = np.array([float(x) for x in w_str.split(",")])
                weights[cls] = w
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{file_path}' est manquant.")
        exit(1)
    return weights


def save_predictions(predictions, file_path):
    """Sauvegarde les prédictions dans un fichier CSV."""
    df = pd.DataFrame(predictions, columns=['Index', 'Hogwarts House'])
    df.to_csv(file_path, index=False)


def check_file(file):
    """Vérifie si le fichier existe et affiche un message d'erreur s'il est manquant."""
    if not os.path.isfile(file):
        print(f"Erreur : Le fichier '{file}' est manquant.")
        exit(1)


def main():
    weight_filename = 'weight.txt'
    test_filename = 'dataset_test.csv'

    # Vérification des fichiers
    check_file(test_filename)

    # Charger les données de test
    X_test = preprocess_test_data(test_filename)

    # Charger les poids du fichier
    weights = load_weights(weight_filename)

    # Charger les classes (maison) de poids
    classes = list(weights.keys())

    # Effectuer la prédiction
    predictions = predict(X_test, weights, classes)

    # Sauvegarder les résultats dans un fichier CSV
    result = pd.read_csv(test_filename)[['Index']]
    result['Hogwarts House'] = predictions
    result.to_csv('houses.csv', index=False)
    print("Prédictions sauvegardées dans houses.csv.")


if __name__ == "__main__":
    main()
