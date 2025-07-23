import pandas as pd
import numpy as np
import os


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║             Fonctions relatives à l'analyse des données               ║
# ║                  contenues dans un fichier CSV.                       ║ 
# ╚═══════════════════════════════════════════════════════════════════════╝

class Color:
    YELLOW = '\033[33m'
    WHITE = '\033[37m'
    RED = '\033[31m'
    # Réinitialisation de la couleur
    ENDC = '\033[0m'
    BLUE = '\033[34m'
    GREEN = '\033[32m'

def calculate_statistics(dataCsv):
    """
    Résume les statistiques descriptives pour une liste ou série de valeurs numériques.

    La fonction calcule pour les valeurs valides (non nulles) :
        - Count : nombre de valeurs non nulles
        - Mean : moyenne
        - Std : écart-type
        - Min : valeur minimale
        - 25% : premier quartile
        - 50% : médiane
        - 75% : troisième quartile
        - Max : valeur maximale
        - IQR : intervalle interquartile (75% - 25%)

    Args:
        dataCsv (iterable): Une liste ou série contenant des valeurs numériques (peut contenir des NaN).

    Returns:
        tuple: Un tuple contenant (count, mean, std_dev, min_value, q1, median, q3, max_value, iqr).
               Si aucune valeur valide n'est trouvée, retourne un tuple de zéros.
    """

    count = 0                       
    total = 0                       
    total_squared = 0
    # Initialisation minimum à l'infini positif              
    min_value = float('inf')
    # Initialisation maximum à l'infini négatif
    max_value = float('-inf')
    values = [] 

    for value in dataCsv:
        # On ignore les valeurs manquantes
        if pd.notnull(value):
            count += 1
            total += value
            total_squared += value ** 2
            values.append(value)
            # Mise à jour la valeur minimale
            min_value = min(min_value, value)
            # Mise à jour la valeur maximale
            max_value = max(max_value, value)

    # Si aucune valeur valide, retourne un tuple de zéros
    if count == 0:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0)

    # Calcul de la moyenne
    mean = total / count
    # Calcul de la variance
    variance = (total_squared / count) - (mean ** 2)
    # Calcul de l'écart-type
    std_dev = variance ** 0.5

    # Trie des valeurs pour le calcul des quartiles
    values.sort()
    # Calcul du premier quartile (25%)
    q1 = values[int(0.25 * count)]
    # Calcul de la médiane (50%)
    median = values[int(0.5 * count)]
    # Calcul du troisième quartile (75%)
    q3 = values[int(0.75 * count)]
    # Calcul de l'écart interquartile (IQR)
    iqr = q3 - q1

    # Retourne un tuple avec toutes les statistiques
    return (count, mean, std_dev, min_value, q1, median, q3, max_value, iqr)


def summarize_dataCsv(dataCsv):
    """
    Résume les statistiques descriptives pour chaque colonne numérique d'un fichier CSV.

    Pour chaque colonne numérique, la fonction calcule :
        - Count : nombre de valeurs non nulles
        - Mean : moyenne
        - Std : écart-type
        - Min : valeur minimale
        - 25% : premier quartile
        - 50% : médiane
        - 75% : troisième quartile
        - Max : valeur maximale
        - IQR : intervalle interquartile (75% - 25%)

    Les colonnes non numériques sont marquées comme "Valeur non calculable".

    Args:
        dataCsv (pd.dataCsv): Le fichier CSV à analyser.

    Returns:
        dict: Un dictionnaire contenant un résumé statistique pour chaque colonne numérique.
    """
    summary = {}

    # Parcourt chaque colonne du dataCsv
    for column in dataCsv.columns:
        # Vérifie si la colonne est de type numérique
        if pd.api.types.is_numeric_dtype(dataCsv[column]):
            # Calcule les statistiques personnalisées
            stats = calculate_statistics(dataCsv[column])

            # Remplissage du dictionnaire
            summary[column] = {
                'Count': stats[0],
                'Mean' : stats[1],
                'Std'  : stats[2],
                'Min'  : stats[3],
                '25%'  : stats[4],
                '50%'  : stats[5],
                '75%'  : stats[6],
                'Max'  : stats[7],
                'IQR'  : stats[8],
            }
        else:
            # Si la colonne n'est pas numérique on l'affiche
            summary[column] = "Valeur non calculable"

    return summary

def print_summary(summary):
    """
    Affiche de manière formatée un résumé statistique des colonnes d’un fichier CSV.

    Cette fonction prend un dictionnaire `summary` produit par `summarize_dataCsv`
    contenant les statistiques descriptives par colonne, et les affiche en tableau.

    Les couleurs sont gérées à partir de la classe `Color` pour rendre la sortie plus agréable à lire.

    Args:
        summary (dict): Dictionnaire contenant les statistiques par colonne.
    """
    # Liste des en-têtes du tableau de description
    headers = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'IQR']

    # Affiche l'entête avec les noms des statistiques
    print(f"{Color.BLUE}{'':<30}", end="")
    for header in headers:
        print(f"{header:^15}", end="")
    print()  

    # Parcourt chaque colonne du résumé
    for column, values in summary.items():
        # On ignore volontairement la colonne Index car inutile à analyser
        if column.lower() == 'index':
            continue

        # Affiche le nom de la colonne
        print(f"{Color.BLUE}{column:<30}", end="")

        # Si les valeurs sont un dictionnaire de statistiques numériques
        if isinstance(values, dict):
            # Affichage des valeurs sur uniquement 2 décimales pour la lisibilité
            for value in values.values():
                print(f"{Color.GREEN}{value:^15.2f}{Color.ENDC}", end="")
        else:
            # Si non numérique, affiche une info textuelle centrée
            print(f"{' ' * 50}{Color.YELLOW}{'Valeur non calculable':^15}{Color.ENDC}", end="")

        print(Color.ENDC)

def variance(values):
    """
    Calcule la variance d'une liste de nombres.
    
    Args:
        values (list of float): Liste des valeurs numériques.
    
    Returns:
        float: Variance des valeurs. Retourne 0 si la liste est vide ou contient moins de 2 éléments.
    """
    n = len(values)
    if n < 2:
        return 0

    mean = sum(values) / n
    squared_diffs = [(x - mean) ** 2 for x in values]
    return sum(squared_diffs) / (n - 1)


def load_csv(filename):
    """
    Charge un fichier CSV dans un DataFrame pandas.

    Args:
        filename : Chemin vers le fichier CSV à charger.

    Returns:
        pd.DataFrame | None: Le DataFrame chargé, ou None en cas d'erreur.
    """
    try:
        return pd.read_csv(filename)
    except Exception as e:
        print(f"{Color.YELLOW}Erreur lors du chargement du fichier : {filename} {Color.ENDC}")
        return None


def clean_csv(filename):
    """
    Charge un CSV avec load_csv, supprime les colonnes inutiles et garde les colonnes numériques
    (dont 'Hogwarts House' si elle est présente), puis supprime les lignes contenant des NaN.
    """
    data = load_csv(filename)

    # Supprimer les colonnes inutiles
    cols_to_drop = ['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])

    # Garder uniquement les colonnes numériques + 'Hogwarts House' si présente
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if 'Hogwarts House' in data.columns:
        numeric_data = data[['Hogwarts House']].join(numeric_data)

    # Supprimer les lignes contenant des valeurs manquantes
    cleaned_data = numeric_data.dropna()

    return cleaned_data


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║          Fonctions relatives aux méthodes d'optimisation              ║
# ║                   par gradient descendant.                            ║ 
# ╚═══════════════════════════════════════════════════════════════════════╝

def gradient_descent(X, y, weights, learning_rate=0.01, epochs=10000, lambda_reg=0.0001):
    m = len(y)
    cost_history = []

    for epoch in range(epochs):
        # Ne pas mélanger ici : utiliser X et y directement
        z = np.dot(X, weights[1:]) + weights[0]
        predictions = sigmoid(z)

        gradient = np.zeros(len(weights))
        gradient[0] = (1 / m) * np.sum(predictions - y).item()
        gradient[1:] = (1 / m) * X.T @ (predictions - y)

        gradient[1:] += (lambda_reg / m) * weights[1:]

        weights -= learning_rate * gradient

        cost = (-1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions)) + (lambda_reg / (2 * m)) * np.sum(weights[1:] ** 2)
        cost_history.append(cost)


    return weights, cost_history


def stochastic_gradient_descent(X, y, weights, learning_rate=0.001, epochs=500, lambda_reg=0.001):
    """Fonction de gradient descendant stochastique pour entraîner le modèle."""
    m = len(y)
    cost_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(m):
            xi = X_shuffled[i:i+1]
            yi = y_shuffled[i:i+1]

            z = np.dot(xi, weights[1:]) + weights[0]
            prediction = sigmoid(z)

            gradient = np.zeros(len(weights))
            gradient[0] = prediction.item() - yi.item()
            gradient[1:] = xi.flatten().T * (prediction - yi).item()

            gradient[1:] += (lambda_reg / m) * weights[1:]

            weights -= learning_rate * gradient

    return weights, cost_history

def mini_batch_gradient_descent(X, y, weights, learning_rate=0.001, epochs=500, lambda_reg=0.001, batch_size=128):
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

            z = np.dot(xi, weights[1:]) + weights[0]
            predictions = sigmoid(z)

            gradient_dw = np.zeros(len(weights))
            gradient_dw[0] = np.mean(predictions - yi).item()
            gradient_dw[1:] = (1 / batch_size) * xi.T @ (predictions - yi)

            gradient_dw[1:] += (lambda_reg / m) * weights[1:]

            weights -= learning_rate * gradient_dw

        z = np.dot(X, weights[1:]) + weights[0]
        predictions = sigmoid(z)
        cost = (-1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions)) + (lambda_reg / (2 * m)) * np.sum(weights[1:] ** 2)
        cost_history.append(cost)

        if progress_bar:
            progress_bar.update(1)

    return weights, cost_history

def sigmoid(z):
    """La fonction sigmoïde pour la classification binaire."""
    return 1 / (1 + np.exp(-z))


def save_weights(weights, filename):
    """Sauvegarde des poids dans un fichier."""
    with open(filename, 'w') as f:
        for cls, w in weights.items():
            f.write(f"{cls}:{','.join(map(str, w))}\n")

def preprocess_training_data(file_path):
    """Charge et nettoie les données (normalisation et sélection des colonnes pertinentes)."""
    data = pd.read_csv(file_path)

    # Sélection des colonnes pertinentes
    features = ['Astronomy', 'Defense Against the Dark Arts',
                'Divination', 'Ancient Runes', 'Charms', 'Flying']
    
    X = data[features].values
    y = data['Hogwarts House'].values
    
    # Remplacer les NaN par la moyenne de chaque colonne
    X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

    # Normalisation Min-Max des données
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min)
    
    return X, y, list(data['Hogwarts House'].unique())


def preprocess_test_data(file_path):
    """Charge et nettoie les données de test (normalisation Z-score maison)."""
    data = pd.read_csv(file_path)

    features = ['Astronomy', 'Defense Against the Dark Arts',
                'Divination', 'Ancient Runes', 'Charms', 'Flying']
    
    X = data[features].values

    # Remplacer les NaN par la moyenne de chaque colonne
    X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

    # Normalisation Z-score maison
    X = z_score_normalize(X)
    
    return X


def z_score_normalize(X):
    """
    Normalise les données X en appliquant la standardisation (Z-score).
    Centre chaque colonne en soustrayant la moyenne, puis divise par l'écart-type.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Éviter la division par zéro au cas où std = 0
    std[std == 0] = 1
    X_norm = (X - mean) / std
    return X_norm


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


def softmax(x):
    """Calcul de la fonction softmax pour obtenir des probabilités entre 0 et 1."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)