import pandas as pd
from sklearn.metrics import accuracy_score

def calculate_accuracy(predictions_file, truth_file):
    # Charger les fichiers CSV en prenant la première ligne comme en-tête
    predictions = pd.read_csv(predictions_file, header=0)
    truth = pd.read_csv(truth_file, header=0)

    # Afficher les noms de colonnes pour vérifier leur exactitude
    print("Colonnes du fichier predictions :", predictions.columns)
    print("Colonnes du fichier truth :", truth.columns)

    # Vérifier si la colonne 'Hogwarts House' existe
    if 'Hogwarts House' not in predictions.columns or 'Hogwarts House' not in truth.columns:
        print("Erreur: la colonne 'Hogwarts House' est absente dans l'un des fichiers.")
        return None
    
    # Extraire les valeurs des maisons pour la comparaison
    predicted_labels = predictions['Hogwarts House']
    true_labels = truth['Hogwarts House']

    # Calculer l'accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

def main():
    predictions_file = 'houses.csv'
    truth_file = 'dataset_truth.csv'

    accuracy = calculate_accuracy(predictions_file, truth_file)
    if accuracy is not None:
        print(f"L'accuracy du modèle est : {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
