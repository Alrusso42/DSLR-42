import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main(filename):
    try:
        data = pd.read_csv(filename)
        
        # Vérifier si la colonne 'Hogwarts House' est présente
        if 'Hogwarts House' not in data.columns:
            raise ValueError("La colonne 'Hogwarts House' est manquante dans le dataset.")
        
        # Garder uniquement les colonnes numériques et la colonne 'Hogwarts House'
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data = data[['Hogwarts House'] + list(numeric_cols)]

        # Supprimer les lignes avec des valeurs manquantes (NaN)
        data = data.dropna()

        colors = {'Gryffindor': 'red', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue', 'Slytherin': 'green'}

        sns.set(style="whitegrid")
        pair_plot = sns.pairplot(data, hue='Hogwarts House', palette=colors, diag_kind="hist")
        pair_plot.fig.suptitle('Pair Plot des Caractéristiques par Maison', y=1.02)


        plt.show()

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{filename}' est introuvable.")
    except ValueError as e:
        print(f"Erreur : {e}")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {e}")

if __name__ == "__main__":
    filename = 'dataset_train.csv'
    main(filename)
