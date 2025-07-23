import pandas as pd
import matplotlib.pyplot as plt
from utils import clean_csv

def find_most_similar_features(data):
    """
    Trouve les deux colonnes numériques les plus corrélées dans un DataFrame.
    """
    correlation_matrix = data.drop(columns='Hogwarts House').corr()

    max_corr = 0
    feature1 = None
    feature2 = None

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            corr_value = abs(correlation_matrix.iloc[i, j])
            if corr_value > max_corr:
                max_corr = corr_value
                feature1, feature2 = correlation_matrix.columns[i], correlation_matrix.columns[j]

    return feature1, feature2

def plot_features(data, feature1, feature2):
    """
    Affiche un scatter plot des deux features données, coloré par maison.
    """
    colors = {'Gryffindor': 'red', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue', 'Slytherin': 'green'}

    plt.figure(figsize=(10, 6))

    for house, color in colors.items():
        house_data = data[data['Hogwarts House'] == house]
        plt.scatter(house_data[feature1], house_data[feature2], label=house, color=color, alpha=0.6)

    plt.title(f'Scatter Plot de {feature1} et {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend(title='Hogwarts House')
    plt.grid(True)
    plt.show()

def main(filename):
    data = clean_csv(filename)
    feature1, feature2 = find_most_similar_features(data)
    plot_features(data, feature1, feature2)

if __name__ == "__main__":
    filename = 'dataset_train.csv'
    main(filename)
