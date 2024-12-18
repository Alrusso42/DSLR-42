import pandas as pd
import matplotlib.pyplot as plt

def find_most_similar_features(data):
    numeric_data = data.select_dtypes(include='number')
    correlation_matrix = numeric_data.corr()

    max_corr = 0
    feature1 = None
    feature2 = None

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > max_corr:
                max_corr = abs(correlation_matrix.iloc[i, j])
                feature1, feature2 = correlation_matrix.columns[i], correlation_matrix.columns[j]

    return feature1, feature2

def main(filename):
    data = pd.read_csv(filename)
    
    feature1, feature2 = find_most_similar_features(data)

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

if __name__ == "__main__":
    filename = 'dataset_train.csv'
    main(filename)
