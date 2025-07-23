import pandas as pd
import matplotlib.pyplot as plt
import sys
from utils import variance, load_csv, clean_csv  # clean_csv ajouté ici

def find_homogeneous_course(data):
    list_of_course = [col for col in data.columns if col != 'Hogwarts House']
    
    homogeneous_course = None
    min_variance = float('inf')

    for course in list_of_course:
        house_scores = []
        for house, group in data.groupby('Hogwarts House'):
            values = group[course].dropna().tolist()
            v = variance(values)
            house_scores.append(v)

        total_variance = sum(house_scores) / len(house_scores)
        if total_variance < min_variance:
            min_variance = total_variance
            homogeneous_course = course
    
    return homogeneous_course

def plot_histogram(data, course):
    plt.figure(figsize=(10, 6))
    for house in data['Hogwarts House'].unique():
        subset = data[data['Hogwarts House'] == house]
        plt.hist(subset[course], bins=20, alpha=0.5, label=house)

    plt.title(f'Histogramme des scores pour {course}')
    plt.xlabel('Scores')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(filename):
    data = clean_csv(filename)
    homogeneous_course = find_homogeneous_course(data)
    if homogeneous_course:
        print(f"La matière avec la distribution de notes la plus homogène est : {homogeneous_course}")
        plot_histogram(data, homogeneous_course)
    else:
        print("Aucune matière homogène trouvée.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py dataset_train.csv")
    else:
        main(sys.argv[1])
