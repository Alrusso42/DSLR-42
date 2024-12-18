import pandas as pd
import matplotlib.pyplot as plt
import sys

def load_data(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except Exception as e:
        print(f"Erreur lors du chargement du fichier : {e}")
        return None

def find_homogeneous_course(data):
    houses = data['Hogwarts House'].unique()
    scores_columns = [col for col in data.columns if col not in ['Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']]
    
    homogeneous_course = None
    min_variance = float('inf')
    
    for course in scores_columns:
        # Calculer la variance des scores par maison
        house_scores = data.groupby('Hogwarts House')[course].var()
        
        # Prendre la variance totale
        total_variance = house_scores.mean()
        
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
    data = load_data(filename)
    if data is not None:
        homogeneous_course = find_homogeneous_course(data)
        if homogeneous_course:
            print(f"La matière avec la distribution de scores la plus homogène est : {homogeneous_course}")
            plot_histogram(data, homogeneous_course)
        else:
            print("Aucune matière homogène trouvée.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py dataset_train.csv")
    else:
        main(sys.argv[1])
