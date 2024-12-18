import pandas as pd
import sys

class CColor:
    YELLOW = '\033[33m'
    WHITE = '\033[37m'
    RED = '\033[31m'
    ENDC = '\033[0m'
    BLUE = '\033[34m'
    GREEN = '\033[32m'

def calculate_statistics(data):
    count = 0
    total = 0
    total_squared = 0
    min_value = float('inf')
    max_value = float('-inf')
    values = []

    for value in data:
        if pd.notnull(value):
            count += 1
            total += value
            total_squared += value ** 2
            values.append(value)
            if value < min_value:
                min_value = value
            if value > max_value:
                max_value = value

    if count == 0:
        return (0, 0, 0, 0, 0, 0, 0, 0)

    # Calcul de la moyenne
    mean = total / count
    
    # Calcul de l'écart-type
    variance = (total_squared / count) - (mean ** 2)
    std_dev = variance ** 0.5

    # Calcul des quantiles
    values.sort()
    q1 = values[int(0.25 * count)] if count > 0 else 0
    median = values[int(0.5 * count)] if count > 0 else 0
    q3 = values[int(0.75 * count)] if count > 0 else 0
    iqr = q3 - q1

    return (count, mean, std_dev, min_value, q1, median, q3, max_value, iqr)

def describe(filename):
    try:
        data = pd.read_csv(filename)
    except Exception as e:
        print(f"{CColor.YELLOW}Erreur lors du chargement du fichier : {e}{CColor.ENDC}")
        return
    
    summary = {}
    
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            stats = calculate_statistics(data[column])
            summary[column] = {
                'Count': stats[0],
                'Mean': stats[1],
                'Std': stats[2],
                'Min': stats[3],
                '25%': stats[4],
                '50%': stats[5],
                '75%': stats[6],
                'Max': stats[7],
                'IQR': stats[8],
            }
        else:
            summary[column] = "Valeur non calculable"
    
    headers = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'IQR']
    print(f"{CColor.BLUE}{'':<30}", end="")
    for header in headers:
        print(f"{CColor.BLUE}{header:^15}", end="")
    print()
    
    for column, values in summary.items():
        print(f"{CColor.BLUE}{column:<30}", end="")
        if isinstance(values, dict):
            for value in values.values():
                print(f"{CColor.GREEN}{value:^15.6f}{CColor.ENDC}", end="")
        else:
            print(f"{' ' * 50}{CColor.YELLOW}{'Valeur non calculable':^15}{CColor.ENDC}", end="")
        print(CColor.ENDC)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"{CColor.YELLOW}Usage: python3 describe.py dataset_train.csv{CColor.ENDC}")
    else:
        describe(sys.argv[1])
