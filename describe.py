import sys
from utils import Color, load_csv, summarize_dataCsv, print_summary

def describe(filename):
    dataCsv = load_csv(filename)
    if dataCsv is None:
        return
    
    summary = summarize_dataCsv(dataCsv)
    print_summary(summary)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"{Color.YELLOW}Usage: python3 describe.py dataset_train.csv{Color.ENDC}")
    else:
        describe(sys.argv[1])
