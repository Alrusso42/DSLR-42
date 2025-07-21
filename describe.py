import sys
from utils import CColor, load_csv, summarize_dataframe, print_summary

def describe(filename):
    data = load_csv(filename)
    if data is None:
        return
    
    summary = summarize_dataframe(data)
    print_summary(summary)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"{CColor.YELLOW}Usage: python3 describe.py dataset_train.csv{CColor.ENDC}")
    else:
        describe(sys.argv[1])
