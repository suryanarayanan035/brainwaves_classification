import argparse
import argparse
from preprocess import preprocess_data
def preprocess(dataset_root_folder):
    preprocess_data(dataset_root_folder);

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action")
    parser.add_argument("--dataset",required=False)
    args = parser.parse_args()
    action = args.action
    dataset = args.dataset
    if action == "preprocess":
        preprocess_data(dataset)
    