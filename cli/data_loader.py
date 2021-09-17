import os
import numpy as np
import pandas as pd

"""
    This function reads signal from given folders and created a list of signal and labels
"""
def prepare_tuple_dataset(dataset_root_folder="thoughts"):
    signals = []
    labels = []
    class_folders = os.listdir(dataset_root_folder) #folders of seperate signals like left,right and pick_object
    print("Folders Detected")
    timeframe_start = 251
    timeframe_end = 1501
    print(f"Samples Taken from: {timeframe_start} - {timeframe_end}")
    for class_folder in class_folders:
        if class_folder == ".ipynb_checkpoints" or class_folder == "pick_object":
            continue
        for file in os.listdir(dataset_root_folder+"/"+class_folder):
            file_name = dataset_root_folder+"/"+class_folder+"/"+file 
            if file == ".ipynb_checkpoints":
                continue
            else:
                tmp_dataset = pd.read_csv(file_name,skiprows=6,header=None)
                tmp_dataset=tmp_dataset.iloc[timeframe_start:timeframe_end,1:9]
                signals.append(tmp_dataset.T.to_numpy().reshape(8,1250))
                labels.append(np.array(1 if class_folder == "left" else 0))
            
    return signals,labels
