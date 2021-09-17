from re import X
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import prepare_tuple_dataset
from filters import filter_signals
from feature_extraction import calculate_statistical_parameters, split_dataset
def preprocess_data(dataset_root_folder="dataset",sample_frequency=250,order=9,band=[30,100],test_size=0.3):
    signals,labels = prepare_tuple_dataset(dataset_root_folder)
    filtered_signals=filter_signals(signals,sample_frequency,order,band)
    signal_parameters=calculate_statistical_parameters(filtered_signals)
    signal_dataset=np.asarray(signal_parameters)
    x_train,x_test_,y_train,y_test=split_dataset(signal_dataset,labels,test_size)
    return x_train,x_test_,y_train,y_test
    