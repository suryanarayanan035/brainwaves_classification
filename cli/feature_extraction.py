from tqdm import tqdm 
import tensorflow as tf
import numpy as np
from scipy.stats import skew,kurtosis
from sklearn.model_selection import train_test_split

def calculate_statistical_parameters(signals):
    parameters = []
    for signal in tqdm(signals):
        channel_parametrs = []
        for channel in signal:
            max_amp = tf.convert_to_tensor(np.max(channel),"double")
            min_amp  = tf.convert_to_tensor(np.min(channel),"double")
            mean  = tf.convert_to_tensor(np.mean(channel),"double")
            median  = tf.convert_to_tensor(np.median(channel),"double")
            std_dev  = tf.convert_to_tensor(np.std(channel),"double")
            var = tf.convert_to_tensor(np.var(channel),"double")
            rms = tf.convert_to_tensor(np.sqrt(np.mean((signals[0][0]-mean)**2)),"double")
            kurtosis_val=tf.convert_to_tensor(kurtosis(signals[0][0]),"double")
            skew_val = tf.convert_to_tensor(skew(signals[0][0]),"double")
            percentile=tf.convert_to_tensor(np.percentile(channel,100))
            channel_parametrs.append(tf.convert_to_tensor([max_amp,
                                     min_amp,
                                     mean,
                                     median,
                                     std_dev,
                                     var,
                                     rms,
                                     kurtosis_val,
                                     skew_val,
                                     percentile]))
        parameters.append(channel_parametrs)
    return parameters


def split_dataset(signal_parameters,labels,test_size=0.3):
    signal_dataset=np.asarray(signal_parameters)
    reshaped_data=[]
    for i in range(len(signal_dataset)):
        reshaped_data.append(tf.convert_to_tensor(np.expand_dims(signal_dataset[i],-1)))
    x_train,x_test,y_train,y_test = train_test_split(reshaped_data,labels,test_size=0.3)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train,x_test,y_train,y_test