import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score
def plot_loss_accuracy_chart(history,validation=True):
    plt.plot(history.history['accuracy'])
    if validation==True:
        plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    if validation==True:
        plt.plot(history.history['val_loss']) 
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def print_metrics(y_true,y_predictions):
    print("Confusion Matrix:\n ",confusion_matrix(y_true,y_predictions))
    print("Accuracy Score: ",accuracy_score(y_true,y_predictions))
    print("ROC AUC Score: ",roc_auc_score(y_true,y_predictions))