import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix


class Trainer:
    def __init__(self, model, epochs:int, train_ds:tf.python.data.ops.prefetch_op._PrefetchDataset, test_ds:tf.python.data.ops.prefetch_op._PrefetchDataset, valid_ds:tf.python.data.ops.prefetch_op._PrefetchDataset, callbacks):
        self.model = model
        self.epochs = epochs
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.valid_ds = valid_ds
        self.callbacks = callbacks
        self.history = None

    def train(self):
        self.history = self.model.fit(
        self.train_ds,
        self.epochs,
        validation_data=self.valid_ds,
        callbacks=self.callbacks
    )
        

    def plot_history(self):
        plt.plot(self.history.history['loss'], label='train loss')
        plt.plot(self.history.history['val_loss'], label='val loss')
        plt.legend()
        plt.show()

        plt.plot(self.history.history['accuracy'], label='train accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val acc')
        plt.legend()
        plt.show

    
    def evaluate(self):
        self.model.evaluate(self.test_ds)

    def get_test_data_class(test_path:list, class_names:list):
        names = []
        for i in test_path:
            name = i.split("/")[-2]
            name_idx = class_names.index(name)
            names.append(name_idx)
        names = np.array(names, dtype=np.int32)
        return names


    def show_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_mark = np.arange(len(classes))
        plt.xticks(tick_mark, classes, rotation=45)
        plt.yticks(tick_mark, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.axis]
            print("normalized confusion matrix")

        else:
            print("confusion matrix without normalization")
        
        thresh = cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.xlabel("predicted label")
        plt.ylabel("True label")

    def plot_confusion_matrix(self, x_test):
        prediction = self.model.predict(self.test_ds, verbose=0)
        np.around(prediction)
        y_pred_classes = np.argmax(prediction, axis=1)
        classes = self.get_test_data_class(x_test)
        cm = confusion_matrix(y_true=classes, y_pred=y_pred_classes)
        self.show_confusion_matrix(cm=cm, classes=classes, title="confusion matrix", )


    
