# External Imports
from itertools import cycle

import matplotlib.pyplot as plt
import os
import time
import numpy as np
from sklearn import metrics
import pickle
from keras.models import model_from_json
import pandas as pd


# Project Level Imports
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import config


# Plot the accuracy vs val accuracy and loss vs val loss
def plot_cnn_history(history, titleDetail):
    plt.style.use('ggplot')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x, acc, 'royalblue', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    # plt.title('Training and validation accuracy' + " " + titleDetail)
    plt.legend()
    plt.plot(x, loss, 'lightsteelblue', label='Training loss')
    plt.plot(x, val_loss, 'rosybrown', label='Validation loss')
    plt.ylim(bottom=0, top=1.5)
    plt.title('CNN Train/Test History - ' + " " + titleDetail)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy/Loss")
    plt.legend()

    saveLoc = os.path.join(config.runtimeCfg["model_result_path"], titleDetail + "-cnnHistory.png")
    print(saveLoc)
    plt.savefig(saveLoc)

    plt.show()


def plot_roc_curve(labels, predictions, titleDetail):

    saveLoc = os.path.join(config.runtimeCfg["model_result_path"], titleDetail + "-ROC.png")
    print(saveLoc)
    plt.savefig(saveLoc)

    plt.show()


# Store confusion matrix
def storeConfMatrix(labels, predictions, titleDetail):
    saveLoc = os.path.join(config.runtimeCfg["model_result_path"], titleDetail + "-confMatrix"+".txt")
    with open(saveLoc, "w") as f:
        f.writelines(np.array2string(metrics.confusion_matrix(labels, predictions)))
    f.close()


# Store classification report
def storeClassReport(labels, predictions, titleDetail):
    saveLoc = os.path.join(config.runtimeCfg["model_result_path"], titleDetail + "-classReport" + ".txt")
    with open(saveLoc, "w") as f:
        f.writelines(metrics.classification_report(labels, predictions))
    f.close()


# Save the model and weights
def saveModel(modelName, model):
    save_dir = os.path.join(config.runtimeCfg["model_result_path"], modelName)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, modelName+".h5")
    model.save(model_path)
    print('IO Log - Saved trained model at %s ' % model_path)
    model_json = model.to_json()
    with open(save_dir + "/" + modelName+".json", "w") as json_file:
        json_file.write(model_json)


# Save the model and weights
def savePickleModel(modelName, model):
    save_dir = os.path.join(config.runtimeCfg["model_result_path"], modelName)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, modelName+".sav")
    pickle.dump(model, open(model_path, 'wb'))
    print('IO Log - Saved trained model at %s ' % model_path)


# Load the models from pickled files
def loadPickledModel(filepath):
    try:
        loaded_model = pickle.load(open(filepath, 'rb'))
        return loaded_model
    except FileNotFoundError:
        print("ERROR - The imported model files were not found")
        exit(1)


# Load model and weights from Json
def loadJsonModel(weightFile, modelFile):
    try:
        jsonFile = open(modelFile, 'r')
        loadedModel = jsonFile.read()
        jsonFile.close()
        loadedModel = model_from_json(loadedModel)

        # load weights
        loadedModel.load_weights(weightFile)
        print("Loaded Model From Disk")
        return loadedModel
    except FileNotFoundError:
        print("ERROR - The imported model files were not found")
        exit(1)


# Persist our model and associated evaluation data
def storeCnnResults(iteration, origin, cnnHistory, predictions, labels, model):
    # Create a new directory
    if iteration == 0:
        named_tuple = time.localtime()  # get struct_time
        time_string = time.strftime("%m-%d-%Y-%H-%M-%S", named_tuple)
        config.runtimeCfg["model_result_path"] = os.path.join(config.cfg["results_save_loc"], origin+"_"+time_string)
        os.mkdir(config.runtimeCfg["model_result_path"])
        print("IO LOG - Storing Model Results in:", config.runtimeCfg["model_result_path"])

    # Store the plotted CNN History, confusion matrix, and classification report
    if os.path.exists(config.runtimeCfg["model_result_path"]):
        modelTitle = origin+"-iter-"+str(iteration)

        # Save the results
        if cnnHistory != "":
            plot_cnn_history(cnnHistory, modelTitle)
        # plot_roc_curve(labels, predictions, modelTitle)
        storeConfMatrix(labels, predictions, modelTitle)
        storeClassReport(labels, predictions, modelTitle)

        # Save the model
        saveModel(modelTitle, model)

