# External Imports
from keras.layers import Dense, Conv1D, Flatten, Dropout, Activation, MaxPooling1D, BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn import metrics
from keras.backend import clear_session
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import pandas as pd


# Project Level Imports
import model_evaluation_audio


# Encode the classification labels, return both the labels and the encoder
def encodeLabels(dataset, labelKey):
    lb = LabelEncoder()
    labels = np_utils.to_categorical(lb.fit_transform(dataset[labelKey]))
    return labels, lb


# Decode the predictions into the original classification labels
def decodePredictions(predictions, labelEncoder):
    predWeights = predictions
    predictions = predictions.argmax(axis=1)
    originalPreds = predictions.astype(int).flatten()
    predictions = (labelEncoder.inverse_transform(originalPreds))
    return pd.DataFrame({'predictedValues': predictions})


# Create a CNN model using the specified structure
def model_create_CNN(inputShape, outputShape):
    model = Sequential()
    model.add(Conv1D(256, 5, padding='same',
                     input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))  # 0.8 - lead to lower accuracy, less spread
    # model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))  # 0.2 - lead to lower accuracy, less spread
    model.add(Flatten())
    model.add(Dense(outputShape))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


# Fit the model to our data (returns the model history)
def model_fit_CNN(model, x_train, y_train, batchSize, epochs, x_test, y_test):
    # Changing Dimension for CNN model
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    cnn_history = model.fit(x_train, y_train, batch_size=batchSize, epochs=epochs, validation_data=(x_test, y_test), verbose=True)
    return cnn_history


# Run the audio model
def run_model_audio(featureDF, fullDF, labelKey, kfoldSplits, origin, batchSize, epochLimit):

    labelDF, lEncoder = encodeLabels(fullDF, labelKey)
    originalLabelDF = fullDF[labelKey]

    # K-fold init
    kf = KFold(n_splits=kfoldSplits, shuffle=True)
    iteration = 0
    totalAcc = 0
    bestAcc = 0
    accuracies = []
    bestModel = ""
    mX = []
    mY = []
    for trainIndex, testIndex, in kf.split(featureDF):

        xTrain, xTest = featureDF[trainIndex], featureDF[testIndex]
        yTrain, yTest = labelDF[trainIndex], labelDF[testIndex]
        inShape = (featureDF.shape[1], 1)
        AP_Model = model_create_CNN(inShape, yTrain.shape[1])

        modelHist = model_fit_CNN(AP_Model, xTrain, yTrain, batchSize, epochLimit, xTest, yTest)

        xTest = np.expand_dims(xTest, axis=2)
        pred = AP_Model.predict(xTest)

        mX.extend(pred)
        mY.extend(yTest)

        decodedPreds = decodePredictions(pred, lEncoder)
        print("\nPredictions:\n", decodedPreds)

        score = metrics.accuracy_score(originalLabelDF[testIndex], decodedPreds)
        accuracies.append(score)
        if score > bestAcc:
            bestModel = AP_Model
            bestAcc = score
        totalAcc += score
        print("Model Accuracy:", score)
        print("Conf Matrix:\n", metrics.confusion_matrix(originalLabelDF[testIndex], decodedPreds))
        print(metrics.classification_report(originalLabelDF[testIndex], decodedPreds))

        # Store the results from our model
        model_evaluation_audio.storeCnnResults(iteration, origin, modelHist, decodedPreds, originalLabelDF[testIndex], AP_Model)

        clear_session()
        iteration += 1
    #print("Avg Accuracy:", totalAcc / kfoldSplits)
    print("Run Summary:\nMin:", min(accuracies), "\nMax:", max(accuracies), "\nMean:", totalAcc/kfoldSplits)
    return bestModel


# Run the audio model
def run_pretrained_model_audio(featureDF, fullDF, labelKey, origin, AP_Model):

    labelDF, lEncoder = encodeLabels(fullDF, labelKey)
    originalLabelDF = fullDF[labelKey]

    iteration = 0
    mX = []
    mY = []

    featureDF = np.expand_dims(featureDF, axis=2)
    pred = AP_Model.predict(featureDF)

    mX.extend(pred)
    mY.extend(labelDF)

    decodedPreds = decodePredictions(pred, lEncoder)
    print("\nPredictions:\n", decodedPreds)

    score = metrics.accuracy_score(originalLabelDF, decodedPreds)
    print("Model Accuracy:", score)
    print("Conf Matrix:\n", metrics.confusion_matrix(originalLabelDF, decodedPreds))
    print(metrics.classification_report(originalLabelDF, decodedPreds))

    # Store the results from our model
    model_evaluation_audio.storeCnnResults(iteration, origin, "", decodedPreds, originalLabelDF, AP_Model)

    clear_session()

