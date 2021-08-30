# External Imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# Project Level Imports
import config


# Extract the sentences and their labels from a dataset
def extractTextFeaturesAndLabels(inputDF, utteranceKey, labelKey):
    utterances = inputDF[utteranceKey].values
    labels = inputDF[labelKey].values
    return utterances, labels


# Change labels into categorical values (returns train labels, test labels and the encoder)
def encodeTextLabels(trainLabels, testLabels):
    lb = LabelEncoder()
    trainLabels = np_utils.to_categorical(lb.fit_transform(trainLabels))
    testLabels = np_utils.to_categorical(lb.fit_transform(testLabels))
    return trainLabels, testLabels, lb


# Create the vectorizer object, fitted to the input data
def createVectorizer(lowercase, inputDF):
    vectorizer = CountVectorizer(min_df=0, lowercase=lowercase)
    vectorizer.fit(inputDF)
    return vectorizer


# Convert the train and test sentences into vectors
def vectorizeSentences(trainSentences, testSentences, vectorizer):
    X_train = vectorizer.transform(trainSentences)
    X_test = vectorizer.transform(testSentences)
    return X_train, X_test

