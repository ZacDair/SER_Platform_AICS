# External Imports
import numpy as np
import itertools

# Project Level Imports
import datasource_analysis
import feature_extraction_audio
import io_operations
import model_creation_audio
import model_evaluation_audio
import config


def experiment_1():
    """ Function running experiment_1: raw audio labelling, audio feature extraction, audio model training & testing"""

    """
                          Raw Audio Data Labelling                                   
    """

    # Define parameters to use for labelling
    labellingFilename = "Labelled_SUBESCO_AUDIO"
    featureOutputFilename = "mfcc(12)_zcr_cc_plus48khz"
    dataOriginName = "SUBESCO"

    # Check for presence or absence of the specified file (create or load file)
    if io_operations.checkIfFileExists(labellingFilename+".csv", dataOriginName):
        dataDF = io_operations.loadDataset(labellingFilename, dataOriginName)
        print(dataDF)
    else:
        # Load, Label and if needed transcribe an audio dataset
        dataDF = datasource_analysis.identifyData(dataOriginName, "AUDIO", ".wav")
        print()
        # Persist the found files and associated values to disk
        io_operations.saveDataset(dataDF, labellingFilename, dataOriginName)
        print()

    """                         
                            Audio Feature Extraction
    """

    # Define the list of features, and the required arguments (Originates from Librosa)
    featureSet = ["mfcc", "zero_crossing_rate", "chroma_cens", "chroma_stft", "melspectrogram", "spectral_contrast", "tonnetz"]
    argDict = {'mfcc': {'n_mfcc': 12, 'sr': 48000}, 'chroma_cens': {'sr': 48000}, "chroma_stft": {'sr': 48000}, "melspectrogram": {'sr': 48000}, "spectral_contrast": {'sr': 48000}, "tonnetz": {'sr': 48000}}

    # Check for presence or absence of the specified file (create or load file)
    if io_operations.checkIfFileExists(featureOutputFilename+".pickle", dataOriginName):
        dataDF = io_operations.loadPickle(featureOutputFilename, dataOriginName)
        print(dataDF)
    else:
        # Run the feature extraction loop function
        dataDF = feature_extraction_audio.extractFeatures(dataDF, featureSet, argDict, True, 48000, 3)
        print()
        # Persist the features to disk, in a loadable pickle form, and viewable csv
        io_operations.savePickle(dataDF, featureOutputFilename, dataOriginName)
        io_operations.saveDataset(dataDF, featureOutputFilename, dataOriginName)
        print(dataDF)

    """                         
                                        Audio Model Creation
    """

    # Extract the audio features from the dataframe
    # Replace calm with neutral
    dataDF['emotion'] = dataDF['emotion'].replace("calm", "neutral")

    # Merge gender and emotion
    # dataDF['emotion'] = dataDF[['gender', 'emotion']].agg('-'.join, axis=1)

    print("Class Distribution:")
    print(dataDF['emotion'].value_counts())

    # Run model for each permutation of the features
    # counter = 1
    # for perm in itertools.permutations(featureSet, len(featureSet)):
    #     for val in list(perm):
    #         if counter == 1:
    #             dataDF['features'] = dataDF[val]
    #         else:
    #             dataDF['features'] = dataDF['features']+dataDF[val]

    # Select our feature and convert to the required shape
    featureDataFrame = dataDF['mfcc'].values.tolist()
    featureDataFrame = np.asarray(featureDataFrame)

    # Run our model code
    model_creation_audio.run_model_audio(featureDataFrame, dataDF, "emotion", 5, dataOriginName+"_MFCC", 128, 150)

    # Load a pre-trained model
    # weightFile = "F:\\emo_detect\\results\\EMO_DB_08-30-2021-16-21-25\\EMO_DB-iter-0\\EMO_DB-iter-0.h5"
    # modelFile = "F:\\emo_detect\\results\\EMO_DB_08-30-2021-16-21-25\\EMO_DB-iter-0\\EMO_DB-iter-0.json"
    # pretrained_model = model_evaluation_audio.loadJsonModel(weightFile, modelFile)
    #
    # model_creation_audio.run_pretrained_model_audio(featureDataFrame, dataDF, "emotion", dataOriginName, pretrained_model)

    # Merge Features into singular vector
    dataDF['feature'] = dataDF['mfcc'] + dataDF['melspectrogram']
    print(dataDF['feature'])

    # Select our feature and convert to the required shape
    featureDataFrame = dataDF['feature'].values.tolist()
    featureDataFrame = np.asarray(featureDataFrame)

    # Run our model code
    model_creation_audio.run_model_audio(featureDataFrame, dataDF, "emotion", 5, dataOriginName+"_MFCC-MELSPEC", 128, 150)


experiment_1()
