# Project Level Imports
import datasource_analysis
import feature_extraction_audio
import io_operations


def workflow_1():

    """                      Raw Data Labelling                                   """
    # Check for presence or absence of the specified file (create or load file)
    if io_operations.checkIfFileExists("Labelled_RAVDESS_AUDIO", "RAVDESS"):
        dataDF = io_operations.loadDataset("Labelled_RAVDESS_AUDIO", "RAVDESS")
        print(dataDF)
    else:
        # Load, Label and if needed transcribe an audio dataset
        dataDF = datasource_analysis.identifyData("RAVDESS", "AUDIO", ".wav")
        print()
        # Persist the found files and associated values to disk
        io_operations.saveDataset(dataDF, "Labelled_RAVDESS_AUDIO", "RAVDESS")
        print()

    """                         Feature Extraction                                   """
    # Define the list of features, and the required arguments
    featureSet = ["mfcc", "zero_crossing_rate", "chroma_cens"]
    argDict = {'mfcc': {'n_mfcc': 10, 'sr': 25000}, 'chroma_cens': {'sr': 25000}}

    # Run the feature extraction loop function
    featureDF = feature_extraction_audio.extractFeatures(dataDF, featureSet, argDict, True)
    print()
    io_operations.saveDataset(dataDF, "mfcc_zcr_cc", "RAVDESS")
    print(featureDF)


workflow_1()
