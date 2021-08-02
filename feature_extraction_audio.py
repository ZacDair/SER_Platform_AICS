# External Imports
import librosa
import numpy as np
import time
import inspect

# Project Level Imports
import config


def getRawAudioSignal(filepath, duration, sample_rate):
    vector, sr = librosa.load(filepath, duration=duration, res_type=config.cfg["audio_res_type"])
    return vector, sr


def get_mfcc_cc_zcr_numpy(raw_audio_vector, sample_rate, n_mfcc):
    mfccs = np.mean(librosa.feature.mfcc(y=raw_audio_vector, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0)
    zcR = np.mean(librosa.feature.zero_crossing_rate(y=raw_audio_vector).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_cens(y=raw_audio_vector).T, axis=0)
    feature = np.concatenate((mfccs, zcR))
    return np.concatenate((feature, chroma))


def get_mfcc_cc_zcr(raw_audio_vector, sample_rate, n_mfcc):
    return {"mfccs": librosa.feature.mfcc(y=raw_audio_vector, sr=sample_rate, n_mfcc=n_mfcc),
            "zcr": librosa.feature.zero_crossing_rate(y=raw_audio_vector),
            "chroma_cens": librosa.feature.chroma_cens(y=raw_audio_vector)}


def getFeaturesFromList(raw_audio_vector, featureList, argDict):
    res = {}
    for feature in featureList:
        try:
            func = config.cfg["audio_feature_funcs"][feature]
            if feature in argDict:
                for arg in argDict[feature]:
                    setattr(func, arg, argDict[feature][arg])

            res[feature] = func(raw_audio_vector)

        except KeyError:
            print("ERROR - No Function found with name:", feature)
            config.showAudioExtractionFunctions()
            exit(-1)
    return res


def extractFeatures(dataDF, featureList, argDict, verbosity):
    print("Starting audio feature extraction\nFeatures:", featureList)
    print("Warning - This may take some time...")
    features = {}
    for row in dataDF.itertuples():
        if verbosity:
            named_tuple = time.localtime()  # get struct_time
            time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
            print("Extracting Features for:", row.Index, "at", time_string)

        audioVector, sr = getRawAudioSignal(row.Index, 3, 25000)
        features[row.Index] = getFeaturesFromList(audioVector, featureList, argDict)

        # Quit early for speed purposes in dev
        if len(features) == config.cfg["dev_audio_limit"]:
            break

    # Concat our new column of features for each audio file
    dataDF['features'] = dataDF.index.map(features)
    return dataDF


