# External Imports
import librosa
import inspect

# Project Level Imports
# None

"""
Storage of various global configurations
Access to these values can be used by importing config.py
"""

cfg = {"datasource_path": "datasources",
       "dataset_save_loc": "datasets",
       "results_save_loc": "results",
       "transcription_method": "google",
       "audio_res_type": "kaiser_best",
       "dev_audio_limit": 0,
       "audio_feature_funcs": dict(inspect.getmembers(librosa.feature, inspect.isfunction))
}

runtimeCfg = {}


def showAudioExtractionFunctions():
    print("HELP - Possible Functions are:", cfg["audio_feature_funcs"].keys())


def showAudioFunctionArgs(functionName):
    print("Attempting to retrieve arguments for function: ", functionName)
    try:
        print(inspect.signature(cfg["audio_feature_funcs"][functionName]))
    except KeyError:
        print("ERROR - No Function found with name:", functionName)
        showAudioExtractionFunctions()
