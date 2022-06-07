# AICS - Emotion Recognition

This repository contains the code to create and conduct emotion recognition experiments, on uni-modalities(text, speech) and a combined approach for bi-modal analysis.

## Installation

1. Install [Python 3.7](https://www.python.org/downloads/release/python-370/).
1. Clone this repository.
1. Install the required dependencies:
    * This can be done with: ```python setup.py build``` and then ```python setup.py install```
    * or using the [requirements.txt](https://github.com/ZacDair/emo_detect/blob/main/requirements.txt) file
1. Add an audio dataset following the instructions below

## Adding an Audio Dataset
1. Download an audio dataset, and place the contents in the datasources directory.  
*Below is an example of how the datasources should look*  
```Bash
|           
+---datasources
|   +---EMO_DB
|   |   |   EMO_DB_KEYS.json   
|   |   \---EMO_DB_DATA
|   |       \---wav
|   |               16b10Tb.wav
|   |               16b10Td.wav
|   |               16b10Wa.wav
|   |               16b10Wb.wav
|   |               
|   +---RAVDESS
|   |   |   RAVDESS_KEYS.json  
|   |   \---RAVDESS_DATA
|   |       \---Audio_Speech_Actors_01-24
|   |           +---Actor_01
|   |           |       03-01-01-01-01-01-01.wav
|   |           |       03-01-01-01-01-02-01.wav
|   |           |       03-01-01-01-02-01-01.wav
|   |           |       03-01-01-01-02-02-01.wav

```
**NOTE:** The DATASOURCE_NAME_DATA directory and DATASOURCE_KEYS.json are **important**

2. Create a **DATASOURCE_NAME**_KEYS.json file (replacing **DATASOURCE_NAME** with the appropriate name)
    * For examples see [EMO_DB_KEYS.json](https://github.com/ZacDair/emo_detect/blob/main/datasources/EMO_DB/EMO_DB_KEYS.json) and [RAVDESS_KEYS.json](https://github.com/ZacDair/emo_detect/blob/main/datasources/RAVDESS/RAVDESS_KEYS.json)
1. Populate the _KEYS.json with the desired identifiers
1. Create or run an experiment


## Creating or Running Experiments

An example experiment is defined as ```experiment_1()``` this experiment provides an end to end emotion classification on a given dataset and selected audio features.  
This involves the labelling of the audio files as per their filename identifiers and the _KEYS.json file, acoustic feature extraction using a defined list of features, the creation training and testing of a CNN for emotive classification, and finally, the evaluation metrics of the experiment are then persisted to the disk.

### Customizing or Editing ```experiment_1()```
A number of parameters and values can be changed tweaking many aspects of the project.  
Such as the input datasource: `dataOriginName = "EMO_DB"` on line 22 of [main.py](https://github.com/ZacDair/emo_detect/blob/69f083e026dbd997b2df8c1d001fff25052f0305/main.py#L22)  

Audio feature hyperparameters such as sample rate or other important [librosa](https://librosa.org/) function arguments: `argDict = {'mfcc': {'n_mfcc': 12, 'sr': 48000}` on line 42 of [main.py](https://github.com/ZacDair/emo_detect/blob/69f083e026dbd997b2df8c1d001fff25052f0305/main.py#L42)  

Duration, Sample Rate, Verbosity of the feature extraction process: `dataDF = feature_extraction_audio.extractFeatures(dataDF, featureSet, argDict, True, 48000, 4)` the last three parameters on line 50 of [main.py](https://github.com/ZacDair/emo_detect/blob/69f083e026dbd997b2df8c1d001fff25052f0305/main.py#L50)  

Emotions or other labels can be altered: `dataDF['emotion'] = dataDF['emotion'].replace("calm", "neutral")` replacing **calm** with **neutral** on line 63 of [main.py](https://github.com/ZacDair/emo_detect/blob/69f083e026dbd997b2df8c1d001fff25052f0305/main.py#L63)  

The audio feature or features to use: `featureDataFrame = dataDF['mfcc'].values.tolist()` on line 65 of [main.py](https://github.com/ZacDair/emo_detect/blob/69f083e026dbd997b2df8c1d001fff25052f0305/main.py#L68)

Model parameters such as batch size or epoch length: `model_creation_audio.run_model_audio(featureDataFrame, dataDF, "emotion", 5, dataOriginName, 128, 150)` the last two parameters on line 77 of [main.py](https://github.com/ZacDair/emo_detect/blob/69f083e026dbd997b2df8c1d001fff25052f0305/main.py#L77)  

More granular parameters can be changed inside the respective .py files (model params in model_creation_audio.py, feature extraction params in feature_extraction_audio.py, etc)

Additionally, the [config.py](https://github.com/ZacDair/emo_detect/blob/main/config.py) file contains the dictionary `cfg` which contains a number of useful paths and other data that can be edited if needed (for example the save location of models)


## Viewing Results
The experiment outputs the results as it goes, however, to review these post run the [results](https://github.com/ZacDair/emo_detect/tree/main/results) directory will contain subdirectories for each experiment run, detailing the dataset and time conducted, inside this directory will be the train/test history, confusion matrix, classification reports, and the models.


## Audio Datasets
[RAVDESS](https://zenodo.org/record/1188976#.YRJD6IhKiiM)  
[SAVEE](http://kahlan.eps.surrey.ac.uk/savee/)  
[EMO_DB](http://emodb.bilderbar.info/start.html)  
[MELD](https://affective-meld.github.io/)  
[IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm)

## Example experiment

```python
def experiment_1():

    """Raw Audio Data Labelling"""

    # Define parameters to use for labelling
    labellingFilename = "Labelled_EMO_DB_AUDIO"
    feautreOutputFilename = "mfcc_data"
    dataOriginName = "EMO_DB"

    # Check for presence or absence of the specified file (create or load file)
    if io_operations.checkIfFileExists(labellingFilename+".csv", dataOriginName):
        dataDF = io_operations.loadDataset(labellingFilename, dataOriginName)
    else:
        # Load, Label and if needed transcribe an audio dataset
        dataDF = datasource_analysis.identifyData(dataOriginName, "AUDIO", ".wav")
        
        # Persist the found files and associated values to disk
        io_operations.saveDataset(dataDF, labellingFilename, dataOriginName)
        

    """Audio Feature Extraction"""

    # Define the list of features, and the required arguments (Originates from Librosa)
    featureSet = ["mfcc"]
    argDict = {'mfcc': {'n_mfcc': 12, 'sr': 48000}}

    # Check for presence or absence of the specified file (create or load file)
    if io_operations.checkIfFileExists(feautreOutputFilename+".pickle", dataOriginName):
        dataDF = io_operations.loadPickle(feautreOutputFilename, dataOriginName)
    else:
        # Run the feature extraction loop function
        dataDF = feature_extraction_audio.extractFeatures(dataDF, featureSet, argDict, True, 48000, 4)
        
        # Persist the features to disk, in a loadable pickle form, and viewable csv
        io_operations.savePickle(dataDF, feautreOutputFilename, dataOriginName)
        io_operations.saveDataset(dataDF, feautreOutputFilename, dataOriginName)

    """Audio Model Creation"""

    # Extract the audio features from the dataframe and convert to the required shape
    featureDataFrame = dataDF['mfcc'].values.tolist()
    featureDataFrame = np.asarray(featureDataFrame)

    # Run our model code
    model_creation_audio.run_model_audio(featureDataFrame, dataDF, "emotion", 5, dataOriginName, 128, 150)


"""Run the previously defined experiment"""
experiment_1()
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Attribution
When using the contens of this repository in your research, please cite:
@article{https://doi.org/10.48550/arxiv.2112.09596,
  doi = {10.48550/ARXIV.2112.09596},  
  url = {https://arxiv.org/abs/2112.09596},  
  author = {Dair, Zachary and Donovan, Ryan and O'Reilly, Ruairi},  
  keywords = {Sound (cs.SD), Audio and Speech Processing (eess.AS), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, I.2; J.2, 91C99},  
  title = {Linguistic and Gender Variation in Speech Emotion Recognition using Spectral Features},  
  publisher = {arXiv},  
  year = {2021},  
  copyright = {Creative Commons Attribution 4.0 International}
}
