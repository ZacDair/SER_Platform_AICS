# External imports
import os
import json
import pandas as pd

# Project level imports
import transcription
import config


# Get all files from a directory and subdirectories, or all files of a specified type (eg: ".wav")
# returns a list of file paths
# def getAllFiles(rootSearchPath, specificExt):
#     filePaths = []
#     # Search dirs and subdirs
#     for root, dirs, files in os.walk(rootSearchPath):
#         for file in files:
#             if specificExt == "":
#                 filePaths.append(os.path.join(root, file))
#             elif file.endswith(specificExt):
#                 filePaths.append(os.path.join(root, file))
#
#     print("Found", len(filePaths), "files in", rootSearchPath)
#     return filePaths


# Get the _KEYS.json file containing the identifier structure
def getKeyIdentifiers(path, datasourceName):
    # Get the key value pairs from the file to begin labelling
    try:
        with open(os.path.join(path, datasourceName + "_KEYS.json"), "r") as keyFile:
            try:
                return json.load(keyFile)
            except json.JSONDecodeError as e:
                print("ERROR - Please ensure the *_KEYS.json file is structured correctly.")
                print(e)
                return ""
    except FileNotFoundError as e:
        print("ERROR - Please ensure the *_KEYS.json file exists.")
        print(e)
        return ""


# Utilising the identifer-value pairs label the files accordingly
def labelAllAudioFiles(identifierDict, rootSearchPath, specificExt):
    # Identify the delimiter to use on the filenames
    delimiter = ""
    if "delimiter" in identifierDict.keys():
        delimiter = identifierDict.pop("delimiter")

    # Identify if statements are given, else use transcription
    hasStatements = False
    if "statements" in identifierDict.keys():
        if identifierDict["statements"] != "":
            hasStatements = True

    # Store the file info as a dict, path as key
    result = {}

    # Search dirs and subdirs
    for root, dirs, files in os.walk(rootSearchPath):
        for file in files:

            # Convert the filename, into it's identifer components
            filenameComponents = os.path.splitext(file)[0]
            if delimiter != "":
                filenameComponents = filenameComponents.split(delimiter)

            if specificExt != "":
                # For each of the file identifier categories (emotion, gender, modality, etc)
                fileInfo = dict.fromkeys(identifierDict.keys())
                for k in identifierDict.keys():
                    # When no statements are given transcribe
                    if k == "statements" and not hasStatements:
                        print("Warning - No statements given, using transcription. This may take some time.")
                        fileInfo[k] = transcription.transcribe(os.path.join(root, file))
                    else:
                        possibleIdentifiers = identifierDict[k]["keyValue"].keys()
                        identifierIndex = identifierDict[k]["index"]
                        identifierSize = identifierDict[k]["size"]
                        # Compare the actual file identifier with the values as defined in *_KEYS.json
                        fileIdValue = filenameComponents[identifierIndex:(identifierIndex+identifierSize)]
                        if fileIdValue in possibleIdentifiers:
                            fileInfo[k] = identifierDict[k]["keyValue"][fileIdValue]
                        else:
                            print("WARNING - Unable to determine filename identifier (", fileIdValue, ")")

                # Store the file path, and the file info dict
                result[os.path.join(root, file)] = fileInfo

    return result


# Utilises a datasource name to identify and file extension and label each data file using keys from an external file
# returns a pandas dataframe of file paths and their corresponding labels or ""
def identifyData(dataName, dataType, fileExt):
    # Store the basepath of the datasource in question
    basepath = os.path.join(config.cfg["datasource_path"], dataName)

    # Display a notice to the user
    print("Starting data identification process\nType:", dataType, "\nLocation:", basepath,"\n")

    # Identify the type of data (Audio - Multiple Files, or Text - Single File)
    if dataType != "AUDIO" and dataType != "TEXT":
        print("ERROR - Please ensure the type is either 'AUDIO' or 'TEXT'")
        exit(-1)

    # Process the audio files - labelling
    if dataType == "AUDIO":
        # Get the identifiers dict
        identifierDict = getKeyIdentifiers(basepath, dataName)
        # Get the files
        dataFiles = labelAllAudioFiles(identifierDict, os.path.join(basepath, dataName + "_DATA"), fileExt)
        # If no data files were found return an empty string
        if len(dataFiles) == 0:
            print("ERROR - No files were returned from the labelling process")
            return ""
        # Use the filename, and key-value identifiers to label
        else:
            print(len(dataFiles), "files were returned from the labelling process")
            return pd.DataFrame.from_dict(dataFiles, orient='index')

    # Load with pandas or alternative and we may still want to use the _KEYS file just adjusted for dataframes
    else:
        try:
            if fileExt.lower() == '.csv':
                return pd.read_csv(os.path.join(basepath, dataName + "_DATA" + fileExt))
            elif fileExt.lower() == '.xlsx':
                return pd.read_excel(os.path.join(basepath, dataName + "_DATA" + fileExt))
            else:
                print("Error - Unsupported file extension ("+fileExt+")\n")
        except FileNotFoundError as e:
            print("Error - No file was found\n", e)


# print(identifyData("RAVDESS", "AUDIO", ".wav"))
