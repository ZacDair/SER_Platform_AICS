# External Imports
import os
import pandas as pd

# Project Level Imports
import config


# Save a dataset from Pandas Dataframe to specified location
def saveDataset(dataset, filename, origin):
    print("Saving File ("+filename+")...")
    # Join the paths
    saveLoc = os.path.join(config.cfg["dataset_save_loc"], origin)
    filePath = os.path.join(saveLoc, filename)

    # Check existence of dataset directory (ie: RAVDESS or SAVEE etc)
    if not os.path.exists(saveLoc):
        print("IO Log - Save Directory was missing, now created.")
        os.mkdir(saveLoc)

    # Increment a counter and write the file preventing duplication
    i = 1
    while os.path.exists(filePath+".csv"):
        filePath = filePath+"-"+str(i)
        i += 1

    dataset.to_csv(filePath+".csv")
    print("IO Log -", filePath+".csv", "Saved to disk.")


def loadDataset(filename, origin):
    print("Loading File (" + filename + ")...")
    # Join the paths
    saveLoc = os.path.join(config.cfg["dataset_save_loc"], origin)
    filePath = os.path.join(saveLoc, filename)

    # Check existence of dataset directory (ie: RAVDESS or SAVEE etc)
    if not os.path.exists(saveLoc):
        print("IO ERROR - Save Directory ("+saveLoc+") was missing.")
        exit(-1)

    # Try and read the file
    try:
        return pd.read_csv(filePath+".csv", header=[0], index_col=[0])
    except FileNotFoundError:
        print("IO ERROR - File ("+filePath+".csv) was not found.")
        exit(-1)


def checkIfFileExists(filename, origin):
    # Join the paths
    saveLoc = os.path.join(config.cfg["dataset_save_loc"], origin)
    filePath = os.path.join(saveLoc, filename+'.csv')
    if os.path.exists(filePath):
        return True
    else:
        return False




