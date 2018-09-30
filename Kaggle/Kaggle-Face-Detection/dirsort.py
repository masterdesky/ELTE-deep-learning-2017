import os
import re
import shutil
import pandas as pd

# GLOBALS
PathTrain = 'input/trainData/'                # Input TRAIN data
PathTrainLabels = 'input/train_labels.csv'    # TRAIN labels
PathTest = 'input/testData/' 
NewDirHappy = 'input/trainData/happy/'
NewDirNotHappy = 'input/trainData/not_happy/'

### -------- CHOOSE MODE -------- ###
def _mode(InputData):
  
  if(InputData == 'train'):
    path = PathTrain

  elif(InputData == 'test'):
    path = PathTest
  
  return(path)


### --------INITIALIZATION -------- ###
def _init(ActualPath):

  if(ActualPath == PathTrain):
    # Read in train_labels.csv dataset
    TrainLabelsDataFile = pd.read_csv(PathTrainLabels)

    # `ImageLabels[i]` is the label for the image in `TrainLabelsDataFile[i].
    ImageLabels = TrainLabelsDataFile['isHappy'].values

    # A vector of filenames
    ImageIndex = TrainLabelsDataFile['ID'].values
    # List for contain filenames
    FileNamesList = []
    # Cycle through filenames
    for i in range(len(ImageIndex)):
        FileNamesList.append(ActualPath + str(i+1) + '.png')

  else:
    print("Errorpls")

  return(FileNamesList, ImageLabels)


### --------MAKE DIRS -------- ###
def _newdirs(FileNames, ImageLabels):

  for i in range(len(FileNames)):
    if(ImageLabels[i] == 1):
      NewFileName = NewDirHappy + str(i+1) + '.png'
      os.rename(FileNames[i], NewFileName)

    elif(ImageLabels[i] == 0):
      NewFileName = NewDirNotHappy + str(i+1) + '.png'
      os.rename(FileNames[i], NewFileName)


### -------- MAIN -------- ###
def __main__(mode):
  path = _mode(mode)
  FileNames, ImageLabels = _init(path)
  _newdirs(FileNames, ImageLabels)

#mode = 'train'
#__main__(mode)

