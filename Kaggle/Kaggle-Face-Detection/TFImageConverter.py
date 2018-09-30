import tensorflow as tf
import zipfile as zf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

PathTrain = 'input/trainDataCopy/'                # Input TRAIN data
PathTrainLabels = 'input/train_labels.csv'    # TRAIN labels
PathTest = 'input/testData/'                  # Input TEST data
PathTestLabels = 'input/beHappy.csv'          # TEST labels

### -------- CHOOSE MODE -------- ###
def _mode(InputData):
  
  if (InputData == 'train'):
    path = PathTrain

  elif (InputData == 'test'):
    path = PathTest
  else:
    print("For sys.argv[1] type 'train' or 'test'!!")

  return(path)

### --------INITIALIZATION -------- ###
def _init(ActualPath):

  if (ActualPath == PathTrain):
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

  elif (ActualPath == PathTest):
    # Read in test_labels.csv dataset
    TestLabelsDataFile = pd.read_csv(PathTestLabels)

    # `ImageLabels[i]` is the label for the image in `TestLabelsDataFile[i].
    ImageLabels = TestLabelsDataFile['isHappy'].values

    # A vector of filenames
    ImageIndex = TestLabelsDataFile['ID'].values
    # List for contain filenames
    FileNamesList = []
    # Cycle through filenames
    for i in range(len(ImageIndex)):
      FileNamesList.append(ActualPath + str(i+9001) + '.png')
    
  else:
    print("Szar vagy.")

  return(FileNamesList, ImageLabels)


### -------- MAIN -------- ###
def __main__(mode):
  path = _mode(mode)
  FileNames, ImageLabels = _init(path)

  return(FileNames, ImageLabels)
'''
### -------- TESTING -------- ###
InputMode = sys.argv[1]
index = int(sys.argv[2])

FileNames, ImageLabels = __main__(InputMode)
ExampleImage = plt.imread(FileNames[index])
plt.title("This picture's happiness level is: " + str(ImageLabels[index]))
plt.imshow(ExampleImage)
plt.show()
'''