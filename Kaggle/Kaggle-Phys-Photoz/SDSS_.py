### SDSS DATA ANALYSIS
### Redshift estimation from magnitudes



# ------ INITIALIZATION ------
from datetime import datetime
from datetime import timedelta
start = datetime.now()          ### Measuring time
TotalTime = timedelta(0)        ### Measuring total elapsed time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#import git

from sklearn import linear_model
from sklearn import ensemble
from sklearn import neighbors
from sklearn import neural_network
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict

# Defining path for input and output folders and files
PathInputFolder = 'input/'                  # Folder of input data
PathTrain = 'input/train.csv.gz'            # Input TRAIN data
PathTest = 'input/test.csv.gz'              # Input TEST data
PathSample = 'input/sampleSubmission.csv'   # Input SAMPLE data

PathOutputFolder = 'output/'                # Folder of input data
PathOutput = 'output/sdss_predict.csv'      # Output file

# Defining URLs for GitHub repository and folder for push
RepositoryName = 'https://github.com/masterdesky/Kaggle-Phys-Photoz'

print("Files in input directory: ", os.listdir(PathInputFolder))            ### Checking if
print("Files in output directory: ", os.listdir(PathOutputFolder))          ### everything is OK

print("#### Time for initialization: ", datetime.now() - start, '\n')
TotalTime += datetime.now() - start ### Measuring time total

# ------ INITIALIZATION END ------



# ------ DEFINE AND CHOOSE USABLE MODELS FOR REGRESSION ------
def Models():

    # Linear models
    forest = ensemble.RandomForestRegressor(n_estimators=150, min_samples_split=15,min_samples_leaf=3)
    extra = ensemble.ExtraTreesRegressor(n_estimators=150,max_features="auto", max_depth=30,min_samples_split=15,min_samples_leaf=3,)
    linear = linear_model.LinearRegression()
    neural_1 = neural_network.MLPRegressor(hidden_layer_sizes=(120, ), activation='relu',learning_rate_init=0.0001)
    ridge = linear_model.Ridge()
    KN = neighbors.KNeighborsRegressor(n_neighbors=120, weights='uniform')

    # Polynomial models
    polyness = make_pipeline(PolynomialFeatures(3), neural_1)

    ############ MODIFY TO CHOOSE MODEL ############
    model = ridge
    ############ MODIFY TO CHOOSE MODEL ############

    print("Choosen model: ", model, '\n')

    return(model)
# ------ DEFINE AND CHOOSE USABLE MODELS FOR REGRESSION END ------



# ------ READ IN TRAIN DATA AND FIT THEM ON CHOOSEN MODEL ------
def Train(model):

    # Read in parameters
    TrainDataFrame = pd.read_csv(PathTrain)                     # Load train data from .csv
    print("Train data ready!")
    x = TrainDataFrame[['u','g','r','i','z']].values            # Format x as model.fit expects it
    y = TrainDataFrame['redshift'].values                       # Format y as model.fit expects it

    # Fit choosen model
    model.fit(x,y)
    print("Success! Model fitted!")

    return(model, x, y)

# ------ READ IN PARAMTETERS AND FIT THEM ON CHOOSEN MODEL END ------



# ------ READ IN TEST DATA, PREDICT REDSHIFTS, WRITE OUTPUT FILES ------
def Test(model):

    TestDataFrame =  pd.read_csv(PathTest)                 # Read test data
    print("Test data ready!")
    x_test = TestDataFrame[['u','g','r','i','z']].values   # Get test x
    y_pred = model.predict(x_test)                         # Predict test y from x
    print("Test prediction finished!")

    # Write y in the sample submission table (order is good)
    ResultDataFrame =  pd.read_csv(PathSample)
    ResultDataFrame['redshift'] =  y_pred

    # Save submission (no index column)
    ResultDataFrame.to_csv(index=False, mode='w+')

    return(model, y_pred)

# ------ READ IN TEST DATA, PREDICT REDSHIFTS, WRITE OUTPUT FILES END------



# ------ CROSS VALUE PREDICTION ------
def CorrWithCrossVal(model, x, y):

    y_pred = cross_val_predict(model, x, y, cv=5)  

    print('RMSE for cross:',((y-y_pred)**2).mean()**0.5)          # Root Mean Square Error
    print('Pearson corr for cross:',np.corrcoef(y,y_pred)[0,1])   # Pearson correlation coefficient

    return(y_pred)

# ------ CROSS VALUE PREDICTION END ------



# ------ CROSS VALUE PREDICTION ------
def PlottingData(y,y_pred):
    
    plt.plot(y,y_pred,'.')              # Plot data points
    plt.plot([0,1],[0,1])             # y = x line for reference
    plt.xlabel('Z by model')            # Z predicted by using the model above
    plt.ylabel('Z by cross value')      # Z predicted by cross-val-pred
    plt.show()

# ------ PLOTTING ------



# ------ REFRESH GITHUB FILES ------
'''
def GitHub():
    MasterRepo = git.Repo(Repository_name)
    MasterRepo.git.add(u=True)
    MasterRepo.git.commit("Updates after running from terminal")
    MasterRepo.git.push()

'''

# ------ REFRESH GITHUB FILES END ------



# ------ MAIN ------

## CHOOSE MODELS
start = datetime.now() ### Measuring time
model = Models()
print("#### Time for choosing model: ", datetime.now() - start, '\n')
TotalTime += datetime.now() - start ### Measuring time total

## TRAIN DATA
start = datetime.now()       ### Measuring time
model, x, y = Train(model)
print("#### Time for reading in and fitting on train: ", datetime.now() - start, '\n')
TotalTime += datetime.now() - start ### Measuring time total

# TEST DATA
start = datetime.now()       ### Measuring time
model, y_pred_model = Test(model)
print("#### Time for reading in test, predict redshifts and wrting them to files: ", datetime.now() - start, '\n')
TotalTime += datetime.now() - start ### Measuring time total

## CORRELATION WITH CROSSVAL
start = datetime.now()       ### Measuring time
y_pred_cross = CorrWithCrossVal(model, x, y)
print("#### Time for correlating dataset with cross_value: ", datetime.now() - start, '\n')
TotalTime += datetime.now() - start ### Measuring time total

## PLOTTING DATASET
start = datetime.now()       ### Measuring time
PlottingData(y,y_pred_cross)
print("#### Time for correlating with dataset: ", datetime.now() - start, '\n')
TotalTime += datetime.now() - start ### Measuring time total
'''
## UPDATE GITHUB REPOSITORY
start = datetime.now()       ### Measuring time
GitHub()
print("#### Time for updating GitHub repository: ", datetime.now() - start, '\n')
TotalTime += datetime.now() - start ### Measuring time total
'''
## PRINT TOTAL RUNTIME
print("#### Total time for running the whole program: ", TotalTime) ### Measuring time total end

# ------ MAIN END ------