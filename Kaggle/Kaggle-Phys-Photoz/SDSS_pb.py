# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

pathinput = 'C:/Users/Lordpb/Documents/GitHub/Kaggle-Phys-Photoz/input'
pathtrain = 'C:/Users/Lordpb/Documents/GitHub/Kaggle-Phys-Photoz/input/train.csv.gz'
pathtest = 'C:/Users/Lordpb/Documents/GitHub/Kaggle-Phys-Photoz/input/test.csv.gz'
pathsample = 'C:/Users/Lordpb/Documents/GitHub/Kaggle-Phys-Photoz/input/sampleSubmission.csv'

pathcovar = 'C:/Users/Lordpb/Documents/GitHub/Kaggle-Phys-Photoz/input/my_photoz_test.csv' # BT data
pathoutput = 'C:/Users/Lordpb/Documents/GitHub/Kaggle-Phys-Photoz/output/sdss_predict.csv' # Output file

print(os.listdir(pathinput))
# Any results you write to the current directory are saved as output.

from scipy.optimize import curve_fit  # load a curve fitter

###################
# Fit (Train)

df = pd.read_csv(pathtrain)  # load train data
x = df[['u','g','r','i','z']].values.T  # format x as scipy expects it
y = df['redshift'].values  # format y as scipy expects it

# Define the linear functions to fit
def lin(x,a0,a1,a2,a3,a4,b):
    """Return linear prediciton with 5 variables."""
    return a0*x[0]+a1*x[1]+a2*x[2]+a3*x[3]+a4*x[4]+b

(a0,a1,a2,a3,a4,b),cov = curve_fit(lin,x,y)  # fit it

print('Parameters:',a0,a1,a2,a3,a4,b)  # print parameters
print('Covariances:',cov)  # print covariances


###################
# Predict

testdf =  pd.read_csv(pathtest)  # read test data 
x = testdf[['u','g','r','i','z']].values  # get test x
# predict test y from x
y = a0*x[:,0] + a1*x[:,1] + a2*x[:,2] + a3*x[:,3] + a4*x[:,4] + b

# write y in the sample submission table (order is good)
resultdf =  pd.read_csv(pathsample) 
resultdf['redshift'] =  y  

# save submission (no index column)
resultdf.to_csv(pathoutput, index=False)

