import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from quick_ds import Speedy_Data_Science
from quick_ds import Linear_Regression
from quick_ds import Multiple_Regression

tennisStats = pd.read_csv('model_projects\\tennis_ace_project\\tennis_stats.csv')
