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

#___________Creating import callers___________
sds = Speedy_Data_Science()
lr = Linear_Regression()
mlr = Multiple_Regression()


#___________EDA ____________
tennisStats = pd.read_csv('model_projects\\tennis_ace_project\\tennis_stats.csv')
#print(tennisStats.head())
#print(tennisStats.info())

# Establish different frames to compare, taking from the main frame
rankings_and_winnings = tennisStats[['Player','Ranking','Winnings']]
rankings_and_winnings = rankings_and_winnings.sort_values('Ranking')
print(rankings_and_winnings.to_string())

players_aces_winnings = tennisStats[['Player','Aces','Winnings']]
players_aces_winnings = players_aces_winnings.sort_values('Aces')
print(players_aces_winnings.head())

# ______________________Plotting charts___________________________
#sds.scatter(rankings_and_winnings, 'Ranking', 'Winnings', 'Player Rankings', 'Player Winnings')
#lr.lr_model(tennisStats, 'ServiceGamesWon', 'Winnings', 'Service Games Won', 'Winnings')

# ______________________Not Plotting Charts________________________
