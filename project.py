import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Cleaning and prepping data
pokemon = pd.read_csv('data_science_portfolio_projects\Final_intro_project\pokedex_(Update_04.21).csv')
pokemon.drop(pokemon.iloc[:, 31:50], inplace=True, axis=1)
pokemon.drop(['against_fairy'],axis = 1)
#print(pokemon.head(100))
#print(pokemon.info())

# Investigating stats
stat_focused = pokemon[['name', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]
attack = pokemon[['name','attack']]
defense = pokemon[['name','defense']]
special_attack = pokemon[['name','sp_attack']]
special_defense = pokemon[['name','sp_defense']]
speed = pokemon[['name','speed']]

## Looking for pokemon with exceptional stats
broken_pokemon = stat_focused[(stat_focused.attack > 100) & (stat_focused.defense > 100)
                             & (stat_focused.sp_attack > 100)
                             & (stat_focused.sp_defense > 100)
                             & (stat_focused.speed > 100)]

great_pokemon = stat_focused[(stat_focused.attack > 85) & (stat_focused.defense > 85)
                             & (stat_focused.sp_attack > 85)
                             & (stat_focused.sp_defense > 85)
                             & (stat_focused.speed > 85)]

print("Busted pokemon:")
print(broken_pokemon)
print("Great pokemon:")
print(great_pokemon)
