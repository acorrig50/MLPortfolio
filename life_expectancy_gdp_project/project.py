import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels
import matplotlib.pyplot as plt
import math
from quick_ds import Speedy_Data_Science

sds = Speedy_Data_Science()
sns.set_theme(style='darkgrid')

# ______EDA______

# Establish the frame and gather info 
df = pd.read_csv('programming_practice\ds_course_projects\life_expectancy_gdp_project\\all_data.csv') 
print(df.head())
print(df.info())

# Divide the frame into smaller frames by specifying the value in the country column
chile = df[df.Country == 'Chile']
china = df[df.Country == 'China']
germany = df[df.Country == 'Germany']
mexico = df[df.Country == 'Mexico']
usa = df[df.Country == 'United States of America']
zimbabwe = df[df.Country == 'Zimbabwe']

# Create a list containing the divided frames for exploration by iteration later on
df_list = [chile, china, germany, mexico, usa, zimbabwe]

# Find the average LE of each country 
chile_average_le = np.average(chile['Life expectancy at birth (years)'])
china_average_le = np.average(china['Life expectancy at birth (years)'])
germany_average_le = np.average(germany['Life expectancy at birth (years)'])
mexico_average_le = np.average(mexico['Life expectancy at birth (years)'])
usa_average_le = np.average(usa['Life expectancy at birth (years)'])
zimbabwe_average_le = np.average(zimbabwe['Life expectancy at birth (years)'])

# Create a list with the average LE of each country
country_le_list = [chile_average_le, china_average_le, germany_average_le, 
                   mexico_average_le, usa_average_le, zimbabwe_average_le
]


# ____GRAPHING_____
# Plot out basic line plot to see if LE rises as years go by

# USA
plt.plot(usa.Year, usa['Life expectancy at birth (years)'])
plt.title("American Life expectancy at birth (years)")
#plt.show()
plt.close()

plt.plot(usa.Year, usa.GDP)
plt.title("Change in American GDP Over Time")
##plt.show()
plt.close()

# CHINA
plt.plot(china.Year, china['Life expectancy at birth (years)'])
plt.title("Chinese Life Expectancy at Birth")
##plt.show()
plt.close()

plt.plot(china.Year, china.GDP)
plt.title('Change in Chinese GDP Over Time')
#plt.show()
plt.close()

# GERMANY
plt.plot(germany.Year, germany['Life expectancy at birth (years)'])
plt.title("German Life Expectancy at Birth")
#plt.show()
plt.close()

plt.plot(germany.Year, germany.GDP)
plt.title("Change in German GDP Over Time")
#plt.show()
plt.close()

# Mexico
plt.plot(mexico.Year, mexico['Life expectancy at birth (years)'])
plt.title("Mexican Life Expectancy at birth")
#plt.show()
plt.close()

plt.plot(mexico.Year, mexico.GDP)
plt.title("Change in Mexican GDP Over Time")
#plt.show()
plt.close()

# _____Charting GDP Against LE_____
plt.plot(usa.GDP, usa['Life expectancy at birth (years)'])
plt.title("GDP VS Life Expectancy")
plt.show()
plt.close()

# *** MEXICO ***
plt.plot(mexico.GDP, mexico['Life expectancy at birth (years)'], color='red', linestyle='--')
plt.title("Mexican GDP VS Life Expectancy")
plt.xlabel("GDP")
plt.ylabel("Life Expectancy")
plt.show()
plt.close()


# _____Comparing Countries stats_____
sns.scatterplot(data= usa, x=usa.GDP, y=usa['Life expectancy at birth (years)'], color='green')
sns.scatterplot(data= mexico, x=mexico.GDP, y=mexico['Life expectancy at birth (years)'], color='red')
plt.legend(['USA','Mexico'])
plt.show()
plt.close()

sns.catplot(data = usa, kind='violin', x='Year', y='Life expectancy at birth (years)', hue='Country' ,split=True)
plt.show()
plt.close()













