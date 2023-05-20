import pandas as pd
import numpy as np
from quick_ds import Statistics 


# _____EDA_____
heartdisease = pd.read_csv('course_projects\heart_disease_research_project\heart_disease.csv')
yes_hd = heartdisease[heartdisease.heart_disease == 'presence']
no_hd = heartdisease[heartdisease.heart_disease == 'absence']

print(heartdisease.head())
print(yes_hd.head())
print(no_hd.head())

# Looking for average cholesterol in both diseased and non diseased patients
yes_avg = np.average(yes_hd.chol)
no_avg = np.average(no_hd.chol)
print("The average cholesterol for patients WITH heart disease is: {}".format(yes_avg))
print("The average cholesterol for patients WITHOUT heart disease is: {}".format(no_avg))

# Analyzing average thalach for both diseased and non diseased patients
yes_thalach = np.average(yes_hd.thalach)
no_thalach = np.average(no_hd.thalach)
print("The average thalach for patients WITH heart disease is: {}".format(yes_thalach))
print("The average thalach for patients WITHOUT heart disease is: {}".format(no_thalach))

# Getting the total amount of 1's and 0's from fbs column
print(yes_hd.fbs.value_counts())
print(no_hd.fbs.value_counts())

#_____Asking Questions_____
# 1. Do people with heart disease have high cholesterol levels, above 240 mg/dl, on average?
#   To answer this, we are going to create a null and alternative hypothesis 
#   Null: People with heart disease have an average cholesterol level that is greater than 240 mg/dl
#   Alt: People with heart disease have an average cholesterol greater than 240 mg/dl
#   The mean chol for heart disease positive is: 

# Taking a 1 side approach to each of these
stats = Statistics()
print("Heart disease patients sample t test results: {}".format(stats.sample_t_test(yes_hd, 'chol', 240)))
# Results came back as a number lower than .05, meaning the difference was significant and that heart disease patients 
# more than likely have cholesterol numbers higher than 240 mg/dl

# 2. Do people WITHOUT heart disease have cholesterol levels of 240 mg/pl on average?
print("Non diseased patients sample t test results: {}".format(stats.sample_t_test(no_hd, 'chol', 240)))
# Results came back as a number MUCH greater than .05, meaning the difference was not significant enough to conclude that 
# non diseased patients have a heartrate of 240 mg/dl

# 3. How many patients are present within the TOTAL dataset, not just yes_hd and no_hd
total_patients = len(heartdisease)
print("Total amount of patients: {}".format(total_patients))

# 4. Calculating the number of patients with fbs higher than 120 mg/dl. They are indicated with 1's, while the patients
#   with 0's indicate that their blood sugar was LESS than 120 mg/dl
higher_than_threshold = len(heartdisease[heartdisease.fbs == 1])
lower_than_threshold = len(heartdisease[heartdisease.fbs == 0])
print("Patients higher than fbs threshold: {}".format(higher_than_threshold))
print("Patients lower than threshold: {}".format(lower_than_threshold))
print("Percentage of experimental population that has high FBS: {}%".format(higher_than_threshold/total_patients))
print("Percentage of total population that has high FBS: {}%".format((.08 * 332915073) / 332915073))
print("8% of the experiment population is: {} ".format(.08 * total_patients))

# 5. Does this sample come from a population in which the rate of fbs > 120 mg/dl is equal to 8%?
#   Going to use a binomial test to see...
print(stats.my_binom(higher_than_threshold, total_patients, .08))
from scipy.stats import binom_test
print(binom_test(higher_than_threshold, total_patients, .08, alternative='greater'))
