import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Literally just a lazy way to quickly plot data, will need to import later? idk
class Speedy_Data_Science():

    # Current quantitative methods

    def test_method(self):
        print("Hello, you have succesfully imported Speedy Data Science!")

    def scatter(self, df, column_1: str, column_2: str, xlabel: str, ylabel: str):
        plt.scatter(x=df[column_1], y=df[column_2])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        plt.close()

    def histogram_double(self, df, column_1: str, column_2: str, color1: str, color2: str, label1: str, label2: str, title: str):
        plt.hist(df[column_1], color=color1,
                    label=label1, normed=True, alpha=.5)
        plt.hist(df[column_2], color=color2,
                    lable=label2, normed=True, alpha=.5)
        plt.legend()
        plt.title(title)
        plt.show()
        plt.close()

    def boxplot(self, df, column_1: str, column_2: str):
        sns.boxplot(data=df, x=df[column_1], y=df[column_2])
        plt.show()
        plt.close()

    def covariance(self, df, column_1: str, column_2: str):
        covariance, p = np.cov(df[column_1], df[column_2])
        return covariance

    def correlation(self, df, column_1: str, column_2: str):
        correlation_value = pearsonr(df[column_1], df[column_2])
        return correlation_value

    # Current categorial methods

    def frequency_table(self, df, column_name: str):
        print("Frequency table:")
        print(df[column_name].value_counts())
        print("########")
        print("Frequency proportion table:")
        print(df[column_name].value_counts(normalize=True))

    def count_plot(self, df, column_name: str):
        sns.countplot(x= df[column_name], data=df)
        plt.show()
        plt.close()

    def pie_chart(self, df, column_name: str):
        df[column_name].value_counts().plot.pie()
        plt.show()
        plt.close()

    # Print statement may be a no-no
    def contingency_table(self, df, column_1: str, column_2: str):
        cont_table = pd.crosstab(df[column_1], df[column_2])
        print(cont_table)
        return cont_table

    # Might need work
    def contingency_table_proportions(self, contingency_table, df):
        return other_function / len(df)

    def chi2(self, contingency_table_input, cont_return_value):
        chi2, pval, dof, expected = chi2_contingency(
            contingency_table_input)
        return chi2

    def expected_contingency_table(self, contingency_table_function, cont_return_value):
        chi2, pval, dof, expected = chi2_contingency(
            contingency_table_function)
        return expected

    # Test methods, want to handle multiple categorical metrics at once for speedy analysis

    def cont_prop_chi2_expected(self, df, column_1: str, column_2: str):
        cont_table = pd.crosstab(df[column_1], df[column_2])
        cont_table_prop = cont_table / len(df)
        chi2, pval, dof, expected = chi2_contingency(cont_table)

        return cont_table, cont_table_prop, chi2, expected

# Class for probability, small for now, will expand later
class Probability():
   
    def prob_a_or_b(self, a, b, all_possible_outcomes):
        # probability of event a
        prob_a = len(a)/len(all_possible_outcomes)
        
        # probability of event b
        prob_b = len(b)/len(all_possible_outcomes)
        
        # intersection of events a and b
        inter = a.intersection(b)
        
        # probability of intersection of events a and b
        prob_inter = len(inter)/len(all_possible_outcomes)
        
        # add return statement here
        return (prob_a + prob_b - prob_inter)
    
    def sampling_distribution_graph(self, df, column_name: str, sample_size: int, x_label: str):
        # We wan't to create a list of means, then plot them. Essentially we are
        # looking for the mean of the mean
        sample_size_length = sample_size
        sample_means = []
        
        for i in range(sample_size_length):
            sample = np.random.choice(df[column_name], sample_size_length, replace=False)
            this_sample_mean = np.mean(sample)
            sample_means.append(this_sample_mean)
            
        sns.histplot(sample_means, stat='density')
        plt.axvline(np.mean(sample_means),color='orange',linestyle='dashed', label=f'Mean of the Sample')
        plt.title("Population Distribution of the Mean")
        plt.legend()
        plt.xlabel(x_label)
        plt.show()
        plt.close()
        
    def standard_error(self, samp_size: int, sample_mean):
        # last two 3 lines constitute uncertainty, the code will provide a range at which most averages will fall in
        
        standard_error = np.std(samp_size) / (samp_size**.5)
        print("The standard error is: {}".format(standard_error))
        observed_estimate = standard_error * 1.96
        print("The observed mean is within {} and {}".format(((sample_mean - observed_estimate), (sample_mean + observed_estimate))))
        return standard_error, observed_estimate
    
        
    

print("Debugging successful, Quick DS has no errors... at the moment.")