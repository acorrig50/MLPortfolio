import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Literally just a lazy way to quickly plot data, will need to import later? idk
class Speedy_Data_Science():

    #### CURRENT QUANTITATIVE METHODS

    def test_method(self):
        print("Hello, you have succesfully imported Speedy Data Science!")

    # Under construction..
    def min_max_mean_median_mode(self, df, column_name):
        print("Min value: {}".format(df[column_name].min()))
        print("Max value: {}".format(df[column_name].max()))
        print("Mean: {}".format(df[column_name].mean()))
        print("Median: {}".format(df[column_name].median()))
        print("Mode: {}".format(df[column_name].mode()))
        
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

    #### CURRENT CATEGORICAL METHODS

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
    
class Quick_Machine_Learning():
   
    
    # For reference, I should make a function that does all of these steps in one go, but for now I will keep seperate for the sake
    # of utilizing one at a time
   
    #### DATA TRANSFORMATION FUNCTIONS
   
    def centering_display(self, df, column_name: str, xlabel: str):
        # Get the mean of the feature
        mean_distance = np.mean(df[column_name])
        
        # Take the distance array and subtract the mean_distance, this will create a new series
        centered_distance = df[column_name] - mean_distance
        
        # Plot out the newly centered distances and label accordingly
        plt.hist(centered_distance, color='g')
        plt.xlabel(xlabel)
        plt.ylabel("Distance from the Mean")
        plt.show()
        plt.close()
        
    def centering_value(self, df, column_name: str):
        # Get the mean of the feature
        mean_distance = np.mean(df[column_name])
        
        # Take the distance array and subtract the mean_distance, this will create a new series
        centered_distance = df[column_name] - mean_distance
        
        # Return the array holding centered values
        return centered_distance
        
    # Might be useless, don't know yet
    def standardizing(self, df, column_name: str):
        # Find the mean of the feature
        distance_mean = np.mean(df[column_name])
        
        # Finding the std of the dataframe's feature
        distance_standard_deviation = np.std(df[column_name])

        # Take each datapoint in the std(distance) and subtract the mean, then divide by std
        distance_standardized = (df[column_name] - distance_mean) / distance_standard_deviation
        return distance_standardized
    
    def standard_scalar(self, df, column_name: str):
        # Define Scaler variable
        scaler = StandardScaler()
        
        # Reshape the feature
        feature_reshaped = np.array(df[column_name]).reshape(-1,1)
        
        # Scale the array and fit_transform it
        feature_scaled = scaler.fit_transform(feature_reshaped)
        
        return feature_scaled

    def min_max_scaler(self, df, column_name: str):
        mmscaler = MinMaxScaler()
        distance = df[column_name]
        reshaped_distance = np.array(distance).reshape(-1,1)
        distance_normalized = mmscaler.fit_transform(reshaped_distance)
        return distance_normalized
    
    def binning(self, df, binning_column: str, column_name: str, bins: list):
        # Create a new column that is binned, using the bins input as a list
        df[binning_column] = pd.cut(df[column_name], bins, right=False)  
        # Plot the column out by counting its values
        df[binning_column].value_counts().plot(kind='bar')
        plt.show()
        plt.close()
    
    def natural_log(self, df, column_name: str):
        log = np.log(df[column_name])
    
    # x is the set of x-values, and y is the set of y-values
    def get_gradient_at_b(x, y, m, b):
        # Create a variable called diff
        diff = 0
        N = len(x)
        for i in range(0, len(x)):
            y_val = y[i]
            x_val = x[i]
            diff += (y_val - ((m * x_val) + b))

        b_gradient = (-2/N) * diff

        return b_gradient
    def get_gradient_at_m(x, y, b, m):
        N = len(x)
        diff = 0
        for i in range(N):
            x_val = x[i]
            y_val = y[i]
            diff += x_val * (y_val - ((m * x_val) + b))
        m_gradient = -(2/N) * diff  
        return m_gradient
        



print("Debugging successful, Quick DS has no errors... at the moment.")