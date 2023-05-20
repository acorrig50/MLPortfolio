import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Literally just a lazy way to quickly plot data, will need to import later? idk
class Speedy_Data_Science():

    #### CURRENT QUANTITATIVE METHODS
     
    # Under construction..
    def min_max_mean_median_mode(self, df, column_name):
        print("Min value: {}".format(df[column_name].min()))
        print("Max value: {}".format(df[column_name].max()))
        print("Mean: {}".format(df[column_name].mean()))
        print("Median: {}".format(df[column_name].median()))
        print("Mode: {}".format(df[column_name].mode()))
        
    def scatter(self, df, column_1: str, column_2: str, xlabel: str, ylabel: str):
        plt.scatter(x=df[[column_1]], y=df[[column_2]], alpha=.4)
        # This line breaks it
        #plt.plot(range(x_beginning, x_end), range(y_beginning,y_end))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        plt.close()
        
    def histogram_single(self, df, column_1: str, color1: str, label1: str, title: str):
        plt.hist(df[column_1], color=color1, label=label1, alpha=.5)
        plt.legend()
        plt.title(title)
        plt.show()
        plt.close()

    def histogram_double(self, df, column_1: str, column_2: str, color1: str, color2: str, label1: str, label2: str, title: str):
        plt.hist(df[column_1], color=color1, label=label1, alpha=.5)
        plt.hist(df[column_2], color=color2, label=label2, alpha=.5)
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
    
    # AKA CLT, used in the Null hypothesis process
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
    
class Data_Transformation():
   
    
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
        
# Needs work
class Linear_Regression():
    # x is the set of x-values, and y is the set of y-values
    # PT1
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
    
    # PT2 / AKA how to find loss
    def get_gradient_at_m(x, y, b, m):
        N = len(x)
        diff = 0
        for i in range(N):
            x_val = x[i]
            y_val = y[i]
            diff += x_val * (y_val - ((m * x_val) + b))
        m_gradient = -(2/N) * diff  
        return m_gradient
    
    # PT3
    def step_gradient(x, y, b_current, m_current):
        b_gradient = get_gradient_at_b(x, y, b_current, m_current)
        m_gradient = get_gradient_at_m(x, y, b_current, m_current)
        b = b_current - (0.01 * b_gradient)
        m = m_current - (0.01 * m_gradient)

        return [b, m]
    
    # PT4
    def gradient_descent(x, y, learning_rate, num_iterations):
        b = 0
        m = 0
        for i in range(num_iterations):
            b, m =step_gradient(b, m, x, y, learning_rate)

        return [b, m]
    
        # def lr_future_plot(self, df, x_column, y_column, x_beg, x_end):
        # Plot the data as is
        line_fitter = LinearRegression()
        
        feature_1 = df[x_column]
        feature_2 = df[y_column]
        feature_1 = feature_1.values.reshape(-1,1)
        feature_2 = feature_2.values.reshape(-1,1)
        
        # Setting proper x and y ranges
        feature_1 = np.array(range(x_beg, x_end))
        feature_2 = np.array(range(y_beg, y_end))
        
        plt.scatter(feature_1, feature_2)
        plt.show()
        
        # Create a line fitter from the sklearn library
        line_fitter.fit(feature_1, feature_2)
        y_predict = line_fitter.predict(feature_1)
        plt.plot(feature_1, y_predict)
        plt.show()
        
        # Setting up new x-axis to reach further out into future numbers
        X_future = np.array(range(x_beg, x_end))
        X_future = X_future.reshape(-1,1)
        # Predicting future outcome
        future_prediction = line_fitter.predict(X_future)
        plt.plot(X_future, future_prediction)
        plt.show()
        plt.close()
        
    def linear_regression_plot_no_df(self, x_column, y_column):
        # Plot the data as is
        plt.plot(df[x_column], df[y_column],'o')
        plt.show()
        plt.clf()
        
        # Create a line fitter from the sklearn library
        line_fitter = LinearRegression()
        line_fitter.fit(df[x_column], df[y_column])
        predicted = line_fitter.predict(df[x_column])
        
        # Plotting the prediction and line of best fit
        plt.plot(df[x_column], predicted)
        plt.show()
        plt.close()
   
    # Most optimized model wrapper 
    def lr_model(self, df, column_1, column_2, x_label, y_label):
        features = df[[column_1]]
        outcome = df[[column_2]]
        features = features.values.reshape(-1,1)
        outcome = outcome.values.reshape(-1,1)
        
        # Plotting line of best fit
        plt.scatter(features, outcome, alpha=.4)
        
        # Setting up training and testing data
        features_training, features_testing, outcome_training, outcome_testing = train_test_split(features, outcome, train_size = .8, test_size=.2)
        l_model = LinearRegression()
        l_model.fit(features_training, outcome_training)
        
        score = l_model.score(features_testing, outcome_testing)
        print("Model scoring: {}".format(score))
        
        model_prediction = l_model.predict(features_testing)
        plt.scatter(outcome_testing, model_prediction, alpha=.4)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        plt.close()
     
    def lr_future_plot(self, df, column_1: str, column_2: str, x_beg: int, x_end: int):
        regr = LinearRegression()
        plt.ion()
        
        X = df[column_1]
        X = X.values.reshape(-1,1)
        y = df[column_2]
        y = y.values.reshape(-1,1)

        plt.scatter(X,y)
        plt.show()


        regr.fit(X,y)
        y_predict = regr.predict(X)

        plt.plot(X, y_predict)
        plt.show()

        X_future = np.array(range(x_beg, x_end))
        X_future = X_future.reshape(-1,1)

        future_predict = regr.predict(X_future)
        plt.plot(X_future, future_predict)
        plt.show()     
        
        score = regr.score(X, y)
        print("Model scoring: {}".format(score))

class Multiple_Regression():
    
    # The data list variable must be a list of numbers that match the data types of the columns we feed to the x value
    # within the function. The list contents must also be surrounded with double brackets: data_list = [[1,2,3, etc]] 
    def multi_lin_reg(self, x, y, data_list: list):
        # Establish the train_test_split_method, seperating the data into training and test units
        x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=.8, test_size=.2)
        
        # Create the linear regression model and feed it the training data
        mlr = LinearRegression()
        mlr.fit(x_train,y_train)
        y_predict = mlr.predict(x_test)    
        
        # Create a prediction for the list fed by parameter and return it
        prediction_variable = mlr.predict(data_list)
        return "Predicted number: %.2f" % prediction_variable 
        
    # To use properly, the main dataframe must be split into two seperate dataframes, the x dataframe holding
    # the indepent variables, and the y dataframe holding the singular dependent variable
    def multi_lin_reg_scatter(self, x, y, x_label: str, y_label: str, title: str):
        # Establish the train_test_split_method, seperating the data into training and test units
        x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=.8, test_size=.2)
        
        # Create the linear regression model and feed it the training data
        mlr = LinearRegression()
        mlr.fit(x_train,y_train)
        y_predict = mlr.predict(x_test)
        
        # and regression loss for both training and testing sets
        print("The training error score: {}".format(mlr.score(x_train,y_train)))
        print("The testing error score: {}".format(mlr.score(x_test,y_test)))
        
        # Give option to make graph or to not make graph
        fork = input("Produce scatterplot? y for yes, n for no:")
        if fork == 'y':
            # Plot the y_test and y_predict variables to compare them, alpha is how clear or unclear the dots appear
            plt.scatter(y_test, y_predict, alpha=0.4)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.show()
            plt.close() 
     
class Logistic_Regression():
    
    # __________ GENERAL USE OF FUNCTION ___________
    # 1. model.fit(features, labels) / 
    # 2. Label is what we are trying to predict using the features, which means features can be MULTIPLE columns
    # 3. Used for binary classification, 0 and 1, yes or no, True or False
    def log_regression(df, features, labels):
        # Establish the x and y axis with columns put into the function
        X = df[[features]]
        y = df[[labels]]
        
        # Transforming the x axis 
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
        # Partition training and testing data 
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25, random_state=27)

        # Create and fit the model
        model = LogisticRegression()
        model.fit(X_train,y_train)
        
        # Printing the predicted outcomes and probabilities for the test data
        print("Outcomes: " + model.predict(X_test))
        print("Probabilities: " + model.predict_proba(X_test)[:,1])
        
        # Creating and printing the confusion matrix
        y_pred = model.predict(X_test)
        print("True classes: {}".format(y_test))
        print("Confusion matrix: {}".format(confusion_matrix(y_test, y_pred)))
        
        # Printing statistics from confusion matrix: accuracy, precision, recall, F1
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        print("Accuracy: {}".format(accuracy_score(y_true,y_pred)))
        print("Precision: {}".format(precision_score(y_true,y_pred)))
        print("Recall: {}".format(recall_score(y_true,y_pred)))
        print("F1: {}".format(f1_score(y_true,y_pred)))
        
        # Save the intercept and coef to new variables
        intercept = model.intercept_
        coef = model.coef_
        
        # Calculate the log odds
        log_odds =intercept + coef * X
        
        # Now we calculate the predicted probability of the two columns
        predicted_probability = np.exp(log_odds)/(1+ np.exp(log_odds))
        ## Can return or plot this line above, not sure what to do with it yet

class Statistics():
    
    # __________ GENERAL USE OF FUNCTION ___________
    # Takes in a frame and a column name, with int or float values and
    # finds the probability of 'prediction_mean' occurence. It then returns 
    # this in the form of 'pval'
    # NOTE: THIS IS ONLY FOR NULL, CHANGE THE 'prediction_mean' VAR TO CREATE AN ALTERNATIVE HYPOTHESIS
    def sample_t_test(self, df, column_name: str, prediction_mean: float):
        from scipy.stats import ttest_1samp
        
        tstat, pval = ttest_1samp(df[[column_name]], prediction_mean)
        return pval

    # __________ GENERAL USE OF FUNCTION ___________
    # Because it is used for binomial categorical simulation, the function takes in two outcomes,
    # either outcome can be str, int, or float. It then uses a size, or how many iterations it will 
    # repeat itself. The outcome_1 and outcome_2 parameters will set the likelihood of option_1 and 2's occurence
    def binomial_simulation(self, option_1, option_2, size: int, outcome_1: float, outcome_2: float):
        binom_sim = np.random.choice([option_1, option_2], size = size, p= [outcome_1, outcome_2])
        return binom_sim
    
    # __________ GENERAL USE OF FUNCTION ___________
    # The function is exactly the same as the binomial_simulation function save that this one appends the results of the binomial to
    # a list that is then averaged out and printed to the console. This helps use the binomial sim at a large scale and fine tune the 
    # probability of outcomes to the most minuscule detail 
    def binomial_simulation_list(self, option_1: str, option_2, size: int, outcome_1: float, outcome_2: float, loop_length: int):
        option_1_list = []
        option_2_list = []
        for i in range(loop_length):
            binom_sim = np.random.choice([option_1, option_2], size = size, p= [outcome_1, outcome_2])
            option_1_list.append(np.sum(binom_sim == option_1))
            option_2_list.append(np.sum(binom_sim == option_2))
        print("Option 1 average: {}".format(np.average(option_1_list) / size))
        print("Option 2 average: {}".format(np.average(option_2_list) / size))
        
    # __________ GENERAL USE OF FUNCTION ___________
    # Instead of returning the values, graphs the outcomes instead
    def binomial_simulation_plot(self, option_1: str, option_2, size: int, outcome_1: float, outcome_2: float, loop_length: int):
        option_1_list = []
        option_2_list = []
        for i in range(loop_length):
            binom_sim = np.random.choice([option_1, option_2], size = size, p= [outcome_1, outcome_2])
            option_1_list.append(np.sum(binom_sim == option_1))
            option_2_list.append(np.sum(binom_sim == option_2))
            
        plt.hist(option_1_list)
        plt.axvline(np.average(option_1_list), color='r')
        plt.show()
        plt.close()

        
        plt.hist(option_2_list)
        plt.axvline(np.average(option_2_list), color='r')
        plt.show()
        plt.close()

    # __________ GENERAL USE OF FUNCTION ___________
    # Fairly simple function, may just delete due to ease of use
    def confidence_interval(self, binomial_sim_function, range_1: float, range_2: float):
        np.percentile(binomial_sim_function, [range_1,range_2])
        
    # __________ GENERAL USE OF FUNCTION ___________
    # Takes in the same parameters as the binomial simulation functionns but returns the p-value of option 1 and 2 that are LESS THAN
    # what the average of the list was
    # This function works best as an alternate hypothesis tool
    def one_sided_p_value_lessthan(self, option_1: str, option_2, size: int, outcome_1: float, outcome_2: float, loop_length: int):
        option_1_list = []
        option_2_list = []
        for i in range(loop_length):
            binom_sim = np.random.choice([option_1, option_2], size = size, p= [outcome_1, outcome_2])
            option_1_list.append(np.sum(binom_sim == option_1))
            option_2_list.append(np.sum(binom_sim == option_2))
            
        option_1_list = np.array(option_1_list)
        option_2_list = np.array(option_2_list)
        
        p_value_option_1 = np.sum(option_1_list <= np.average(option_1_list)) / len(option_1_list)
        p_value_option_2 = np.sum(option_2_list <= np.average(option_2_list)) / len(option_2_list)
        
        return p_value_option_1, p_value_option_2
    
    # __________ GENERAL USE OF FUNCTION ___________
    # Returns the p-value of option 1 and 2 that are GREATER THAN the average of the list
    # This function works best as an alternate hypothesis tool
    def one_sided_p_value_greaterthan(self, option_1: str, option_2, size: int, outcome_1: float, outcome_2: float, loop_length: int):
        option_1_list = []
        option_2_list = []
        for i in range(loop_length):
            binom_sim = np.random.choice([option_1, option_2], size = size, p= [outcome_1, outcome_2])
            option_1_list.append(np.sum(binom_sim == option_1))
            option_2_list.append(np.sum(binom_sim == option_2))
            
        option_1_list = np.array(option_1_list)
        option_2_list = np.array(option_2_list)
        
        p_value_option_1 = np.sum(option_1_list >= np.average(option_1_list)) / len(option_1_list)
        p_value_option_2 = np.sum(option_2_list >= np.average(option_2_list)) / len(option_2_list)
        
        return p_value_option_1, p_value_option_2
    
    # __________ GENERAL USE OF FUNCTION ___________
    # Default test for many functions in Python
    # 
    def two_sided_p_value(self, option_1: str, option_2, size: int, outcome_1: float, outcome_2: float, loop_length: int, beg: int, end: int):
        option_1_list = []
        option_2_list = []
        for i in range(loop_length):
            binom_sim = np.random.choice([option_1, option_2], size = size, p= [outcome_1, outcome_2])
            option_1_list.append(np.sum(binom_sim == option_1))
            option_2_list.append(np.sum(binom_sim == option_2))
            
        option_1_list = np.array(option_1_list)
        option_2_list = np.array(option_2_list)
        
        p_value_option_1_two_sided = np.sum((option_1_list <= beg) | (option_1_list >= end)) / len(option_1_list)
        p_value_option_2_two_sided = np.sum((option_2_list <= beg) | (option_2_list >= end)) / len(option_2_list)
        
        return p_value_option_1_two_sided, p_value_option_2_two_sided
    
    # __________ GENERAL USE OF FUNCTION ___________
    def my_binom(self, observed_amount: int, amount_tests: int, p: float):
        from scipy.stats import binom_test
        
        binom_alternatives= ['two-sided','greater','less']
        hypotheses_list = []
        for i in range(3):
            hypotheses_list.append(binom_test(observed_amount, n = amount_tests, p = p, alternative=binom_alternatives[i]))
        
        print("Greater OR less than {} p-value: {}".format(observed_amount, hypotheses_list[0]))
        print("greater than {} p-value: {}".format(observed_amount, hypotheses_list[1]))
        print("less than {} p-value: {}".format(observed_amount, hypotheses_list[2]))
    
    # __________ GENERAL USE OF FUNCTION ___________
    # Takes the p-value variable and runs it against a few tests, returning as not significant if its greater than 5%
    # and significant it is less than 5%
    # 
    # Being less than 5% could indicate that there is a significant difference in the numbers we are looking for, hinting that there
    # may be a required change for the parameter we tested 
    #
    # Test will run 10000 times by default, add parameter to edit execution count
    def sigthresh(self, option_1, option_2, sim_size,  p: float, sig_threshold: float, alternative: str):
        from scipy.stats import binom_test

        # Initialize num_errors
        false_positives = 0
        # Set significance threshold value
        sig_threshold = sig_threshold

        # Run binomial tests & record errors
        for i in range(10000):
            sim_sample = np.random.choice([option_1, option_2], size=sim_size, p=[p, 1-p])
            num_correct = np.sum(sim_sample == option_1)
            p_val = binom_test(num_correct, sim_size, p, alternative=alternative)
            if p_val < sig_threshold:
                false_positives += 1

        # Print proportion of type I errors 
        print("False positives: {}".format(false_positives/1000))
        print("The pvalue is: {}".format(p_val))



print("Debugging successful, Quick DS has no errors... at the moment.")