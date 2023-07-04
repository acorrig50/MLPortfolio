import pandas as pd
import numpy as np

# Data cleaning functions below

def data_custodian(df):
    
    # Cool visual
    print("Data Custodian is booting up...")
    for i in range(5):
        print("...")
        for j in range(5):
            print("...")
    
    
    # Print information about the data set
    print(df.info())
    print(df.shape)
    
    # Turn the columns into a list and then display them
    columns = df.columns.tolist()
    print(columns)
    
    # _____NOTE_____
    # This next section will perform actions that may or may not be necessary. For brevity, remove this
    # section if needed
    
    # Get rid of duplicate rows
    df = df.drop_duplicates()
    
    # Change the name of columns to lowercase
    df.columns = map(str.lower(), df.columns)
    
    # Renaming nested function
    def rename_columns():
        print("Here are the names of the columns: ")
        print(df.columns)
        answer = input("Would you like to rename any of them? 1 for yes, any other key for no.")
        if answer == 1:
            while exit_condition != 0:
                column_to_change = input("Enter the name of the column you would like to change: ")
                changed_name = input("Enter the new name for the column: ")
                df = df.rename({str(column_to_change) : str(changed_name)}, axis= 1)
                
                exit_condition = input("Do you want to exit the column name changer? 1 for yes, 0 for no.")
        else:
            print("You chose to not rename any columns, moving on...")
    
    # Call the rename function
    rename_columns()
    
    # Now we move on to handling missing data
    # prints out the total number of missing data in each column
    print(df.isna().sum())
    
    
    # Now we see if data needs to be melted
    # NOTE: STILL UNDER CONSTRUCTION
    def data_melt():
        print("Welcome to the melter")
        answer = input("Is there melting that needs to be done?: 1 for yes, 0 for no.")
        
        if answer == 1:
            while exit_condition != 0:
                column_to_melt = input("Enter the column you would like to melt: ")
                name_to_repace = input("Enter the replacement name: ")
                df = df.melt(
                    id_vars = [column_to_melt],
                    var_name = [name_to_repace],
                    value_name = []
                    
                )
                exit_condition = input("Would you like to continue melting?: 1 for yes, 0 for no")    
        else:
            pass
    
    
    


data_custodian()
    
    
    
    
    
    
    
    
    