import sys
import sqlite3
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine  

def load_data(messages_filepath, categories_filepath):
    """
    Load csv files with messages and categories, then merge two loaded datasets.
    
    inputs:
        messages_filepath (str): path of csv file containing messages dataset
        categories_filepath (str): path of csv file containing categories dataset
       
    outputs:
        df (DataFrame): contains merged datasets of messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge (categories, left_on = 'id', right_on = 'id', how = 'inner', validate = 'many_to_many')
    
    return df

def clean_data(df):
    """
    Clean the input dataframe by removing duplicate rows and converting categories from strings to binary values.

    inputs:
        df (DataFrame): The input dataframe containing merged content of messages and categories datasets.

    outputs:
        df (DataFrame): The cleaned dataframe with duplicate rows removed and categories converted to binary values.
    """
    # create a dataframe of 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row to extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.rstrip('- 0 1'))
    categories.columns = category_colnames
    
    # convert category values to numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column], errors='coerce')
        
    # replace current category columns in df with new ones
    df.drop(['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, sort=False)
    
    # remove rows with 'related' column equal to 2
    df = df[df['related'] != 2]
    
    # remove duplicates
    df = df.drop_duplicates(subset=['message'])
    
    return df

def save_data(df, database_filename):
    """
    Save the cleaned version of the merged message and categories data into an SQLite database.
    
    inputs:
        df (DataFrame): The dataframe containing the cleaned version of merged message and categories data.
        database_filename (str): The filename for the output database
    
    outputs:
        None
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessagesCategories', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
