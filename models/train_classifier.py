import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from termcolor import colored, cprint
import warnings
import re
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
warnings.simplefilter('ignore')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer



def load_data(database_filepath):
    """
    Load and merge datasets from an SQLite database.
    
    Inputs:
        database_filepath (str): the name of the database
    
    Outputs:
        X (Series): messages
        Y (DataFrame): remaining columns
        category_names (list): List of category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('MessagesCategories', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    category_names = list(np.array(Y.columns))
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the input text data for natural language processing.

    Inputs:
        text (str): text data to be tokenized

    Outputs:
        lemmatized (list): List of lemmatized and tokenized words after converting to lowercase,
                           removing non-alphanumeric characters, tokenizing, normalizing, and removing stop words
    """
    # Converting everything to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize words
    tokens = word_tokenize(text)

    # Lemmatization and remove stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return lemmatized
 

def build_model():
    """
    Build a machine learning pipeline with hyperparameter tuning using grid search.
    Inputs:
        None
    Outputs:
        cv (GridSearchCV): Grid search object with a pipeline containing tokenization,
                          TF-IDF transformation, and multi-output classification using
                          a Random Forest Classifier. The grid search explores
                          different parameter combinations for optimization.
    """
    # Define the machine learning pipeline
    pipe = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Define hyperparameter grid for grid search
    parameters = {
        'vect__min_df': [1, 5],
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [10, 25],
        'clf__estimator__min_samples_split': [2, 4]
    }

    # Create a grid search object
    cv = GridSearchCV(pipe, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of a machine learning model on the test set for each category.

    Inputs:
        model: The trained machine learning model
        X_test: The test set for input features
        Y_test: The true labels for the test set
        category_names: List of category names

    Output:
        None (Prints precision, recall, and F-score for each category)
    """
    Y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        precision, recall, fscore, support = precision_recall_fscore_support(Y_test[col],
                                                                            Y_pred[:, i],
                                                                            average='weighted')

        print('\nReport for the column ({}):\n'.format(colored(col, 'green', attrs=['bold', 'underline'])))

        if precision >= 0.75:
            print('Precision: {}'.format(colored(round(precision, 2), 'blue')))
        else:
            print('Precision: {}'.format(colored(round(precision, 2), 'yellow')))

        if recall >= 0.75:
            print('Recall: {}'.format(colored(round(recall, 2), 'blue')))
        else:
            print('Recall: {}'.format(colored(round(recall, 2), 'yellow')))

        if fscore >= 0.75:
            print('F-score: {}'.format(colored(round(fscore, 2), 'blue')))
        else:
            print('F-score: {}'.format(colored(round(fscore, 2), 'yellow')))



def save_model(model, model_filepath):
    """
    Save the trained machine learning model to a pickle file.

    Inputs:
        model: The trained machine learning model.
        model_filepath: The file path where the model will be saved.

    Output:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
