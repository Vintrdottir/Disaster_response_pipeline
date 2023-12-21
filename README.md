# Disaster Response Message Categorization

This project aims to develop a **Machine Learning Pipeline** to categorize messages sent during disaster events, enabling their efficient forwarding to the appropriate disaster relief agency. Additionally, a user-friendly web app has been created for data visualization, where emergency workers can input new messages and receive classification results across various categories.

## Dataset
The dataset used in this project, provided by Figure Eight, comprises real messages sent during disasters. The goal is to build a robust model that can accurately categorize these messages to streamline the response efforts.

## Project Components

### ETL Pipeline: `process_data.py`
The **ETL (Extract, Transform, Load) pipeline** is responsible for data preprocessing. It performs the following tasks:

1. Loads the messages and categories datasets.
2. Merges the two datasets based on common identifiers.
3. Cleans the data by handling duplicates and addressing missing or inconsistent information.
4. Stores the processed data in an SQLite database.

#### Arguments
- `messages_filepath`: Path to the CSV file containing messages (e.g., `disaster_messages.csv`).
- `categories_filepath`: Path to the CSV file containing categories (e.g., `disaster_categories.csv`).
- `database_filename`: Path to the destination SQLite database (e.g., `disaster_response_db.db`).

### ML Pipeline: `train_classifier.py`
The **Machine Learning (ML) pipeline** focuses on training a multi-label classifier for text messages. The key steps are as follows:

1. Loads data from the SQLite database generated by the ETL pipeline.
2. Splits the dataset into training and test sets for model evaluation.
3. Builds a comprehensive text processing and machine learning pipeline.
4. Utilizes `GridSearchCV` for model tuning, optimizing hyperparameters.
5. Outputs evaluation results on the test set for model assessment.
6. Exports the final trained model as a pickle file.

#### Arguments
- `database_filepath`: Path to the SQLite destination database (e.g., `disaster_response_db.db`).
- `model_filepath`: Path to the pickle file where the ML model is saved (e.g., `classifier.pkl`).

### Flask Web App
The Flask web app provides a user interface for interacting with the trained model. Emergency workers can input new messages into the app, which then classifies the messages across relevant categories.

![image](https://github.com/Vintrdottir/Disaster_response_pipeline/assets/60987792/2d3f5450-3efa-49fb-85f1-c16d653f624f)

![image](https://github.com/Vintrdottir/Disaster_response_pipeline/assets/60987792/ee2152e7-ca94-4968-a96b-374c877c6160)



This comprehensive system, comprising ETL and ML pipelines along with a user-friendly web app, contributes to the effective categorization of disaster response messages, facilitating quicker and more targeted relief efforts.

## License
This project was prepared as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
