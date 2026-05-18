# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:42:33 2026

@author: rmlak
"""

import pandas as pd
import numpy as np
import nltk
import re
import os
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from scipy.sparse import hstack
from xgboost import XGBClassifier

def setup_nltk():
    # List of all resources needed for your Amazon Fake Review project
    required = ['wordnet', 'omw-1.4']
    
    for resource in required:
        try:
            # Check if it's a tokenizer or a corpus
            if resource in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{resource}')
            else:
                nltk.data.find(f'corpora/{resource}')
        except LookupError:
            print(f"Downloading missing resource: {resource}")
            nltk.download(resource)
            
def readDataset(fileName):
    # Get the directory where the script is saved
    script_dir = os.path.dirname(__file__) 

    # Combine that directory with the file name
    file_path = os.path.join(script_dir, fileName)

    # Load the dataset
    dataset = pd.read_csv(file_path)
    
    print("The first 5 lines of the datatset are: ")
    print(dataset.head())
    
    print("The description of the dataset is: ")
    print(dataset.describe())
    
    print("The columns in the dataset are: ")
    print(dataset.columns)
    return dataset

def verifyNullValuesInDataSet(dataset):
    #finding if data is null 
    print("Is data null? ", dataset.isnull().values.any())

    #if data is null, replacing the None values to Nan
    if(dataset.isnull().values.any()==True):
        countOfNullData = dataset.isnull().sum() #finds how many empty cells in each column
        null_data = dataset[dataset.isnull().any(axis=1)] # finds only the rows that contain at least one null value
        print(countOfNullData, null_data)
        
        dataset.replace(r'^\s*$', np.nan, regex=True, inplace=True) # replaces empty strings or whitespace with Nan
        print("After replacement of whitespaces and empty strings, is data null? ",dataset.isnull().values.any())
    return dataset

def convertToLowerCase(dataset):
    print("Dataset after converting to lowercase: ")
    dataset = dataset.map(lambda s: s.lower() if isinstance(s, str) else s)
    print(dataset)
    return dataset

def removePunctuations(dataset):
    print("Dataset after removing all punctuations: ")
    dataset['text_'] = dataset['text_'].str.replace(r'[^\w\s]', ' ', regex=True)
    print(dataset)
    return dataset
    
def tokenize(dataset):
    print("Dataset after tokenization: ")
    dataset['text_'] = dataset['text_'].fillna(' ').astype(str)
    dataset['text_'] = dataset['text_'].apply(word_tokenize)
    print("The first 5 values of the text field in dataset after tokenization: ")
    print(dataset['text_'].head())
    return dataset

def removeLinkWords(text):
    # 1. If the data is a LIST, join it into a string
    if isinstance(text, list):
        text = " ".join(text)
        
    # Handle potential None/NaN values immediately
    if not isinstance(text, str):
        return ""
    
    text = text.lower()

    # Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove HTML tags (<br>, <div>, etc.)
    text = re.sub(r'<.*?>', '', text)

    # Remove HTML entities (e.g., &quot;, &amp;)
    text = re.sub(r'&\w+;', '', text)

    # Remove Platform-Specific Nouns (the "True Noise")
    # We use \b to ensure we only match whole words (so 'amazon' doesn't break 'amazing')
    noise_words = ['amazon', 'product', 'item', 'shipping', 'seller', 'package', 'delivered']
    noise_pattern = r'\b(' + '|'.join(noise_words) + r')\b'
    text = re.sub(noise_pattern, '', text)

    # Clean Formatting Artifacts (newlines, tabs, excessive spaces)
    # This turns "hello \n\n   world" into "hello world"
    text = " ".join(text.split())
    
    return text

def lemmatizeTokens(tokens):
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    if not isinstance(tokens, str):
        return []
    
    # 2. Break string into a list of words (Tokens)
    tokens = word_tokenize(tokens)
    
    # 3. Lemmatize each word in that list
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    
    return lemmas

def countDuplicateReviews(dataset):
    print("Duplicate reviews found are :")
    # Count how many times each review text appears
    duplicate_counts = dataset['text_'].value_counts()
    print(duplicate_counts[duplicate_counts > 1])
    
def countReviewLength(dataset):
    # Calculate word count (number of tokens in the list)
    dataset['review_length'] = dataset['lemmatized'].apply(lambda x: len(x))
    
    # Display to verify
    print("After counting the length of the reviews")
    print(dataset[['processed_text', 'review_length']].head())

    return dataset

def convertLabelField(dataset):
    # Assuming your target column is named 'label_column'
    # Map 'Fake' to 0 and 'Genuine' to 1
    dataset['label_encoded'] = dataset['label'].map({'cg': 0, 'or': 1})
    
    # Safety check: ensure no NaN values in labels
    dataset = dataset.dropna(subset=['label_encoded'])
    
    return dataset

def vectorization(dataset):
    
    # Initialize TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    # Transform the text into a sparse matrix
    X_tfidf = tfidf.fit_transform(dataset['final_text'])
    
    # Prepare the 'Review Length' feature as a column matrix
    # We need to reshape it to (n_samples, 1) to combine it
    X_length = dataset['review_length'].values.reshape(-1, 1)
    
    # Combine TF-IDF features and Review Length into one master matrix (X)
    X = hstack([X_tfidf, X_length])
    
    print(f"Final Feature Matrix Shape: {X.shape}")
    
    return X

def SVM_Model(X_train, X_test, y_train, y_test):
    # Linear SVM 
    svm_model = LinearSVC(random_state=42, C=1.0)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    
    print("--- SVM Results ---")
    print(classification_report(y_test, svm_preds))
    
    # 1. Generate the matrix
    cm = confusion_matrix(y_test, svm_preds)
    
    # 2. Plot it visually
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Genuine'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - SVM: Amazon Fake Review Detection")
    plt.show()
    
def RandomForest_Algorithm(X_train, X_test, y_train, y_test):
    # 2. Random Forest 
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    print("--- Random Forest Results ---")
    print(classification_report(y_test, rf_preds))
    
    # 1. Generate the matrix
    cm = confusion_matrix(y_test, rf_preds)
    
    # 2. Plot it visually
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Genuine'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - RF: Amazon Fake Review Detection")
    plt.show()
    
def MultinomialNaiveBayes_Model(X_train, X_test, y_train, y_test):
    # 2. Random Forest 
    mnb_model = MultinomialNB()
    mnb_model.fit(X_train, y_train)
    
    # Predict and Evaluate
    mnb_preds = mnb_model.predict(X_test)
    
    print("--- Multinomial Naive Bayes Results ---")
    print(classification_report(y_test, mnb_preds))
    
    # 1. Generate the matrix
    cm = confusion_matrix(y_test, mnb_preds)
    
    # 2. Plot it visually
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Genuine'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - MNB: Amazon Fake Review Detection")
    plt.show()
    
def LogisticRegression_Model(X_train, X_test, y_train, y_test):
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train, y_train)
    
    logreg_pred = log_model.predict(X_test)
    print("\n--- Logistic Regression Results---")
    print(classification_report(y_test, logreg_pred))
    
    # 1. Generate the matrix
    cm = confusion_matrix(y_test, logreg_pred)
    
    # 2. Plot it visually
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Genuine'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Log Reg: Amazon Fake Review Detection")
    plt.show()
    
def GradientBoosting_Model(X_train, X_test, y_train, y_test):
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)
    
    gb_pred = gb_model.predict(X_test)
    print("\n--- Gradient Boosting Results---")
    print(classification_report(y_test, gb_pred))
    
    # 1. Generate the matrix
    cm = confusion_matrix(y_test, gb_pred)
    
    # 2. Plot it visually
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Genuine'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Gradient Boost: Amazon Fake Review Detection")
    plt.show()
    
def XGBoost_Model(X_train, X_test, y_train, y_test):
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    xgb_pred = xgb_model.predict(X_test)
    print("\n--- Extreme Gradient Boosting (XGB) Results---")
    print(classification_report(y_test, xgb_pred))
    
    # 1. Generate the matrix
    cm = confusion_matrix(y_test, xgb_pred)
    
    # 2. Plot it visually
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Genuine'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - XGBoost: Amazon Fake Review Detection")
    plt.show()
    

def testingOnAlgorithms(X_train, X_test, y_train, y_test):
    SVM_Model(X_train, X_test, y_train, y_test)
    RandomForest_Algorithm(X_train, X_test, y_train, y_test)
    MultinomialNaiveBayes_Model(X_train, X_test, y_train, y_test)
    LogisticRegression_Model(X_train, X_test, y_train, y_test)
    GradientBoosting_Model(X_train, X_test, y_train, y_test)
    XGBoost_Model(X_train, X_test, y_train, y_test)
    
def comparisonOnAlgorithms(X_train, X_test, y_train, y_test):
    # Dictionary to store results
    results = []

    # List of models to iterate through
    models = [
        ("SVM", LinearSVC(random_state=42, C=1.0)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
        ("Naive Bayes", MultinomialNB()),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("XGBoost", XGBClassifier(n_estimators=100, max_depth=6, random_state=42))
    ]

    for name, model in models:
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Calculate Metrics
        report = classification_report(y_test, preds, output_dict=True)
        accuracy = report['accuracy']
        f1_fake = report['0']['f1-score'] # '0' is 'cg' (Fake)
        
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "F1-Score (Fake)": f1_fake
        })

    # Create a Comparison DataFrame
    comparison_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    
    print("FINAL MODEL COMPARISON")
    print(comparison_df.to_string(index=False))
    
    # Plot the comparison
    comparison_df.plot(x='Model', y=['Accuracy', 'F1-Score (Fake)'], kind='bar', figsize=(10, 5))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

setup_nltk()
print("NLTK environemnt is ready")

dataset = readDataset("fake reviews dataset.csv")

dataset = verifyNullValuesInDataSet(dataset)

dataset = convertToLowerCase(dataset)

dataset = removePunctuations(dataset)

dataset = tokenize(dataset)

dataset['processed_text']=dataset['text_'].apply(removeLinkWords)

# Apply to your tokenized column
# (Using the 'processed_text' column created in the previous step)
dataset['lemmatized'] = dataset['processed_text'].apply(lemmatizeTokens)

# Join the list of lemmas back into a single string
dataset['final_text'] = dataset['lemmatized'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")

# Check the result
print(dataset['final_text'].head())

# Counts the duplicate reviews found
countDuplicateReviews(dataset)

dataset = countReviewLength(dataset)

# Converting the label field in such a way that 'or' as 1 and 'cg' as 0
dataset = convertLabelField(dataset)

X_Data = vectorization(dataset)

# Extract the labels as a 1D array
Y_Data = dataset['label_encoded'].values

# Triple-check that X and y have the same number of rows
if X_Data.shape[0] == len(Y_Data):
    print(f"Ready to split! Both have {X_Data.shape[0]} rows.")
else:
    print(f"Error: X has {X_Data.shape[0]} rows but y has {len(Y_Data)} rows. Check for dropped NaNs!")

# Splitting X and Y for Training the model
X_train, X_test, y_train, y_test = train_test_split(X_Data, Y_Data, test_size=0.2, random_state=42, stratify=Y_Data)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Verify proportions of split
print(f"Original ratio: {np.bincount(Y_Data) / len(Y_Data)}")
print(f"Test set ratio: {np.bincount(y_test) / len(y_test)}")

choice=1
while (choice!=9):

    choice=0
    print("Choices: ")
    
    print("1. Test SVM Algorithm")
    print("2. Test Random Forest Model")
    print("3. Test Multinomial Naive Bayes Model")
    print("4. Test Logistic Regression Model")
    print("5. Test Gradient Boosting Model")
    print("6. Test XGBoost Model")
    print("7. Test all the models and show graph for all of them seperately")
    print("8. Make a comparison on all these algorithms")
    print("9. Exit")

    choice=int(input("Enter your choice: "))

    match choice:
        case 1:
            SVM_Model(X_train, X_test, y_train, y_test)
        case 2:
            RandomForest_Algorithm(X_train, X_test, y_train, y_test)
        case 3:
            MultinomialNaiveBayes_Model(X_train, X_test, y_train, y_test)
        case 4:
            LogisticRegression_Model(X_train, X_test, y_train, y_test)
        case 5:
            GradientBoosting_Model(X_train, X_test, y_train, y_test)
        case 6:
            XGBoost_Model(X_train, X_test, y_train, y_test)
        case 7:
            testingOnAlgorithms(X_train, X_test, y_train, y_test)
        case 8:
            comparisonOnAlgorithms(X_train, X_test, y_train, y_test)
        case 9:
            print("Exit")