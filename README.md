\# Project Title: Detection of Fake Reviews Using Machine Learning



This project mainly focusses on detecting fake reviews using various Machine Learning models. This project involves finding which review is Original (OR or genuine) and Computer Generated (CG or Fake). This is done by extracting dataset from open source and training 80% of the dataset into 6 different Machine Learning models namely SVM, Random Forest, Multinomial Naive Bayes, Logistic Regression, Gradient Boosting and XGBoost and then testing those trained models with the remaining 20% of the data. During this process, the accuracy and F1 score of all models are calculated and analyzed to find the best model which differentiates the fake reviews from the genuine ones. 



\## Features:

\##### Dataset Preprocessing

* Verify Null Values In DataSet - checks if there are null values in dataset and discards them
* Convert to Lower Case - converts all the reviews to lowercase letters
* Remove Punctuations - removes all the punctuation marks from the reviews
* Tokenization - converts the dataset reviews into individual tokens, which will be helpful to categorize them
* Remove Link Words - removes all the link words making the reviews clean
* Lemmatization - lemmatizes the reviews to make it to the root form of the word
* Convert Label Field - convert label field to 0 and 1 for 0 as CG and 1 as OR to make model learn properly
* Vectorization - transforms the text into matrix form



\## Built With 

* Language: Python
* Compiler: Python Compiler
* IDE used: Visual Studio Code



\## Getting Started

Follow the below steps to set up and run this project locally on your machine.



\### Pre-requisites

* Please make sure you have a Python compiler installed in your machine.
* Also install python libraries using the following command

  * pip install pandas numpy nltk scikit-learn matplotlib xgboost scipy
* Other libraries get installed when you run the code using setupp\_nltk method



\### Dataset

* Download the Kaggle dataset from 'https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset/data'
* Make sure you store the dataset.csv file in the same folder where the python file resides. 



\### Execution

Execute the script by using 'python filename.py'



\### Menu 

Menu includes many options 

* Options 1 to 6 - to just train and test any model of your choice and displays metrics like precision, recall and F1 score and a matplotlib window showing the confusion matrix
* Option 7 - to loop through all 6 models and showing the metrics and confusion matrix foreach of the models
* Option 8 - to loop through all 6 models and comparing and sharing result comparison of all 6 models in a table format, easier to understand
* Option 9 - to exit



\## License

Distributed under the MIT License. See LICENSE for more information.



