import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import distance
import re

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix, recall_score, precision_score, classification_report

from ingeniring_utils import * # import all existing functions from file cleaning.py
from modeling_utils import *



pd.set_option('display.max_columns', 360)

def categor_var_analysis(df, features):
    """
    Features - list of categorical features
    df - data frame where features located
    """
    print('Discrete Variables')
    
    for feature in features:
        print(df.groupby('{}'.format(feature))['{}'.format(feature)].count())
        print('--'*40)
        

def numeric_var_analysis(df, features):
    """
    Features - list of numerical features
    df - data frame where features located
    """
    print('Continuous Variables')
    print(df[features].describe().transpose())
    
    

def to_dummies(df, features):
    """
    Generating dummy variables for features
    parama: df
            features - list of features
    """
    d_df = []
    for f in features:
        d_df.append(pd.get_dummies(df[f], prefix='{}'.format(str(f)[:]), drop_first=True))
    #import pdb;pdb.set_trace()
    df = df.drop(features, axis = 1)
    df = pd.concat([df] + d_df ,axis=1)
    
    return df


def log_reg (X , y, roc = False, matrix = True, report =True):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.25)
    # Initial Model
    logreg = LogisticRegression(fit_intercept=False, solver='liblinear')
    logreg.fit(X_train, y_train)

    # Probability scores for test set
    y_score = logreg.fit(X_train, y_train).decision_function(X_test)
    # False positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    # Test set predictions
    pred = logreg.predict(X_test)
    
    # Seaborn's beautiful styling
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

    # Print AUC
    print('AUC: {}'.format(auc(fpr, tpr)))

    # Plot the ROC curve
    if roc :
        plt.figure(figsize=(10, 8))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.yticks([i/20.0 for i in range(21)])
        plt.xticks([i/20.0 for i in range(21)])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
        
    if matrix:
        print ('\n\n Cofusion Matrix \n\n')
        # Plot confusion matrix of the test set
        plot_confusion_matrix(logreg, X_test, y_test,
                              display_labels=["Non-Compliant", "Compliant"],
                              values_format=".5g")
        plt.grid(False) # removes the annoying grid lines from plot
        plt.show()
    
    if report:
        
        print('\nGeneral Report \n')
        print(classification_report(y_test,pred))
    



def decision_tree(X, y, max_depth, future_importance = True, confusion_mtrix = True, report =True):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)
    
    tree_clf = DecisionTreeClassifier(max_depth = 3)
    tree_clf.fit(X_train,y_train)
    
    
    tree_clf.feature_importances_
    if future_importance:
        print('Feature Importance: \n')
        n_features = X_train.shape[1]
        plt.figure(figsize=(8,8))
        plt.barh(range(n_features), tree_clf.feature_importances_, align='center') 
        plt.yticks(np.arange(n_features), X_train.columns.values) 
        plt.xlabel('Feature importance')
        plt.ylabel('Feature')
        plt.show()
        
    # Test set predictions
    pred = tree_clf.predict(X_test)
    
    if confusion_mtrix:
        print('Confusion Matrix: \n')
        # Confusion matrix and classification report
        print(confusion_matrix(y_test,pred))
        
    if report:
        print('\nGeneral Report \n')
        print(classification_report(y_test,pred))

    return print("Testing Accuracy for Decision Tree Classifier: {:.4}%".format(accuracy_score(y_test, pred) * 100))



    
    

def decision_tree_smote(X, y, max_depth, future_importance = True, confusion_mtrix = True, report =True):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)
    
    X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train) 

    tree_clf = DecisionTreeClassifier(max_depth = 3)
    tree_clf.fit(X_train_resampled,y_train_resampled)
    
    
    tree_clf.feature_importances_
    if future_importance:
        print('Feature Importance: \n')
        n_features = X_train.shape[1]
        plt.figure(figsize=(8,8))
        plt.barh(range(n_features), tree_clf.feature_importances_, align='center') 
        plt.yticks(np.arange(n_features), X_train.columns.values) 
        plt.xlabel('Feature importance')
        plt.ylabel('Feature')
        plt.show()
        
        
    # Test set predictions
    pred = tree_clf.predict(X_test)
    
    if confusion_mtrix:
        print('Confusion Matrix: \n')
        # Confusion matrix and classification report
        print(confusion_matrix(y_test,pred))
        
    if report:
        print('\nGeneral Report \n')
        print(classification_report(y_test,pred))

    return print("Testing Accuracy for Decision Tree Classifier: {:.4}%".format(accuracy_score(y_test, pred) * 100))