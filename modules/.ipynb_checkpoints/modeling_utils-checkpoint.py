import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import distance
import re

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix, recall_score, precision_score, classification_report



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




def log_reg (X, y, solver='liblinear'):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.25)
    logreg = LogisticRegression(fit_intercept=False, solver= solver)
    logreg.fit(X_train, y_train)

    # predictions
    y_pred_test = logreg.predict(X_test)
    y_pred_train = logreg.predict(X_train)

    # Calculate Accuracy Score 
    print('Accuracy on Train Set:')
    print(metrics.accuracy_score(y_train, y_pred_train))

    print('\n Accuracy on Test Set: ')
    print(metrics.accuracy_score(y_test, y_pred_test))

    # IMPORTANT: first argument is true values, second argument is predicted values
    print("\n Confusion Matrix for Test Set")
    print(metrics.confusion_matrix(y_test, y_pred_test))
    
    print("\n General Repor \n")
    print(classification_report(y_test, y_pred_test))

    return logreg




def log_reg_smote (X, y, solver='liblinear'):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # Previous original class distribution
    print('Original class distribution')
    print(y.value_counts()) 

    # Fit SMOTE to training data
    X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train) 

    # Preview synthetic sample class distribution
    print('\n Synthetic sample class distribution' )
    print(pd.Series(y_resampled).value_counts()) 
    
    print(40*'-')
    
    logreg = LogisticRegression(fit_intercept=False, solver='liblinear')
    logreg.fit(X_resampled, y_resampled)

    # predictions
    y_pred_test = logreg.predict(X_test)
    y_pred_train = logreg.predict(X_train)

    # Calculate Accuracy Score 
    print('\n Accuracy on Train Set:')
    print(metrics.accuracy_score(y_train, y_pred_train))

    print('\n Accuracy on Test Set:')
    print(metrics.accuracy_score(y_test, y_pred_test))

    # IMPORTANT: first argument is true values, second argument is predicted values
    print("\n Confusion Matrix for Test Set\n")
    print(metrics.confusion_matrix(y_test, y_pred_test))
    
    print("\n General Repor: \n")
    print(classification_report(y_test, y_pred_test))

    return logreg
    
    


def decision_tree(X, y, max_depth = None, min_samples_leaf = 1, min_samples_split = 2):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    clf = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = 1, min_samples_split = 2)
    clf.fit(X_train, y_train)

    # predictions
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    # Calculate Accuracy Score 
    print('Accuracy on Train Set:')
    print(metrics.accuracy_score(y_train, y_pred_train))

    print('\n Accuracy on Test Set: ')
    print(metrics.accuracy_score(y_test, y_pred_test))

    print("\n F1 Score for Test Set:")
    print(metrics.f1_score(y_test, y_pred_test))

    # IMPORTANT: first argument is true values, second argument is predicted values
    print("\n Confusion Matrix for Test Set:")
    print(metrics.confusion_matrix(y_test, y_pred_test))

    print("\n General Repor: \n")
    print(classification_report(y_test, y_pred_test))
    
    return clf
    
    

def decision_tree_smote(X, y, max_depth = None, min_samples_leaf = 1, min_samples_split = 2):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train) 
    
    clf = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = 1, min_samples_split = 2)
    clf.fit(X_resampled, y_resampled)

    # predictions
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    # Calculate Accuracy Score 
    print('Accuracy on Train Set:')
    print(metrics.accuracy_score(y_train, y_pred_train))

    print('\n Accuracy on Test Set: ')
    print(metrics.accuracy_score(y_test, y_pred_test))

    print(4*'-')

    # IMPORTANT: first argument is true values, second argument is predicted values
    print("\n Confusion Matrix for Test Set")
    print(metrics.confusion_matrix(y_test, y_pred_test))
    
    print("\n General Repor \n")
    print(classification_report(y_test, y_pred_test))
    
    return clf    
    
    
def plot_feature_importances(model, X):
    """
    Plot Features Importance of The curren model
    """
    n_features = X.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), X.columns.values) 
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.show()
    
    
    
def randome_forest(X, y, max_depth = None):
    
    # Instantiate and fit a RandomForestClassifier, split data and fit model
    forest = RandomForestClassifier(max_depth= None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    forest.fit(X, y)
    
    # predictions and evaluations
    y_pred_test = forest.predict(X_test)
    y_pred_train = forest.predict(X_train)

    # Calculate Accuracy Score 
    print('Accuracy on Train Set:')
    print(metrics.accuracy_score(y_train, y_pred_train))

    print('\n Accuracy on Test Set: ')
    print(metrics.accuracy_score(y_test, y_pred_test))

    # IMPORTANT: first argument is true values, second argument is predicted values
    print("\n Confusion Matrix for Test Set")
    print(metrics.confusion_matrix(y_test, y_pred_test))

    print("\n General Repor :\n")
    print(classification_report(y_test, y_pred_test))
    
    return forest


def AdaBoost(X, y):
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    adaboost_clf = AdaBoostClassifier(random_state=42)
    adaboost_clf.fit(X_train, y_train)
    
    # AdaBoost model predictions
    adaboost_train_preds = adaboost_clf.predict(X_train)
    adaboost_test_preds = adaboost_clf.predict(X_test)
    
    adaboost_confusion_matrix = confusion_matrix(y_test, adaboost_test_preds)
    
    print('Accuracy on Train Set:')
    print(metrics.accuracy_score(y_train, adaboost_train_preds))
    
    print('\n Accuracy on Test Set: ')
    print(metrics.accuracy_score(y_test, adaboost_test_preds))
    
    print("\n Confusion Matrix for Test Set")
    print(adaboost_confusion_matrix)
    
    print("\n General Repor :\n")
    print(classification_report(y_test, adaboost_test_preds))
    
    return adaboost_clf

def gbt(X, y, max_depth = None, show_roc = False):
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    gbt_clf = GradientBoostingClassifier(random_state=42, max_depth = max_depth)
    gbt_clf.fit(X_train, y_train).fit(X_train, y_train)
    
    # AdaBoost model predictions
    gbt_clf_train_preds = gbt_clf.predict(X_train)
    gbt_clf_test_preds = gbt_clf.predict(X_test)
    
    gbt_confusion_matrix = confusion_matrix(y_test, gbt_clf_test_preds)
    
    print('Accuracy on Train Set:')
    print(metrics.accuracy_score(y_train, gbt_clf_train_preds))
    
    print('\n Accuracy on Test Set: ')
    print(metrics.accuracy_score(y_test, gbt_clf_test_preds))
    
    print("\n Confusion Matrix for Test Set")
    print(gbt_confusion_matrix)
    
    print("\n General Repor :\n")
    print(classification_report(y_test, gbt_clf_test_preds))
    
    if show_roc:
        
        y_score = gbt_clf.decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        print('AUC: {}'.format(auc(fpr, tpr)))
        plt.figure(figsize=(10, 8))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
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

    
    return gbt_clf
    
    
