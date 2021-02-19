# Understanding and Predicting Blight Fine

![Blight](https://www.newstatesman.com/sites/default/files/styles/cropped_article_image/public/blogs_2014/07/lafayette.jpg?itok=0My_zKs4)

## Overview

Blight violations are issued by the city to individuals who allow their properties to remain in a deteriorated condition. 
Blight has become a serious problem in Detroit. Every year the city issues millions of dollars in fines to violators 
and every year many of those tickets go unpaid. Following up and enforcing these fines can become extremely costly as well
and as such they want to use predictive analytics to increase ticket compliance.


## Buisness Problem
For this project we wanted to better predict when someone in Detroit would not
only allow their property to fall to blight but also not be compliant with the 
fines issued afterwards.

We based this project on a data challange from the Michigan Data Science Team (MDST) 
and the MIchigan Student Symposium for Interdisciplinary Statistical Sciences (MSSISS). 
They partnered with the City Detroit to better understand this problem.

To answer this question we first needed to understand when and why a resident would not 
comply with a blight violation. This is the task, understanding the factors that play 
into that, be they categorical or numerical.

## Approach

General Approach for this problem was based on Cross Industry Standard Process for Data Mining (CRISP-DM)
Which includes all following pmpotrtant steps: 

1. Look at the big picture. 
2. Get the data. 
3. Discover and visualize the data to gain insights. 
4. Prepare the data for Machine Learning algorithms. 
5. Select a model and train it. 
6. Fine-tune your model. 
7. Present your solution. 
8. Launch, monitor, and maintain your system.

![Blight](images/479px-CRISP-DM_Process_Diagram.png)

## Methodology

1. Data cleaning and preparation. Narrow down datasets to include important features.
2. Merging data sets along usefull columns and synthesize a more useable dataset with features.
3. Visualization variables based on different categories.
4. Identify Apropriate Evaluation metrics for model.
5. Generate classification models and evaluate results.
8. Generalize final model. Perform tunning. 

## Analysis

For this project we started with very large data sets, so the first step was stripping down what we thought would be usefull 
and combining it all into one dataset. This is primarilly what the ingeniring notebook was used for. We also used this notebook 
to do some feature engineering with the crime_count. Target variable was define as compliance and non-compliance forthe blight ticket.
Following visualisations help us understand more data.

![](images/Condition_compliance.png)

As we can see based on some criterias from conditions of the building we can make  assumptions about target variable. In this case building with sugested demolition most likely will get to non-compliance class. Same we can say about building with condition - "poor"

![](images/Fire_Compliance.png)

Chart above proof that all buildings that had fire at site associated with non-complience tickets.

## Modeling

Using Scikit-learn and IMBlearn packages 6 classification models were created :
- Logerithmic Regression. 
- Logerithmic Regression with SMOTE.
- Decession Tree.
- Decession Tree with SMOTE.
- Random Forest.
- Decission Tree with GridSearchCV.
- AdaBoost and Gradient Boosting with Weak Learners.

## Results

    
    After this was done we moved on to trying out different models with 
    the synthesized data. In. order to get a minimum for how much money 
    we can expect for the budgest we decided to go with precision. This 
    would allow us to be certain and correctly allocate revenue with flase 
    negatives giving us an upper bound on the budget. 
    
    With this in mind we decided to start with a baseline model using 
    Logistic Regression. With this model we were able to get a precision 
    score of .61. This isn't a terrible score but because of the class imbalance 
    we got a low F1 score (.03). As a result we decided to try and fix this 
    imbalance using SMOTE, this did help with the F! score  (.33), but drastically 
    reduced our precision rate (.23). 
    
    With this in mind we decided to move forward with tuning for precision, as we 
    wanted to be certain that the stated amount would come through and an upper bound on 
    the budget would only be a possitive thing. 
    
    Next we decided to try a Decission tree model, this resulted in a much better 
    precission rate (.61), but the problem of a low F! score (.31) presisted. We 
    wanted to better tune our model for precision so we decided to use smote again, 
    but the same problem aross of lowering the precession score to raise the F!.



It was determined that the Random Forest model perfromed the best and was utilized for the final implementation. 

## Conclusion
The synthesized data was analyzed and modeled. Some of the significant factors in determining if someone was going to pay on time were Judgement amount, crime count, disposition, and the condition of the lot. Applying the Scikit-learn Models we were able to get a precision score of 91%. The feature that had the most impact was judgement amount


## Future Work
-Tune the Model to better predict on both Recall and Precision
-Better synthesize the data to account for Nan and null values
-Apply a multivariable classification to include people who didn't pay on time but still paid.

## Repository Structure

    ├── README.md                    # The top-level README for reviewers of this project",
    ├── data                         # Synergized Data obtained from University of Michigan and Detroit Open Data Portal",
    ├── modules                      # py files with functions for ingeniring and modeling",
    ├── images                       # Both sourced externally and generated from Code",       
    ├── modeling.ipynb               # Notebook that gpes pver out modling process",                                        
    └── features_ingeniring.ipynb    # Notebook Used for feature engineering before Modeling",
    
    
**Authors** <br>
[Ivan Vanko](https://github.com/vanitoz)<br>
[Kelvin Arellano](https://github.com/Kelvin-Arellano)<br>
