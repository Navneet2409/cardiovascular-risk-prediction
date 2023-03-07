# cardiovascular-risk-prediction
The goal of the project is to develop a tool for the early detection and prevention of CHD, addressing a significant public health concern using machine learning techniques.

## Table of Content
  * [Problem Statement](#problem-statement)
  * [Dataset](#dataset)
  * [Data Pipeline](#data-pipeline)
  * [Project Structure](#project-structure)
  * [Conclusion](#conclusion)
  
  
## Problem Statement
  A group of conditions affecting the heart and blood vessels is known as cardiovascular diseases. They consist of heart disease, which affects the blood vessels that 
  supply the heart muscle.The issue of coronary heart disease is a significant public health concern and early prediction of CHD risk is crucial for preventative 
  measures. We have to build a classification model to predict the 10-year risk of future coronary heart disease (CHD) for patients.
  
 
## Dataset
  The dataset is from an ongoing cardiovascular study on residents of Flamingham, Massachusetts. The data set includes over 3300 records and 16 attributes, each of 
  which is a potential risk factor, including demographic, behavioral, and medical risk factors. For more information on the dataset, please visit the Kaggle website at https://www.kaggle.com/datasets/christofel04/cardiovascular-study-dataset-predict-heart-disea
  
  
## Data Pipeline
  1. Analyze Data: 
      In this initial step, we attempted to comprehend the data and searched for various available features. We looked for things like the shape of the data, the 
      data types of each feature, a statistical summary, etc. at this stage.
  2. EDA: 
      EDA stands for Exploratory Data Analysis. It is a process of analyzing and understanding the data. The goal of EDA is to gain insights into the data, identify 
      patterns, and discover relationships and trends. It helps to identify outliers, missing values, and any other issues that may affect the analysis and modeling 
      of the data.
  3. Data Cleaning: 
      Data cleaning is the process of identifying and correcting or removing inaccuracies, inconsistencies, and missing values in a dataset. We inspected the dataset 
      for duplicate values. The null value and outlier detection and treatment followed. For the imputation of the null value we used the Mean, Median, and Mode 
      techniques, and for the outliers, we used the Clipping method to handle the outliers without any loss to the data.
  4. Feature Selection: 
      At this step, we did the encoding of categorical features. We used the correlation coefficient, chi-square test, information gain, and an extra tree classifier       to select the most relevant features. SMOTE is used to address the class imbalance in the target variable.
  5. Model Training and Implementation:  
      We scaled the features to bring down all of the values to a similar range. We pass the features to 8 different classification models. We also did 
      hyperparameter tuning using RandomSearchCV and GridSearchCV.
  6. Performance Evaluation: 
      After passing it to various classification models and calculating the metrics, we choose a final model that can make better predictions. We evaluated different 
      performance metrics but choose our final model using the f1 score and recall score.
      
      
## Project Structure
```
├── README.md
├── Dataset 
│   ├── [data_cardiovascular_risk.csv](https://github.com/Navneet2409/cardiovascular-risk-prediction/files/10660302/data_cardiovascular_risk.csv)
├── Problem Statement
│
├── Know Your Data
│
├── Understanding Your Variables
│
├── EDA
│   ├── Numeric & Categorical features
│   ├── Univariate Analysis
│   ├── Bivariate and Multivariate Analysis
│
├── Data Cleaning
│   ├── Duplicated values
│   ├── Missing values
│   ├── Skewness
│   ├── Treating Outliers
│
├── Feature Engineering
│   ├── Encoding
|   ├── Feature Selection
|   ├── Extra Trees Classifier
│   ├── Chi-square Test
|   ├── Information Gain
|   ├── Handling Class Imbalance
│
├── Model Building
│   ├── Train Test Split
|   ├── Scaling Data
|   ├── Model Training
│
├── Model Implementation
│   ├── Logistic Regression
|   ├── SVM
|   ├── KNN
│   ├── Decision Tree
|   ├── Random Forest
|   ├── AdaBoost
│   ├── XGBoost
|   ├── LightGBM
|
│   
├── Report
├── Presentation
├── Result
└── Reference
```


## Conclusion
In this project, we tackled a classification problem in which we had to classify and predict the 10-year risk of future coronary heart disease (CHD) for patients. The goal of the project was to develop a tool for the early detection and prevention of CHD, addressing a significant public health concern using machine learning techniques.

    - There were approximately 3390 records and 16 attributes in the dataset.
    - We started by importing the dataset, and necessary libraries and conducted exploratory data analysis (EDA) to get a clear insight into each feature by 
    separating the dataset into numeric and categoric features. We did Univariate, Bivariate, and even multivariate analyses.
    - After that, the outliers and null values were removed from the raw data and treated. Data were transformed to ensure that it was compatible with machine 
    learning models.
    - In feature engineering we transformed raw data into a more useful and informative form, by creating new features, encoding, and understanding important 
    features. We handled target class imbalance using SMOTE.
    - Then finally cleaned and scaled data was sent to various models, the metrics were made to evaluate the model, and we tuned the hyperparameters to make sure the 
    right parameters were being passed to the model. To select the final model based on requirements, we checked model_result.
    - When developing a machine learning model, it is generally recommended to track multiple metrics because each one highlights distinct aspects of model 
    performance. We are, however, focusing more on the Recall score and F1 score because we are dealing with healthcare data and our data is unbalanced.
    - With an f1-score of 0.907 and a recall score of 0.863 on test data, we have noticed that LightGBM Classifier outperforms all other models. It is safe to say 
    that the LightGBM Classifier is the best option for our issue if the f1-score is to be considered.
    - Our highest recall score, 0.938%, came from KNN.
    - The XGBoost and RandomForestClassifier tree-based algorithms also provided the best approach to achieving our goal. We were successful in achieving a 
    respective f1-score of 0.904 and 0.893.
    - The recall score is of the utmost significance in the medical field, where we place a greater emphasis on reducing false negative values because we do not want 
    to mispredict a person's safety when he is at risk. With recall scores of 0.938, 0.870, and 0.863, respectively, KNN, XGB, and LGBM performed the best.

    - Last but not least, we can select the Final model as our KNN classifier due to its highest recall score. It is acceptable to classify a healthy individual as 
    having a 10-year risk of coronary heart disease CHD (false positive) and to follow up with additional medical tests; however, it is categorically unacceptable to 
    miss identifying a particular patient or to classify a particular patient as healthy (false negative).
    
    
