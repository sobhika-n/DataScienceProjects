# Drug-SideEffects-Classification

## Project Overview
The "Drug Side Effects Classification" project is aimed at leveraging machine learning techniques to predict and classify drug side effects based on patient demographics and other relevant features. The project addresses the importance of this classification in healthcare by providing tailored treatments, assessing risks, and aiding drug development. It also highlights the significance of helping patients make informed decisions, improving well-being, and reducing healthcare costs. Ultimately, this project aims to enhance patient care and safety through data-driven predictions of drug side effects.

## Acknowledgments
I would like to express my sincere gratitude to TCSiON for providing me with invaluable internship opportunities. Throughout my internship, I had the privilege of working on various aspects of the project, which included daily activity reporting and the creation of initial, mid, and final project reports and presentations. These experiences significantly contributed to my professional growth and skill development. The knowledge and insights gained during this internship have been instrumental in the successful execution of the "Drug Side Effects Classification" project. I am truly appreciative of the learning opportunities and the chance to apply my skills to real-world projects.hank you, TCSiON, for your continuous support and for being an essential part of my journey toward becoming a more proficient data scientist.

## Python Packages Used
This project utilizes the following Python packages:
Data Manipulation: pandas, numpy, and other relevant packages for efficient importing and handling of data. 
Data Visualization: Utilized packages like seaborn, matplotlib, and plotly for creating informative graphs and visualizations during the analysis process and for understanding the ML models. 
Machine Learning: Implemented packages such as scikit-learn, lightgbm, xgboost to develop and train machine learning models.

## Project Files
- Webmd_Drug_EDA.ipynb : This notebook contains the exploratory data analysis (EDA) and preprocessing steps.
- Webmd_Drug_Classification-Modelling.ipynb : This notebook focuses on building machine learning models and evaluating their performance for drug side effects classification.
  
## Dataset Information
The dataset comprises the following columns:
- Age: Age group of the patient.
- Condition: The medical condition for which the drug is prescribed.
- Date: The date when the review or drug usage occurred.
- Drug: The name of the drug.
- DrugId: A unique identifier for each drug.
- EaseofUse: An effectiveness rating for the ease of use of the drug.
- Effectiveness: An effectiveness rating for the drug.
- Reviews: Patient reviews or comments regarding the drug.
- Satisfaction: A satisfaction rating for the drug.
- Sex: Gender of the patient.
- Sides: Side effects associated with the drug.
- UsefulCount: A count of how useful the review was.
- Name: The name of the reviewer.
- Race: Race or ethnicity of the patient.

## Exploratory Data Analysis (EDA)
In this phase, we delve into the dataset to gain insights and prepare it for further analysis. The EDA process involves the following steps:

### Data Exploration
- Utilizing histograms and bar plots to visualize data distributions.
- Employing box plots to identify and handle outliers.

### Preprocessing
- Cleaning and handling missing values to ensure data quality.

### Visualization

#### Univariate Analysis
- Employing various visualization techniques, including histograms for age distribution, bar plots for top conditions, drugs, and ratings, pie charts for gender and race distribution, violin plots for satisfaction, effectiveness, and ease of use, and count plots for various columns.

#### Multivariate Analysis
- Utilizing bar plots for bivariate analysis to visually explore and compare factors like satisfaction, effectiveness, and ease of use across different categories such as gender, age, and race. This analysis focuses on the top 10 drugs in the dataset.
- Utilizing line plots for Time Series Analysis to study the relationship between conditions and their effectiveness ratings over time, with a focus on the top 10 conditions.

### Feature Engineering
- Tokenization and keyword analysis are applied in this section to create a new column named 'SideEffects' which indicates Severity of side effects experienced.

## Modeling
The second phase of the project involves building machine learning models to predict drug side effects based on patient demographics and drug-related factors. The main steps include:

### Data Preparation
- Preparing the data for modeling.

### Classification Models
- Evaluating several classification models, including Random Forest Classifier, Logistic Regression, LightGBM Classifier, and XGBoost Classifier.

### Model Evaluation
- Assessing model performance using metrics like accuracy, precision, recall, and F1-score.
- Generating confusion matrices and classification reports for detailed model evaluation.

Below are the evaluation metrics, including accuracy, precision, recall, and F1-score, for each of the models used in the project:
### Random Forest Classifier
|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Mild      | 0.98      | 0.99   | 0.98     | 49,110  |
| Moderate  | 0.88      | 0.84   | 0.86     | 4,397   |
| Nil       | 0.82      | 0.76   | 0.79     | 2,516   |

### Logistic Regression
|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Mild      | 0.88      | 1.00   | 0.93     | 49,110  |
| Moderate  | 0.00      | 0.00   | 0.00     | 4,397   |
| Nil       | 0.00      | 0.00   | 0.00     | 2,516   |

### LightGBM Classifier
|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Mild      | 0.93      | 0.99   | 0.96     | 49,110  |
| Moderate  | 0.88      | 0.50   | 0.64     | 4,397   |
| Nil       | 0.82      | 0.33   | 0.47     | 2,516   |

### XGBoost Classifier
|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Mild      | 0.98      | 1.00   | 0.99     | 49,110  |
| Moderate  | 0.99      | 0.88   | 0.93     | 4,397   |
| Nil       | 0.99      | 0.83   | 0.90     | 2,516   |


### Model Selection
- Selecting the best-performing model based on the evaluation results. In this project, the XGBoost Classifier proved to be the preferred choice.

### Cross-Validation
- Applying cross-validation to check for potential overfitting and ensure model robustness.

## Conclusion
In conclusion, this project offers valuable insights into drug side effects classification. The EDA phase provides a comprehensive understanding of the dataset, while the Modeling phase showcases the effectiveness of machine learning in predicting drug side effects based on patient demographics. The XGBoost model, in particular, demonstrated strong performance and reliability, making it a valuable tool for healthcare and pharmaceutical applications.
