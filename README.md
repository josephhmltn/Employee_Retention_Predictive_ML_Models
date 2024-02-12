# Predictive ML Models for Employee Retention

## Dataset
Dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction).

### Data Understanding
14,999 rows, 10 columns - each row is a different employee's self-reported information
- `satisfaction_level` = Self-reported satisfaction level [0-1]
- `last evaluation` = Score of employee's last performance review [0-1]
- `number_project` = Number of projects employee contributes to
- `average_monthly_hours` =  Average number of hours employee worked per month
- `time_spend_company` = How long the employee has been with the company (in years)
- `work_accident` =  Whether or not the employee experienced an accident while at work
- `left` =  Whether or not the employee left the company
- `promotion_last_5years` =  Whether or not the employee was promoted in the last 5 years
- `department` =  Employee's department
- `salary` =  Employee's salary (low, medium, or high)

## Project Overview
This project started as the capstone project for the [Google Advanced Data Analytics Professional Certificate](https://www.coursera.org/professional-certificates/google-advanced-data-analytics#courses) via Coursera and aims to predict employee turnover using various machine learning models. By analyzing HR data, the project identifies key factors leading to employee departures and provides actionable insights to reduce turnover rates.

## Data Loading and Preprocessing
Data preprocessing involved cleaning, encoding categorical variables, and feature scaling to prepare the dataset for model training.

```python
# Load the dataset to get an overview
filename = "HR_data.csv"
filepath = os.path.join(os.getcwd(), filename)
df = pd.read_csv(filepath)

# Display the first few rows of the dataset and its summary information
df_info = df.info()
df_head = df.head()

df_info, df_head
```
```python
# Import libraries for data preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Check for missing values
missing_values_check = df.isnull().sum()

# Encoding categorical variables
# We'll use LabelEncoder for 'Department' and 'salary' as they are nominal categories
le = LabelEncoder()
df['Department_encoded'] = le.fit_transform(df['Department'])
df['salary_encoded'] = le.fit_transform(df['salary'])

# Feature Scaling
# We'll standardize the numerical features using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['satisfaction_level', 'last_evaluation', 'number_project', 
                                           'average_montly_hours', 'time_spend_company', 
                                           'Work_accident', 'promotion_last_5years', 
                                           'Department_encoded', 'salary_encoded']])

# Update the dataframe with scaled features for the next steps
df_scaled = pd.DataFrame(scaled_features, 
                         columns=['satisfaction_level', 'last_evaluation', 
                                  'number_project', 'average_montly_hours', 
                                  'time_spend_company', 'Work_accident', 
                                  'promotion_last_5years', 'Department_encoded', 
                                  'salary_encoded'])

# Include 'left' column to the scaled dataframe as it is our target variable
df_scaled['left'] = df['left']

missing_values_check, df_scaled.head()
```

## Exploratory Data Analysis (EDA)
Extensive EDA was conducted to explore the data, understand feature distributions, and identify potential relationships between variables and the target outcome.

### Insights from EDA
- Highlighted the distribution of employee satisfaction levels, last evaluation scores, and other key features.
- Analyzed the impact of various factors such as the number of projects and average monthly hours on turnover.

### Python Code Snapshot for EDA

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Set the aesthetic style of the plots
sns.set(style="darkgrid")
plt.style.use("dark_background")

# EDA: Distribution of Numerical Features
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plotting distributions of some key numerical features
sns.histplot(data=df, x='satisfaction_level', kde=True, color="skyblue", ax=axs[0])
axs[0].set_title('Satisfaction Level Distribution')

sns.histplot(data=df, x='last_evaluation', kde=True, color="olive", ax=axs[1])
axs[1].set_title('Last Evaluation Score Distribution')

sns.histplot(data=df, x='average_montly_hours', kde=True, color="gold", ax=axs[2])
axs[2].set_title('Average Monthly Hours Distribution')

fig, axs = plt.subplots(1, 2, figsize=(18, 5))

sns.histplot(data=df, x='time_spend_company', kde=False, color="orange", ax=axs[0], binwidth=0.75)
axs[0].set_title('Years at Company Distribution')

sns.histplot(data=df, x='number_project', kde=False, color="purple", ax=axs[1], binwidth=0.75)
axs[1].set_title('Number of Projects Distribution')

# EDA: Categorical Features vs. Target Variable
fig, axs = plt.subplots(1, 2, figsize=(18, 5))

# Salary level vs. Left
sns.countplot(x='salary', hue='left', data=df, palette="bright", ax=axs[0])
axs[0].set_title('Salary Level vs. Employee Turnover')

# Department vs. Left
sns.countplot(y='Department', hue='left', data=df, palette="bright", ax=axs[1])
axs[1].set_title('Department vs. Employee Turnover')

plt.tight_layout()
plt.show()
```
## Logistic Regression Model

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# Prepare the Data
X = df_scaled.drop('left', axis=1)  # Features
y = df_scaled['left']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)  # Increase max_iter if needed for convergence
log_reg.fit(X_train, y_train)

# Predict on the testing set
y_pred = log_reg.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Model performance metrics
model_performance = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'ROC-AUC Score': roc_auc,
    'Confusion Matrix': conf_matrix
}

model_performance
```

```python
# Adjust the Logistic Regression Model with class weight and regularization strength
# Build the Logistic Regression Model with class weight 'balanced' and a different C value
log_reg_adjusted = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5)
log_reg_adjusted.fit(X_train, y_train)

# Predict on the testing set with the adjusted model
y_pred_adjusted = log_reg_adjusted.predict(X_test)

# Evaluate the Adjusted Model
accuracy_adjusted = accuracy_score(y_test, y_pred_adjusted)
precision_adjusted = precision_score(y_test, y_pred_adjusted)
recall_adjusted = recall_score(y_test, y_pred_adjusted)
roc_auc_adjusted = roc_auc_score(y_test, y_pred_adjusted)
conf_matrix_adjusted = confusion_matrix(y_test, y_pred_adjusted)

# Adjusted model performance metrics
adjusted_model_performance = {
    'Accuracy': accuracy_adjusted,
    'Precision': precision_adjusted,
    'Recall': recall_adjusted,
    'ROC-AUC Score': roc_auc_adjusted,
    'Confusion Matrix': conf_matrix_adjusted
}

adjusted_model_performance
```

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search over
param_grid = {
    'C': [0.01, 0.1, 0.5, 1, 10],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear', 'saga']  # Solver capable of handling l1 penalty
}

# Initialize the grid search model
log_reg_grid = GridSearchCV(LogisticRegression(max_iter=1000, 
                                               class_weight='balanced'), 
                                               param_grid, cv=5, 
                                               scoring='roc_auc', 
                                               verbose=1)

# Fit the grid search model
log_reg_grid.fit(X_train, y_train)

# Best parameters and score from grid search
best_params = log_reg_grid.best_params_
best_score = log_reg_grid.best_score_

best_params, best_score
```

```python
# Rebuild the Logistic Regression Model with the optimized parameters
log_reg_optimized = LogisticRegression(max_iter=1000, 
                                       class_weight='balanced', 
                                       C=best_params['C'], 
                                       penalty=best_params['penalty'], 
                                       solver=best_params['solver'])
log_reg_optimized.fit(X_train, y_train)

# Predict on the testing set with the optimized model
y_pred_optimized = log_reg_optimized.predict(X_test)

# Evaluate the Optimized Model
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
precision_optimized = precision_score(y_test, y_pred_optimized)
recall_optimized = recall_score(y_test, y_pred_optimized)
roc_auc_optimized = roc_auc_score(y_test, y_pred_optimized)
conf_matrix_optimized = confusion_matrix(y_test, y_pred_optimized)

# Optimized model performance metrics
optimized_model_performance = {
    'Accuracy': accuracy_optimized,
    'Precision': precision_optimized,
    'Recall': recall_optimized,
    'ROC-AUC Score': roc_auc_optimized,
    'Confusion Matrix': conf_matrix_optimized
}

optimized_model_performance
```

## Decision Tree Model
Decision Trees are a simple yet powerful tool for classification tasks. This section discusses the application of a Decision Tree model to predict employee turnover.

### Summary of Decision Tree Model
- Describes the basics of Decision Tree models and their applicability in predicting outcomes based on a series of decisions.

### Python Code Snapshot for Decision Tree Model

```python
from sklearn.tree import DecisionTreeClassifier

# Train the Decision Tree Model
decision_tree = DecisionTreeClassifier(class_weight='balanced', random_state=42)
decision_tree.fit(X_train, y_train)

# Predict on the testing set
y_pred_dt = decision_tree.predict(X_test)

# Evaluate the Decision Tree Model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)

# Decision Tree model performance metrics
dt_performance = {
    'Accuracy': accuracy_dt,
    'Precision': precision_dt,
    'Recall': recall_dt,
    'ROC-AUC Score': roc_auc_dt
}

dt_performance
```


### Results Summary for Decision Tree Model
- Includes accuracy, precision, recall, and ROC-AUC scores obtained from the Decision Tree model.

## Other Models and Comparisons
The project also explores additional models, including Random Forest and XGBoost, comparing their performance in predicting employee turnover.

### Python Code Snapshot for Random Forest Model

```python
from sklearn.ensemble import RandomForestClassifier

# Train the Random Forest Model
random_forest = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
random_forest.fit(X_train, y_train)

# Predict on the testing set
y_pred_rf = random_forest.predict(X_test)

# Evaluate the Random Forest Model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)

# Random Forest model performance metrics
rf_performance = {
    'Accuracy': accuracy_rf,
    'Precision': precision_rf,
    'Recall': recall_rf,
    'ROC-AUC Score': roc_auc_rf
}

rf_performance
```

### Python Code Snapshot for XGBoost Model

```python
from xgboost import XGBClassifier

# Train the XGBoost Model
xgboost = XGBClassifier(use_label_encoder=False, 
                        eval_metric='logloss', 
                        scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1])
xgboost.fit(X_train, y_train)

# Predict on the testing set
y_pred_xgb = xgboost.predict(X_test)

# Evaluate the XGBoost Model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb)

# XGBoost model performance metrics
xgb_performance = {
    'Accuracy': accuracy_xgb,
    'Precision': precision_xgb,
    'Recall': recall_xgb,
    'ROC-AUC Score': roc_auc_xgb
}

xgb_performance
```

## Actionable Insights and Recommendations
Based on the findings from the models, this section provides detailed recommendations for organizations to address the key factors contributing to employee turnover.

### Enhance Employee Engagement and Satisfaction
- Implement regular, anonymous satisfaction surveys to gauge employee morale.
- Develop targeted programs to address common areas of dissatisfaction.

### Career Development Opportunities
- Offer transparent career progression paths and professional development opportunities.
- Personalize development plans to align with individual career goals.

### Evaluate Workload and Work-Life Balance
- Conduct regular reviews of employee workloads and hours to ensure a healthy balance.
- Promote flexible working options where feasible.

### Tailored Retention Strategies
- Use model insights to develop retention strategies focused on high-risk departments.
- Customize interventions based on the unique needs and feedback of each department.

### Feedback and Recognition Programs
- Establish a continuous feedback loop that encourages open communication.
- Implement recognition programs that highlight and reward employee achievements.

### Leverage Predictive Analytics for Proactive Retention
- Regularly apply predictive models to identify employees at risk of turnover.
- Engage at-risk employees with personalized retention strategies.

### Implementation Steps

1. **Data-Driven Decision-Making**
   - Integrate predictive analytics into HR processes for informed decision-making.
   - Regularly update and refine models with new data to maintain accuracy.

2. **Monitor and Evaluate**
   - Define key metrics to assess the effectiveness of retention strategies.
   - Adjust programs based on ongoing analysis and employee feedback.

3. **Stakeholder Engagement**
   - Ensure leadership buy-in and support for retention initiatives.
   - Communicate the value and impact of strategies to all employees.

4. **Continuous Improvement**
   - Foster a culture that prioritizes feedback and continuous improvement.
   - Stay adaptive to new insights and evolving employee needs.

### Expected Outcomes

- **Reduced Turnover Rates**: Direct interventions aimed at the identified predictors of turnover are expected to lower overall turnover rates.
- **Increased Employee Satisfaction**: Addressing key areas of dissatisfaction should lead to improved employee morale and engagement.
- **Enhanced Productivity**: A focus on workload management and work-life balance is anticipated to boost productivity and performance.

## Strategies for Deployment and Monitoring of Machine Learning Models
Discusses strategies for deploying the predictive models into production and setting up monitoring to ensure their effectiveness over time.

### Deployment Strategies

#### Integration with Existing Systems
- Determine integration approaches with HR and IT systems for seamless operation, possibly via API development or batch processing.

#### Model Serving
- Choose a model serving method, such as deploying as a microservice, using cloud-based ML services, or direct application integration.

#### User Interface
- Develop interfaces for HR to interact with the model, like a dashboard or a standalone app, ensuring user-friendliness.

### Monitoring Strategies

#### Performance Tracking
- Establish metrics for ongoing performance evaluation, such as accuracy, precision, recall, ROC-AUC score, and retention rate improvements.

#### Data Drift and Model Decay
- Implement monitoring for data drift and model decay with automated alerts to indicate significant shifts necessitating retraining.

#### Retraining Protocol
- Develop a protocol for periodic model retraining with new data, including performance degradation thresholds that trigger this process.

#### Feedback Loop
- Create feedback mechanisms for collecting end-user insights on model predictions and usability for continuous improvement.

#### Compliance and Ethics
- Ensure model deployment adheres to regulations and ethical guidelines, especially regarding data privacy and nondiscrimination.

### Deployment and Monitoring Tools

- **Deployment Tools**: Docker, Kubernetes, and cloud platforms (AWS, Google Cloud, Azure) for model deployment.
- **Monitoring Tools**: MLflow, Prometheus, Amazon SageMaker Model Monitor, Azure Machine Learning, Google AI Platform for performance tracking and management.

# Conclusion

The Employee Turnover Prediction embarked on an ambitious journey to harness the power of data science and machine learning in addressing a critical challenge faced by organizations worldwide: employee turnover. Through rigorous data preprocessing, insightful exploratory data analysis, and the deployment of a variety of predictive models, this project has not only illuminated the key factors influencing employee decisions to leave but also offered a beacon for strategic interventions aimed at enhancing retention.

The project's foray into models ranging from the interpretable Decision Tree to the robust Random Forest and the advanced XGBoost has showcased the depth and breadth of machine learning's capabilities in dissecting complex human resource issues. The nuanced understanding these models provide goes beyond mere predictions, offering a rich tapestry of insights that underline the multifaceted nature of employee satisfaction and organizational commitment.

Beyond the technical accomplishments, this project stands as a testament to the transformative potential of applying analytical rigor to human-centric challenges. The actionable recommendations derived from our analysis are not merely suggestions but evidence-backed strategies poised to foster a more engaging, fulfilling, and enduring workplace environment.

Yet, the path forward is as compelling as the ground traversed. The dynamic interplay of workforce trends, evolving organizational cultures, and the relentless pace of technological innovation calls for a continuous cycle of analysis, learning, and adaptation. Future explorations might delve into emerging data sources, leverage cutting-edge algorithms, and refine predictive capabilities to keep pace with the shifting landscape of work.
