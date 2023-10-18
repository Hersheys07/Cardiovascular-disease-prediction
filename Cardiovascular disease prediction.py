#!/usr/bin/env python
# coding: utf-8

# **INFO583: Final Project Python Code**
# > 
# **Professor Mamonov**
# > 
# **Group 2 – Ian Escalona || Harshitha Nagartapeta || 
# Olid Sarker (ID: 50067536)**

# **Business Problem:**
# Cardiovascular disease is a major cause of morbidity and mortality worldwide, and it has significant financial and social implications. The Cardiovascular Disease dataset contains information on various risk factors associated with CVD, such as age, gender, blood pressure, cholesterol levels, and smoking status. 
# 
# The business problem of developing a predictive model for cardiovascular disease could have significant implications for improving patient outcomes and reducing healthcare costs, making it an important area of focus for healthcare providers and insurance companies. Here are some of the financial and social implications of CVD that can be inferred from the dataset:
# 
# **Financial Implications:** 
# Cardiovascular disease is a leading cause of healthcare costs and lost productivity. According to the World Health Organization, cardiovascular disease is responsible for approximately $863 billion in healthcare costs globally. By developing a predictive model to identify high-risk individuals, healthcare providers and insurance companies can target interventions to prevent or manage cardiovascular disease, potentially reducing healthcare costs and increasing efficiency in healthcare delivery.
# 
# **Social Implications:** Cardiovascular disease can have a significant impact on quality of life, including physical, emotional, and financial consequences. Patients with cardiovascular disease may experience reduced mobility, chronic pain, and increased healthcare utilization, which can lead to lost income and social isolation. By identifying high-risk individuals and intervening early, healthcare providers and insurance companies can help prevent the social and economic consequences of cardiovascular disease.

# Dataset source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
# > 
# The dataset we will be using for this project is named "Cardiovascular Disease dataset" which was obtained from Kaggle. This dataset includes 70,000 patient records, 11 features, and a target variable which indicates the presence or absence of Cardiovascular disease. All of the values in this dataset were collected at the moment of medical examination.
# 

# Features:
# 1. age (days)
# 2. height (cm) 
# 3. weight (kg) 
# 4. gender (1 = women, 2 = men)
# 5. ap_hi (systolic blood pressure)
# 6. ap_lo (diastolic blood pressure)
# 7. cholesterol (1 = normal, 2 = above normal, 3 = well above normal)
# 8. gluc (1 = normal, 2 = above normal, 3 = well above normal)
# 9. smoke (whether a patient smokes or not)
# 10. alco (whether a patient drinks alcohol or not)
# 11. active (whether a patient engages in physical activities)
# 12. cardio (target variable – presence or absence of cardiovascular disease)

# In[ ]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


# In[ ]:


from google.colab import files
uploaded = files.upload()

import io
cardio_data = pd.read_csv(io.BytesIO(uploaded['Cardiovascular_Disease.csv']))


# In[ ]:


cardio_data = cardio_data.drop('id', axis = 1)


# In[ ]:


cardio_data.head()


# In[ ]:


# Age has been converted from days to years

cardio_data['age']= (cardio_data['age']/365.242199).round()


# In[ ]:


cardio_data.info()


# In[ ]:


# Gender data type has been converted to category

cardio_data['gender'] =  cardio_data['gender'].astype('category',copy=False)


# In[ ]:


cardio_data.describe()


# In[ ]:


cardio_data.cardio.hist()


# The above histogram shows the distribution of the target variable, "cardio". We see that this dataset is balanced – 35,000 observations are classified as 0 (patient does not have cardiovascular disease) and 35,000 observations are classified as 1 (patient has cardiovascular disease).

# In[ ]:


sns.heatmap(cardio_data.isnull(), cbar=False)


# The above heatmap shows there are no missing values in this dataset. Missing value imputation is not necessary.

# In[ ]:


sns.pairplot(cardio_data[['cardio','age','active','weight','gluc','cholesterol']],hue ='cardio')


# *   Patients that are over the age of 55 years are more likely to have Cardiovascular disease.
# *   Patients who weigh more tend to have Cardiovascular disease.
# *   Patients with high glucose levels are at risk for developing Cardiovascular disease.
# *  Patients with higher cholesterol levels are more likely to have Cardiovascular disease.
# *  Patients who do not regularly engage in physical activities are more likely to develop Cardiovascular disease.
# 
# 
# 
# 
# 
# 
# 
# 
# 

# **Logistic Regression Model:**

# In[ ]:


X = cardio_data.drop('cardio', axis = 1)
y = cardio_data['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 24)


# In[ ]:


log_model = LogisticRegression(solver = 'liblinear')

log_model.fit(X_train, y_train)


# In[ ]:


import statsmodels.api as sm

logit_model = sm.Logit(y_train, X_train)
logmodel_2=logit_model.fit()
print(logmodel_2.summary2())


# Based on the .05 significance level, feature "smoke" does not have a statistically significant effect on the target variable – "cardio". We will rebuild the logistic regression model below excluding this feature.

# In[ ]:


# Logistic Regression model rebuild (excludes features that are not statistically significant)

cardio_sig = cardio_data.drop(columns = ['smoke'])

X_sig = cardio_sig.drop('cardio', axis = 1)
y_sig = cardio_sig['cardio']


# 

# In[ ]:


X_sig_train, X_sig_test, y_sig_train, y_sig_test = train_test_split(X_sig, y_sig, test_size = 0.3, random_state = 24)


# In[ ]:


log_model_sig = LogisticRegression(solver = 'liblinear')

log_model_sig.fit(X_sig_train, y_sig_train)


# In[ ]:


y_sig_predict = log_model_sig.predict(X_sig_test)


# In[ ]:


logit_model_sig = sm.Logit(y_sig_train, X_sig_train)
logmodel_sig_2 = logit_model_sig.fit()
print(logmodel_sig_2.summary2())


# In[ ]:


print("Logistic Regression – Confusion Matrix:")
confusion_matrix(y_sig_test, y_sig_predict)


# In[ ]:


print("Logistic Regression – Classification Report:")
print(classification_report(y_sig_test,y_sig_predict))


# In[ ]:


logit_roc_auc = roc_auc_score(y_sig_test, log_model_sig.predict_proba(X_sig_test)[:,1])
fpr, tpr, thresholds = roc_curve(y_sig_test, log_model_sig.predict_proba(X_sig_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# **kNN Model:**

# In[ ]:


# Rescaling features

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_ = scaler.fit_transform(X)

X_rescaled = pd.DataFrame(X_, columns=X.columns)


# In[ ]:


X_rescaled.describe()


# In[ ]:


X_knn_train, X_knn_test, y_knn_train, y_knn_test = train_test_split(X_rescaled, y, test_size=0.3, random_state=33)


# In[ ]:


from sklearn.model_selection import cross_val_score

max_K = 100
cv_scores = [ ]

for K in range(1,max_K):
    knn = KNeighborsClassifier(n_neighbors = K)
    scores = cross_val_score(knn,X_knn_train,y_knn_train.values.ravel(),cv = 5,scoring = "roc_auc")
    cv_scores.append(scores.mean())


# In[ ]:


sns.lineplot(x=range(1,max_K), y=cv_scores)


# In[ ]:


max(cv_scores)


# In[ ]:


cv_scores.index(max(cv_scores))


# The highest value of ROC AUC occurs when k = 92. We will rebuild an optimized kNN model based on ROC AUC scoring below:

# In[ ]:


X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_rescaled, y, test_size=0.3, random_state=33)


# In[ ]:


# Optimized kNN model

knn_opt = KNeighborsClassifier(n_neighbors=92, metric='euclidean')
knn_opt.fit(X_opt_train, y_opt_train)

y_opt_pred = knn_opt.predict(X_opt_test)


# In[ ]:


print("kNN – Confusion Matrix:")
confusion_matrix(y_opt_test, y_opt_pred)


# In[ ]:


print("kNN – Classification Report:")
print(classification_report(y_opt_test,y_opt_pred))


# In[ ]:


knn_roc_auc = roc_auc_score(y_opt_test, knn_opt.predict_proba(X_opt_test)[:,1])
fpr, tpr, thresholds = roc_curve(y_opt_test, knn_opt.predict_proba(X_opt_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Optimized kNN (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Optimized kNN')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# **Random Forest Model:**

# In[ ]:


rf_model  = RandomForestClassifier(max_depth=5, random_state=0)
rf_model.fit(X_train,y_train)

y_pred_rf = rf_model.predict(X_test)


# In[ ]:


print("Random Forest – Confusion Matrix:")
confusion_matrix(y_test, y_pred_rf)


# In[ ]:


print("Random Forest – Classification Report:")
print(classification_report(y_test,y_pred_rf))


# In[ ]:


rf_roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Random Forest')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# **Boosted Tree Model:**

# In[ ]:


bt_model = AdaBoostClassifier(n_estimators=10000, random_state=24)

bt_model.fit(X_train,y_train)

y_pred_bt = bt_model.predict(X_test)


# In[ ]:


print("Boosted Tree – Confusion Matrix:")
confusion_matrix(y_test, y_pred_bt)


# In[ ]:


print("Boosted Tree – Classification Report:")
print(classification_report(y_test,y_pred_bt))


# In[ ]:


bt_roc_auc = roc_auc_score(y_test, bt_model.predict_proba(X_test)[:,1])
bt_fpr, bt_tpr, bt_thresholds = roc_curve(y_test, bt_model.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(bt_fpr, bt_tpr, label='Boosted tree (area = %0.2f)' % bt_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# **Decision Tree Model:**

# In[ ]:


X1 = cardio_data.drop('cardio', axis = 1)
Y1 = cardio_data['cardio']


# In[ ]:


X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.3, random_state=1)


# In[ ]:


from sklearn import tree
dt_model = tree.DecisionTreeClassifier(min_samples_leaf=5, max_depth=3)

dt_model.fit(X1_train,Y1_train)


# In[ ]:


Y1_pred = dt_model.predict(X1_test)


# In[ ]:


print("Decision Tree – Confusion Matrix:")
confusion_matrix(Y1_test,Y1_pred)


# In[ ]:


print("Decision Tree – Classification Report:")
print(classification_report(Y1_test,Y1_pred))


# In[ ]:


dt_roc_auc = roc_auc_score(Y1_test, dt_model.predict_proba(X1_test)[:,1])
fpr, tpr, thresholds = roc_curve(Y1_test, dt_model.predict_proba(X1_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree (area = %0.2f)' % dt_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Decision Tree')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[ ]:


import graphviz 
dot_data = tree.export_graphviz(dt_model, out_file=None, 
                      feature_names=X1.columns,  
                      class_names=['No cardio','cardio'],
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# **Naive Bayes Model:**

# In[ ]:


cardio_data['age_nb'] = cardio_data['age'].apply(lambda x: 1 if x > 54 else 0)
cardio_data['cholesterol_nb'] = cardio_data['cholesterol'].apply(lambda x: 1 if x > 2 else 0)
cardio_data['weight_nb'] = cardio_data['weight'].apply(lambda x: 1 if x > 74 else 0)


# In[ ]:


cardio_data.hist(column=['age_nb'],)


# In[ ]:


cardio_data.hist(column=['cholesterol_nb'],)


# In[ ]:


cardio_data.hist(column=['weight_nb'],)


# In[ ]:


cardio_data.describe()


# In[ ]:


cardio_data.columns


# In[ ]:


cardio_nb_data = cardio_data[['gender', 'height',
       'gluc', 'smoke', 'alco', 'active', 'cardio', 'age_nb', 'cholesterol_nb',
       'weight_nb']]


# In[ ]:


print('Proportion of patients with cardiovascular disease: ', cardio_nb_data.cardio.mean())


# In[ ]:


X_nb = pd.get_dummies(cardio_nb_data.drop('cardio', axis=1))
y_nb = cardio_nb_data['cardio'].astype('category')


# In[ ]:


X_nb.loc[:, X_nb.isnull().any()].columns


# In[ ]:


X_nb_train, X_nb_test, y_nb_train, y_nb_test = train_test_split(X_nb, y_nb, test_size=0.33, random_state=100)


# In[ ]:


cardio_nb = MultinomialNB(alpha=0.01)
cardio_nb.fit(X_nb_train, y_nb_train) 

y_nb_pred = cardio_nb.predict(X_nb_test)

print(confusion_matrix(y_nb_test,y_nb_pred))
print(classification_report(y_nb_test,y_nb_pred))


# In[ ]:


nb_roc_auc = roc_auc_score(y_nb_test, cardio_nb.predict_proba(X_nb_test)[:,1])
nb_fpr, nb_tpr, nb_thresholds = roc_curve(y_nb_test, cardio_nb.predict_proba(X_nb_test)[:,1])

plt.figure()
plt.plot(nb_fpr, nb_tpr, label='Naive Bayes (area = %0.2f)' % nb_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Naive Bayes')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


get_ipython().system('pip install eli5')


# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(cardio_nb, random_state=1).fit(X_nb_test, y_nb_test)
eli5.show_weights(perm, feature_names = X_nb_test.columns.tolist())


# In[ ]:


x_y = pd.concat([X_nb_test,y_nb_test], axis=1)

sns.pairplot(x_y[['cholesterol_nb','age_nb','weight_nb','cardio']], hue='cardio')


# **Artificial Neural Network Model:**

# In[ ]:


X_ann = cardio_data.drop(cardio_data[['cardio','age_nb','cholesterol_nb','weight_nb']], axis = 1)
y_ann = cardio_data['cardio']


# In[ ]:


X_ann_train, X_ann_test, y_ann_train, y_ann_test = train_test_split(X_ann,y_ann,test_size=0.30,random_state=24)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_ann_train = scaler.transform(X_train)
X_ann_test = scaler.transform(X_test)


# In[ ]:


X_train_df = pd.DataFrame(X_ann_train, columns = X.columns)

X_train_df.describe().transpose()


# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


MLPC_model = MLPClassifier(hidden_layer_sizes=(20,), activation='logistic', solver='adam', random_state=24)

MLPC_model.fit(X_ann_train, y_ann_train)

y_pred_mlpc = MLPC_model.predict(X_ann_test)


# In[ ]:


print(confusion_matrix(y_ann_test,y_pred_mlpc))
print(classification_report(y_ann_test,y_pred_mlpc))
print('ROC AUC: ', roc_auc_score(y_ann_test,MLPC_model.predict_proba(X_ann_test)[:,1]))


# In[ ]:


ann_roc_auc = roc_auc_score(y_ann_test, MLPC_model.predict_proba(X_ann_test)[:,1])
ann_fpr, ann_tpr, ann_thresholds = roc_curve(y_ann_test, MLPC_model.predict_proba(X_ann_test)[:,1])

plt.figure()
plt.plot(ann_fpr, ann_tpr, label='Artificial Neural Network (area = %0.2f)' % ann_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Artificial Neural Network')
plt.legend(loc="lower right")
plt.show()


# **ROC AUC Summary:**

# In[ ]:


logit_roc_auc = roc_auc_score(y_sig_test, log_model_sig.predict_proba(X_sig_test)[:,1])
fpr_log, tpr_log, thresholds = roc_curve(y_sig_test, log_model_sig.predict_proba(X_sig_test)[:,1])
fpr_knn, tpr_knn, thresholds = roc_curve(y_opt_test, knn_opt.predict_proba(X_opt_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])
bt_fpr, bt_tpr, bt_thresholds = roc_curve(y_test, bt_model.predict_proba(X_test)[:,1])
fpr, tpr, thresholds = roc_curve(Y1_test, dt_model.predict_proba(X1_test)[:,1])
nb_fpr, nb_tpr, nb_thresholds = roc_curve(y_nb_test, cardio_nb.predict_proba(X_nb_test)[:,1])
ann_fpr, ann_tpr, ann_thresholds = roc_curve(y_ann_test, MLPC_model.predict_proba(X_ann_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(fpr_knn, tpr_knn, label='Optimized kNN (area = %0.2f)' % logit_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot(bt_fpr, bt_tpr, label='Boosted tree (area = %0.2f)' % bt_roc_auc)
plt.plot(fpr, tpr, label='Decision Tree (area = %0.2f)' % dt_roc_auc)
plt.plot(nb_fpr, nb_tpr, label='Naive Bayes (area = %0.2f)' % nb_roc_auc)
plt.plot(ann_fpr, ann_tpr, label='Artificial Neural Network (area = %0.2f)' % ann_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Summary')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

