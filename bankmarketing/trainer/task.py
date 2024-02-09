#!/usr/bin/env python
# coding: utf-8

# # Bank marketing prediction with XGBOOST
# 
# Machine Learning with Vertex AI custom model
# Requires XGBoost 
# 
# Uses dataset:
# https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset
# 
# HINT: Dataset is available among Google's public datasets: `'gs://cloud-ml-tables-data/bank-marketing.csv`  
# HINT: The target `Deposit` column indicates whether the client purchased a term deposit (2 = yes, 1 = no)  
# see also: https://www.cloudskillsboost.google/focuses/43565?parent=catalog
# 
# HINT: this is the clean version of notebook, without visualizations - to run on clean XGBoost container

# In[22]:


# basics
import pandas as pd

# model, metrics
import sklearn.metrics as metrics 
import sklearn.model_selection as msel 
import sklearn.preprocessing as proc
import xgboost as xgb

# storage
import google.cloud as gcloud


# In[28]:


dataset_path_google = 'gs://cloud-ml-tables-data/bank-marketing.csv' # HINT: public dataset
df=pd.read_csv(dataset_path_google)
df.sample(5)


# In[24]:


print(df.info())
print('nans:')
df.isna().sum()


# ## Data preprocessing
# 
# * No need to convert / remove: NaNs - none of them
# * No need to scale: XGBoost is unaffected
# * Convert categorical / numerical columns neither scaling
#   * until model's `enable_categorical` param is implemented
# * Target column has [1 2] - must be converted to [0 1], as otherwise  `ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0 1], got [1 2]`

# In[25]:


categorical_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
categorical_columns


# In[26]:


label_encoder = proc.LabelEncoder()
for cat in categorical_columns:
    df[cat] = label_encoder.fit_transform(df[cat])


# In[9]:


target = 'Deposit'
X = df.loc[:, df.columns != target]
y = df[target]
y = y.map({2:1, 1:0}) # 2: 'has deposited (1)', 1: 'no deposit (0)'


# In[14]:


random_state = 7
test_train_split = 0.2
X_train, X_test, y_train, y_test = msel.train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state= random_state
)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[15]:


model = xgb.XGBClassifier(
    objective = 'binary:logistic', 
    # HINT: categorical is unsupported: ValueError: Experimental support for categorical data is not implemented for current tree method yet.
    # enable_categorical = True, # HINT: otherwise complains about categorical columns (e.g. need preprocess)
    random_state = random_state)
model.fit(X_train, y_train)


# In[16]:


print(f'Training score: {model.score(X_train, y_train)}| Test score: {model.score(X_test,y_test)}')


# HINT: unbalanced dataset - poor

# In[17]:


y_pred = model.predict(X_test)
metrics.f1_score(y_test, y_pred)


# ## Save model
# 
# Unfortunately can't just: `model.save_model('gs://my-fine-bucket/model')`
# 
# See also:  
# * XGBoost docu - model saving: https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
# * Google cloud documentation: 
#   * [saving model to Google bucket](https://cloud.google.com/ai-platform/training/docs/training-xgboost?ssp=1&darkschemeovr=1&setlang=en-PH&safesearch=moderate#train-and-upload-model)
#   * [exporting models for prediction](https://cloud.google.com/ai-platform/prediction/docs/exporting-for-prediction#joblib)
#     > HINT: model artifact's filename must exactly match one of specified options!
#   * bucket storage https://cloud.google.com/python/docs/reference/storage/latest/google.cloud.storage.blob.Blob#google_cloud_storage_blob_Blob_upload_from_filename

# In[19]:


# TODO: any project/env variables?
# Export the model to a file
model_file = 'model.bst' # HINT: must be that name - see documentation above
model.save_model(model_file)


# In[29]:


bucket_name = 'tomasz-bucket'
gs_model_path = 'bank-marketing/model/xgboost'

bucket = gcloud.storage.Client().bucket(bucket_name)
blob = bucket.blob(f'{gs_model_path}/{model_file}')

blob.upload_from_filename(model_file)


# In[ ]:




