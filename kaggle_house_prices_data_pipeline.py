### Basic Pipeline for a Kaggle Competition
### https://www.kaggle.com/c/house-prices-advanced-regression-techniques

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os


data_path = 'data'

train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))


print("train_data shape: ", train_data.shape)
print("test_data shape: ", test_data.shape)


train_data.head()


train_data.info()


train_data.drop('Id', axis=1, inplace=True)

# %% [markdown]
# #### Non-numeric variables

### Get properties of non-numeric variables
train_uniques_df = train_data.select_dtypes('object').apply(lambda x: pd.Series({'num_unique':x.nunique(), 'unique_values':list(x.unique()), 'num_missing':sum(x.isna())})).transpose()
train_uniques_df['num_missing'] = np.around(train_uniques_df['num_missing'].astype('int')/len(train_data.index), decimals = 2)
train_uniques_df['remove_cols'] = train_uniques_df['num_missing']>(1/3)
train_uniques_df[train_uniques_df['remove_cols']]


### Drop categorical columns with >500 missing values
train_data.drop(train_uniques_df.index[train_uniques_df['remove_cols']], axis=1, inplace=True)

# %% [markdown]
# #### Numeric variables

### Get properties of numeric variables
train_num_uniques_df = train_data.describe().transpose()
train_num_uniques_df['num_missing'] = np.around(1 - (train_num_uniques_df['count']/len(train_data.index)), decimals = 2)
train_num_uniques_df['remove_cols'] = train_num_uniques_df['num_missing']>(1/3)
train_num_uniques_df

### No major numeric columns found that requires dropping, but some require imputation

# %% [markdown]
# #### Explore the data

sns.lmplot(x='YearBuilt', y='SalePrice', data=train_data, hue = 'CentralAir', fit_reg=False, scatter_kws={'alpha':0.3})




# %% [markdown]
# #### Data Preprocessing

X_train = train_data.drop('SalePrice', axis=1)
y_train = train_data['SalePrice'].values


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler


catg_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='median')


cat_remaining = train_uniques_df[(~train_uniques_df['remove_cols'])].index
cat_cols_largen = train_uniques_df[(train_uniques_df['num_unique'] >= 10) & (~train_uniques_df['remove_cols'])].index
cat_cols_smalln = train_uniques_df[(train_uniques_df['num_unique'] < 10) & (~train_uniques_df['remove_cols'])].index
num_cols = X_train.select_dtypes(['int64', 'float64']).columns
print(cat_cols_largen, cat_cols_smalln, num_cols)


cat_cols_largen_idx = [i for i, cat in enumerate(cat_remaining) if cat in cat_cols_largen]
cat_cols_smalln_idx = [i for i, cat in enumerate(cat_remaining) if cat in cat_cols_smalln]


ohe = OneHotEncoder(handle_unknown='ignore')
oe = OrdinalEncoder()
ss = StandardScaler()


from sklearn.compose import ColumnTransformer


cat_col_xmer = ColumnTransformer([('catg_ordinal_large', oe, cat_cols_largen_idx), 
                                  ('catg_onehot_small', ohe, cat_cols_smalln_idx)])


num_pipeline = Pipeline([('impute_median', num_imputer), ('standard_scale', ss)])
cat_pipeline = Pipeline([('impute_mode', catg_imputer), ('catg_encoding', cat_col_xmer)])


data_prep = ColumnTransformer([('cat', cat_pipeline, cat_remaining), 
                              ('num', num_pipeline, num_cols)])


from sklearn.linear_model import LinearRegression


lm_pipeline = Pipeline([('data_prep', data_prep), ('lr', LinearRegression())])


model = lm_pipeline.fit(X_train, y_train)


model.predict(test_data)


my_submission = pd.concat([sample_submission, pd.Series(model.predict(test_data))], axis=1)
my_submission.drop('SalePrice', axis=1, inplace=True)
my_submission.columns = sample_submission.columns
my_submission.head(3)


my_submission.to_csv('submissions/submission1.csv', index=False)
