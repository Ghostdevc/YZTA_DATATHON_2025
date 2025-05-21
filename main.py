# YZTA Datathon 2025
#Importing Libraries

from yzta_datathon_2025_eda.utils import load_train, load_test, check_df, concat_df_on_y_axis, create_date_features, create_time_series_features, one_hot_encoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_train = load_train()
df_test = load_test()

check_df(df_train)
check_df(df_test)

df_train_test = concat_df_on_y_axis(df_train, df_test)

check_df(df_train_test)

# Standardizing the column names
df_train_test.columns = df_train_test.columns.str.lower().str.replace(' ', '_').str.strip()

# Note: The original Turkish column names are replaced with their English equivalents
# 'id' -> 'id', 'tarih' -> 'date', 'ürün' -> 'product', 'ürün_besin_değeri' -> 'nutrition', 'ürün_kategorisi' -> 'category', 'ürün_fiyatı' -> 'price', 'ürün_üretim_yeri' -> 'origin', 'market' -> 'market', 'şehir' -> 'city'

df_train_test.rename(columns={
    'id': 'id',
    'tarih': 'date',
    'ürün': 'product',
    'ürün_besin_değeri': 'nutrition',
    'ürün_kategorisi': 'category',
    'ürün_fiyatı': 'price',
    'ürün_üretim_yeri': 'origin',
    'market': 'market',
    'şehir': 'city'
}, inplace=True)

# Standardizing the string values in the columns
cols_to_standardize = ['product', 'origin', 'market', 'city', 'category']

for col in cols_to_standardize:
    df_train_test[col] = df_train_test[col].str.lower().str.replace(' ', '_').str.strip()
    df_train_test[col] = df_train_test[col].str.replace('ç', 'c').str.replace('ğ', 'g').str.replace('ı', 'i').str.replace('ö', 'o').str.replace('ş', 's').str.replace('ü', 'u')

df_train_test = create_date_features(df=df_train_test)

#sorting the data
df_train_test.sort_values(by=['product', 'origin', 'market', 'city', 'category', 'date'], inplace=True)

lags = [1, 3, 6, 9, 12, 24, 36, 48]

windows = [3, 6, 9, 12, 24, 36, 48]

breakdowns = ['product', 'origin', 'market', 'city', 'category']

df_train_test = create_time_series_features(df=df_train_test, target='price', lags=lags, windows=windows, breakdown_columns=breakdowns,  include_random_noise=True)


columns_to_encode = ['city', 'market', 'product', 'nutrition', 'category', 'origin']
df_train_test = one_hot_encoder(dataframe=df_train_test, categorical_cols=columns_to_encode, drop_first=True)

df_train_test['price'] = np.log1p(df_train_test["price"].values)

# MODEL TRAINING

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def lgbm_rmse(preds, train_data):
    labels = train_data.get_label()  # Extract labels from the Dataset
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    return 'rmse', rmse, False


# Time based validation sets

#[2019.01.01 - 2022-12-01] for training | [2023-01-01 - 2023-12-01] for validation | [2024-01-01 - 2024-12-01] for test

train_start = '2019-01-01'
train_end = '2022-12-01'
valid_start = '2023-01-01'
valid_end = '2023-12-01'
test_start = '2024-01-01'
test_end = '2024-12-01'

df_train_preprocessed = df_train_test[(df_train_test['date'] >= train_start) & (df_train_test['date'] <= train_end)]
df_valid_preprocessed = df_train_test[(df_train_test['date'] >= valid_start) & (df_train_test['date'] <= valid_end)]
df_test_preprocessed = df_train_test[(df_train_test['date'] >= test_start) & (df_train_test['date'] <= test_end)]

cols = [col for col in df_train_test.columns if col not in ['date', 'id', "price"]]

df_test_preprocessed.sort_values(by=['id'], inplace=True)

df_test_preprocessed.head(10)

X_train = df_train_preprocessed[cols]
y_train = df_train_preprocessed["price"]

X_valid = df_valid_preprocessed[cols]
y_valid = df_valid_preprocessed["price"]

X_test = df_test_preprocessed[cols]

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape

# LGBM Model Training
import lightgbm as lgb

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgb_train = lgb.Dataset(data=X_train, label=y_train, feature_name=cols)

lgb_valid = lgb.Dataset(data=X_valid, label=y_valid, reference=lgb_train, feature_name=cols)

model = lgb.train(lgb_params, lgb_train,
                  valid_sets=[lgb_train, lgb_valid],
                  num_boost_round=lgb_params['num_boost_round'],
                  #early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_rmse,
                  #verbose_eval=100
                  )

y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

rmse(np.expm1(y_pred_valid), np.expm1(y_valid))

# Plotting feature importance

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=30, plot=True)

# FINAl MODEL TRAINING
# Final model training on the entire training set (train + valid) and predicting the test set

df_train_final = df_train_test.loc[~df_train_test['price'].isna()]

X_train_final = df_train_final[cols]
y_train_final = df_train_final["price"]

lgb_final_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train_final, label=y_train_final, feature_name=cols)

final_model = lgb.train(lgb_final_params, lgbtrain_all, num_boost_round=model.best_iteration)

# saving the model
import joblib
joblib.dump(final_model, 'lgbm_model.pkl')

#Predicting the test set

test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

########################
# Submission File
########################

submission_df = df_test_preprocessed.loc[:, ["id", "price"]]
submission_df['price'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

submission_df.rename(columns={'price': 'ürün fiyatı'}, inplace=True)

submission_df.head(10)
submission_df.tail(10)
submission_df.describe().T

submission_df.to_csv('yzta_datathon_2025_eda/data/lgb_model_predictions_submission.csv', index=False)
print("Submission file created successfully!")