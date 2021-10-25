import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from helpers.data_prep import *
from helpers.eda import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression,Ridge,ElasticNet,Lasso
from sklearn.ensemble import VotingRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

df = pd.read_csv("hitters.csv")

############################################
# EDA
############################################

check_df(df)


######################
# FEATURE EXTRACTION #
######################

df['New_AtBat_Hits_Ratio'] = df['Hits'] / df['AtBat']
df['New_CAtBat_CHits_Ratio'] = df['CHits'] / df['CAtBat']


df['New_Total_Success'] =  df['Hits'] + df['Runs'] + df['RBI'] + df['Walks'] + df['Assists'] - df['Errors']


# Correlation Analysis #

def target_correlation_matrix(dataframe, corr_th=0.50, target="Salary"):

    corr = dataframe.corr()
    corr_th = corr_th
    try:
        filter = np.abs(corr[target]) > corr_th
        corr_features = corr.columns[filter].tolist()
        sns.clustermap(dataframe[corr_features].corr(), annot=True, fmt=".2f")
        plt.show()
        return corr_features
    except:
        print("High threshold value, decrease corr_th value")


target_correlation_matrix(df, corr_th=0.55, target="Salary")

df['New_High_Corr_Variables'] = df['CRuns'] * df['CRBI']



df.drop(columns = ['CRuns', 'CRBI'],inplace = True)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

## Outlier Check ##

num_cols = [col for col in num_cols if col != 'Salary']

for col in num_cols:
    print(check_outlier(df,col))

for col in num_cols:
    replace_with_thresholds(df,col)

## Handling Missing Values ##

df.dropna(inplace=True)
df.isnull().values.any()

# Label Encoder #
binary_col = [col for col in df.columns if df[col].dtype not in [int,float] and df[col].nunique() == 2]
for col in binary_col:
    df = label_encoder(df,col)


# One Hot Encoder #
cat_cols, num_cols, cat_but_car = grab_col_names(df)
df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

# Scale #
num_cols.remove("Salary")
scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()

############################################
# MODELLING
############################################

y = df['Salary']
X = df.drop("Salary", axis=1)

################################
#  Automated Machine Learning  #
################################

regressors = [('LR', LinearRegression()),
              ("Ridge", Ridge()),
              ("Lasso", Lasso()),
               ("ElasticNet", ElasticNet()),
               ('KNN', KNeighborsRegressor()),
               ("CART", DecisionTreeRegressor()),
               ("RF", RandomForestRegressor()),
               ('GBM', GradientBoostingRegressor()),
               ('XGBoost', XGBRegressor()),
               ('LightGBM', LGBMRegressor()),
              ]



for name, regressor in regressors:
    result = np.mean(np.sqrt(-cross_val_score(regressor,
                                     X,
                                     y,
                                     cv=10,
                                     scoring="neg_mean_squared_error")))

    print(f"mean_squared_error: {result:.4f} : {regressor}")

######################################################
# Automated Hyperparameter Optimization
######################################################

"""
knn_params = {"n_neighbors":range(1,10),
              "weights":["uniform", "distance"],
              "algorithm":["ball_tree", "kd_tree", "brute"],
              "leaf_size":range(1,15)
              }

"""

rf_params = {"criterion":["mse", "poisson"],
             "max_depth": [5,10,None],
             "min_samples_split": [4,15],
             "n_estimators": [500, 1000]}
"""
xgboost_params = {"max_depth": [5, 8, 12, 20],
                  "min_child_weight": [1,2,3],
                  }
"""

gbm_params = {"learning_rate": [0.01, 0.1,0.001,0.2,0.002,0.002],
              "max_depth":[2,4,6,8],
              "n_estimators": [500,1000]
              }

lightgbm_params = {"learning_rate": [0.01, 0.1,0.001,0.2,0.002,0.002],
                   "n_estimators": [300, 500, 1500,2000]}



regressors = [("RF", RandomForestRegressor(),rf_params),
               ('GBM', GradientBoostingRegressor(),gbm_params),
               ('LightGBM', LGBMRegressor(),lightgbm_params),
              ]


best_models = {}


for name, regressor, params in regressors:

    print(f"########## {name} ##########")
    result = np.mean(np.sqrt(-cross_val_score(regressor,
                                              X,
                                              y,
                                              cv=10,
                                              scoring="neg_mean_squared_error")))

    print(f"mean_squared_error: {result:.4f} : {regressor}")

    gs_best = GridSearchCV(regressor, params, cv=10, n_jobs=-1, verbose=False).fit(X, y)
    final_model = regressor.set_params(**gs_best.best_params_)

    result = np.mean(np.sqrt(-cross_val_score(regressor,
                                              X,
                                              y,
                                              cv=10,
                                              scoring="neg_mean_squared_error")))

    print(f"####\nAfter Hyperparameter Optimization ######\nmean_squared_error: {result:.4f} : {regressor}")

    best_models[name] = final_model

######################################################
# Stacking & Ensemble Learning
######################################################

voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('GBM', best_models['GBM']),
                                         ])

voting_reg.fit(X, y)

np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=10, scoring="neg_mean_squared_error")))

######################################################
# Prediction for a New Observation
######################################################

random_user = X.sample(1, random_state=45)


voting_reg.predict(random_user)