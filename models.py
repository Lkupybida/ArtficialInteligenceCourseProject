from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from statsmodels.tools.eval_measures import rmse, mse, meanabs
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.base import clone
from ipywidgets import IntProgress, Label, HBox
from IPython.display import display, clear_output
import numpy as np
import warnings
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error as calc_mape

warnings.filterwarnings("ignore", category=UserWarning)

def custom_score(y_true, y_pred, model, complexity_penalty=0.01):
    mse = mean_squared_error(y_true, y_pred)
    if mse <= 9:
        mse = 9

    if hasattr(model, 'n_estimators'):
        complexity = model.n_estimators
    elif hasattr(model, 'coef_'):
        complexity = np.sum(np.abs(model.coef_))
    else:
        complexity = 0

    return mse + complexity_penalty * complexity


def objective(trial, regressor, X, y, param_space, fixed_params):
    sampled_params = {key: trial.suggest_categorical(key, values) if isinstance(values, list)
    else trial.suggest_int(key, values[0], values[1]) if isinstance(values, tuple) and all(
        isinstance(i, int) for i in values)
    else trial.suggest_float(key, values[0], values[1])
                      for key, values in param_space.items()}
    params = {**fixed_params, **sampled_params}
    model = clone(regressor).set_params(**params)
    kf = KFold(n_splits=5, shuffle=True)
    scores = []
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred_train = model.predict(X_train_fold)
        y_pred_val = model.predict(X_val_fold)
        train_score = custom_score(y_train_fold, y_pred_train, model)
        val_score = custom_score(y_val_fold, y_pred_val, model)
        score = val_score + max(0, train_score - val_score)
        scores.append(score)
    return np.mean(scores)


def optimize_regressor(regressor, param_space, X, y, n_trials=100, fixed_params={}, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    progress = IntProgress(min=0, max=n_trials)
    percentage_label = Label(value="0%")
    progress_box = HBox([progress, percentage_label])
    display(progress_box)
    for i in range(n_trials):
        study.optimize(lambda trial: objective(trial, regressor, X_train, y_train, param_space, fixed_params),
                       n_trials=1)
        progress.value += 1
        percentage = (i + 1) / n_trials * 100
        percentage_label.value = f"{percentage:.2f}%"
    best_params = study.best_params
    best_model = clone(regressor).set_params(**best_params, **fixed_params)
    best_model.fit(X_train, y_train)
    y_pred_test = best_model.predict(X_test)
    test_score = custom_score(y_test, y_pred_test, best_model)
    y_pred_train = best_model.predict(X_train)
    train_score = custom_score(y_train, y_pred_train, best_model)
    overfitting_metric = max(0, train_score - test_score)

    return study, best_model, test_score, overfitting_metric

def train_test(df):
    X = df[df.columns.drop('kWh')]
    y = df['kWh']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train['kWh'] = y_train
    X_test['kWh'] = y_test

    train = X_train
    test = X_test
    return train, test

class NoSuchModelFoundError(Exception):
    """Custom exception raised when a model cannot be found."""
    def __init__(self, message="No such model found"):
        self.message = message
        super().__init__(self.message)

def get_params(model_name, train, categorical):
    param_space_lgbm = {
        'n_estimators': (100, 1000),
        'learning_rate': (0.01, 0.3),
        'num_leaves': (2, 1000),
        'max_depth': (-1, 30),
        'min_data_in_leaf': (1, 100),
        # 'feature_fraction': (0.1, 1.0),
        # 'bagging_fraction': (0.1, 1.0),
        'bagging_freq': (0, 10),
        'lambda_l1': (0.0, 10.0),
        'lambda_l2': (0.0, 10.0),
        # 'min_gain_to_split': (0.0, 10.0),
        'reg_sqrt': [True, False],
        'max_bin': (1, 10000),
    }
    fixed_params_lgbm = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'verbosity': -1,
        # 'early_stopping_round': 50
    }

    param_space_xgb = {
        'n_estimators': (100, 1100),
        'max_depth': (2, 11),
        'learning_rate': (0.01, 0.1),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'gamma': (0, 1),
        'lambda': (1e-8, 10.0),
        'alpha': (1e-8, 10.0),
    }
    fixed_params_xgb = {'objective': 'reg:squarederror'}
    if model_name == 'LGBM':
        model = lgb.LGBMRegressor()
        param_space = param_space_lgbm
        fixed_params = fixed_params_lgbm
        if categorical != None:
            fixed_params['categorical_feature'] = categorical
            for feature in categorical:
                train[feature] = train[feature].astype('category')
    elif model_name == 'XGB':
        model = XGBRegressor()
        param_space = param_space_xgb
        fixed_params = fixed_params_xgb
        if categorical != None:
            fixed_params['enable_categorical'] = True
            for feature in categorical:
                train[feature] = train[feature].astype('category')
    else:
        try:
            raise NoSuchModelFoundError()
        except NoSuchModelFoundError as e:
            print(e)

    return model, param_space, fixed_params, train

def optimize_model(model_name, train, features, categorical=None, trials=1):

    model, param_space, fixed_params, train = get_params(model_name, train, categorical)

    X = train[features]
    X = X.loc[:, ~X.columns.duplicated()]
    y = train['kWh']

    study, best_model, cv_score, test_score = optimize_regressor(model, param_space, X, y,
                                                                      n_trials=trials,
                                                                      fixed_params=fixed_params)

    print("Best parameters:", study.best_params)
    print(f"Nested CV RMSE: {cv_score}")
    print(f"Test set score: {test_score}")
    return study, best_model, cv_score, test_score

def vizualize_percentiles(df, test, val, model, features):
    azk_list = df.columns.tolist()[526 - 81:]
    errors_df = pd.DataFrame(columns=['InternalNum', 'Valid', 'Test', 'Mean', 'Val_Preds', 'Test_Preds'])
    for col in azk_list:
        test_x = test[test[col] == True]
        val_x = val[val[col] == True]
        try:
            valid_preds_x = model.predict(val_x[features])
            test_preds_x = model.predict(test_x[features])
            # print(val_x)
            err_v = mape(val_x['kWh'], valid_preds_x)
            err_t = mape(test_x['kWh'], test_preds_x)
            err_m = (err_v + err_t) / 2
            errors_df.loc[len(errors_df)] = [col, err_v, err_t, err_m, valid_preds_x, test_preds_x]
        except:
            continue

    percentiles = errors_df['Mean'].quantile([0, 0.25, 0.5, 0.75, 1])
    k = 0
    names = ['AZK with smallest error', 'AZK with 25% percentile error', 'AZK with median error',
             'AZK with 75% percentile error', 'AZK with biggest error']
    for i in percentiles:
        num = errors_df[errors_df['Mean'] == i]
        test_x = test[test[num['InternalNum'].values[0]] == True]
        val_x = val[val[num['InternalNum'].values[0]] == True]
        plt.figure(figsize=(15, 6))
        plt.style.use('dark_background')
        plt.title(names[k] + f' Error: {i}')
        plt.plot(test_x['datetime'], num['Test_Preds'].values[0], color='#91be1e', linestyle='--')
        plt.plot(test_x['datetime'], test_x['kWh'], color='white')
        k = k + 1

def mape(y_true, y_pred):
    y_true_adjusted = np.array(y_true) + 1
    y_pred_adjusted = np.array(y_pred) + 1

    mape = np.mean(np.abs((y_true_adjusted - y_pred_adjusted) / y_true_adjusted))
    return mape