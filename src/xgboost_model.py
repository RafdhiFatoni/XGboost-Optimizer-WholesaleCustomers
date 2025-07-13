from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def find_best_param(X_train, y_train, param_grid=None, cv=5, scoring='accuracy', verbose=1):
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [1, 1.5, 2]
        }
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    grid_search = GridSearchCV(xgb, param_grid, cv=cv, scoring=scoring, verbose=verbose, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best parameters found:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    best_params_with_objective = grid_search.best_params_.copy()
    best_params_with_objective["objective"] = "binary:logistic"
    return best_params_with_objective

def train_model(X_train, y_train, **best_params):
    model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    predictions = model.predict(X_test)
    return predictions

print("XGBoost module loaded successfully.")