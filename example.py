from sklearn.pipeline import Pipeline
from preprocessor import get_train_environment

envGen = get_train_environment(n_splits=20)

# The holdout set is not included in the cross validation folds, use it at the very end to evaluate overall performance
stats, X, X_holdout, y, y_holdout = next(envGen)

# Cross validation folds
for fold, stats, X_train, X_test, y_train, y_test in envGen:
    print(fold)
    print(X_train)