import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from os import mkdir, listdir
from os.path import join
import numpy as np
from pickle import load, dump

def addStats(stats, name, var):
    stats[name + "_median"] = var.median()
    stats[name + "_iqr"] = var.astype(np.float64).quantile(.75) - var.astype(np.float64).quantile(.25)

def getStats(X_train, X_test):
    stats = {}
    addStats(stats, "X_train", X_train)
    addStats(stats, "X_test", X_test)
    
    return stats
    
def get_train_environment(n_splits = 20, filename = "FinalData.csv", seed=4321):
    cachedFolds = False
    cachedData = False
    # If you wish to delete the cache, please delete the entire cache folder
    if "cache" in listdir():
        cachedData = True
        if str(n_splits) + ".pickle" in listdir("cache"):
            print("Found cached folds, using those")
            cachedFolds = True
        else:
            print("No cached folds found, generating new folds.")
    else:
        print("No cache found, generating holdout set and folds.")
    
    if cachedData == False:
        data = None
        try:
            data = pd.read_csv(filename)
            data = data.drop("Unnamed: 0", axis=1)
        except:
            print(f"Error, unable to read file {filename}, you can provide the filename via filename = <filename>")
            exit()
        
        data = data.drop(["Name", "appid"], axis=1)

        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])

        X = data.drop("ln.pricep1", axis=1)
        y = data["ln.pricep1"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, shuffle=True, test_size=0.05)

        X_train_scaled = pipeline.fit_transform(X_train)
        X_test_scaled = pipeline.fit_transform(X_test)
        
        # Needed to reverse the scaling
        stats = getStats(X_train, X_test)

        mkdir("cache")

        with open(join("cache", "stats_X_scaled.pickle"), 'wb') as file:
            dump([stats, X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test], file)

        yield stats, X_train_scaled, X_test_scaled, y_train, y_test

        if n_splits == 20:
            print(f"Using default n_splits = {20}, please change depending on what problem you are solving.")

        kf = KFold(n_splits=n_splits)

        toDump = []

        i = 0
        for train_index, test_index in kf.split(X_train, y_train):
            X_folded_train = X_train.iloc[train_index]
            y_folded_train = y_train.iloc[train_index]
            X_folded_test = X_train.iloc[test_index]
            y_folded_test = y_train.iloc[test_index]
            stats = getStats(X_folded_train, X_folded_test)

            X_folded_train_scaled = pipeline.fit_transform(X_folded_train)
            X_folded_test_scaled = pipeline.fit_transform(X_folded_test)

            toDump.append((i, stats, (X_folded_train_scaled, X_folded_test_scaled, y_folded_train, y_folded_test)))
            yield i, stats, X_folded_train_scaled, X_folded_test_scaled, y_folded_train, y_folded_test
            i += 1
        
        print("Dumping KFolds, do not quit program")
        with open(join("cache", str(n_splits) + ".pickle"), 'wb') as file:
            dump(toDump, file)
        
    elif cachedFolds == False:
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])

        cachedData = None

        with open(join("cache", "stats_X_scaled.pickle"), 'rb') as file:
            cachedData = load(file)

        stats, X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test = cachedData

        yield stats, X_train_scaled, X_test_scaled, y_train, y_test

        if n_splits == 20:
            print(f"Using default n_splits = {20}, please change depending on what problem you are solving.")

        kf = KFold(n_splits=n_splits)

        toDump = []

        i = 0
        for train_index, test_index in kf.split(X_train, y_train):
            X_folded_train = X_train.iloc[train_index]
            y_folded_train = y_train.iloc[train_index]
            X_folded_test = X_train.iloc[test_index]
            y_folded_test = y_train.iloc[test_index]
            stats = getStats(X_folded_train, X_folded_test)

            X_folded_train_scaled = pipeline.fit_transform(X_folded_train)
            X_folded_test_scaled = pipeline.fit_transform(X_folded_test)

            toDump.append((i, stats, (X_folded_train_scaled, X_folded_test_scaled, y_folded_train, y_folded_test)))
            yield i, stats, X_folded_train_scaled, X_folded_test_scaled, y_folded_train, y_folded_test
            i += 1
        
        print("Dumping KFolds, do not quit program")
        with open(join("cache", str(n_splits) + ".pickle"), 'wb') as file:
            dump(toDump, file)
        
    
    else:
        cachedData = None

        with open(join("cache", "stats_X_scaled.pickle"), 'rb') as file:
            cachedData = load(file)

        stats, X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test = cachedData

        yield stats, X_train_scaled, X_test_scaled, y_train, y_test

        if n_splits == 20:
            print(f"Using default n_splits = {20}, please change depending on what problem you are solving.")

        dataDump = None

        with open(join("cache", str(n_splits) + ".pickle"), 'rb') as file:
            dataDump = load(file)

        for fold, stats, dat in dataDump:
            X_folded_train_scaled, X_folded_test_scaled, y_folded_train, y_folded_test = dat
            yield fold, stats, X_folded_train_scaled, X_folded_test_scaled, y_folded_train, y_folded_test
