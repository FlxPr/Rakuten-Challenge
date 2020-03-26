"""
    Python script to submit as a part of the project of ELTP 2020 course.

    This script serves as a template. Please use proper comments and meaningful variable names.
"""

"""
    Group Members:
        (1) Michael Ullah
        (2) Felix Poirier
        (3) Wahl Moritz
"""

"""
    Import necessary packages
"""
import numpy as np
import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_accent(string):
    string = string.replace('á', 'a')
    string = string.replace('â', 'a')

    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')

    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')

    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')

    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')

    string = string.replace('ç', 'c')

    return string


def remove_digits(string):
    result = ''.join([i for i in string if not i.isdigit()])
    return result


def raw_to_tokens(raw_string, spacy_nlp):
    # Write code for lower-casing
    string = raw_string.lower()

    string = normalize_accent(string)

    string = remove_digits(string)

    spacy_tokens = spacy_nlp(string)

    string_tokens = [token.orth_ for token in spacy_tokens if not token.is_punct and not token.is_stop]

    clean_string = " ".join(string_tokens)

    return clean_string


def preprocess(df, nlp):
    df.drop(['description', 'productid', 'imageid'], axis=1, inplace=True)
    df['designation'] = df['designation'].apply(lambda x: raw_to_tokens(x, nlp))
    df_train.fillna('', inplace=True)


def model_decision_tree(X_train, y_train, X_test, y_test):
    parameters = {'max_depth': None,
                  'class_weight': 'balanced',
                  'min_samples_leaf': 1,
                  'criterion': 'gini'}
    clf = DecisionTreeClassifier(**parameters)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    return acc, f1


def model_decision_bagging(X_train, y_train, X_test, y_test):
    base_estimator = DecisionTreeClassifier(max_depth=500, class_weight='balanced')
    parameters = {'base_estimator': base_estimator,
                  'n_estimators': [12],
                  'max_samples': [0.8],
                  'max_features': [1.0]}
    clf = BaggingClassifier(**parameters)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    return acc, f1


def model_decision_rf(X_train, y_train, X_test, y_test):
    parameters = {'max_depth': [500],
                  'n_estimators': [130],
                  'min_samples_leaf': [1],
                  'class_weight': ['balanced'],
                  'max_features': [0.8],
                  'n_jobs': -1}

    clf = RandomForestClassifier(**parameters)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    return acc, f1


def model_decision_gb(X_train, y_train, X_test, y_test):
    parameters = {'learning_rate': [0.12],
                  'n_estimators': [300],
                  'min_samples_leaf': [1],
                  'max_depth': [7],
                  'ccp_alpha': [0.0]}

    clf = GradientBoostingClassifier(**parameters)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    return acc, f1


def model_decision_xgb(X_train, y_train, X_test, y_test):
    parameters = {
        'learning_rate': [0.1],
        'max_depth': [7],
        'n_estimators': [100],
        'objective': ['multi:softmax'],
        'n_jobs': [1]
    }

    clf = XGBClassifier(**parameters)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    return acc, f1


def model_decision_adaboost(X_train, y_train, X_test, y_test):
    base_estimator = DecisionTreeClassifier(max_depth=3, class_weight='balanced')
    parameters = {'n_estimators': 100,
                  'base_estimator': base_estimator,
                  'learning_rate': 1.0}

    clf = AdaBoostClassifier(**parameters)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    return acc, f1


if __name__ == "__main__":
    df_train = pd.read_csv('Data/X_train_update.csv', index_col=0)
    df_test = pd.read_csv('Data/X_test_update.csv', index_col=0)
    y_train = pd.read_csv('Data/Y_train_CVw08PX.csv', index_col=0).iloc[df_train.index].values.ravel()

    nlp_fr = spacy.load("fr_core_news_sm")
    preprocess(df_train, nlp_fr)
    preprocess(df_test, nlp_fr)

    df_train['y'] = y_train
    df_train_sample = df_train.sample(frac=0.001)
    y_train_sample = df_train_sample['y']
    df_train_sample.drop('y', axis=1, inplace=True)

    vectorizer = TfidfVectorizer(
        max_df=0.01,
        max_features=50,
        strip_accents='ascii',
        analyzer='word')

    X_train = vectorizer.fit_transform(df_train_sample.designation.values).todense()
    X_test = vectorizer.transform(df_test.designation.values).todense()

    model_1_acc, model_1_f1 = model_decision_tree(X_train, y_train, X_test, y_test)
    model_2_acc, model_2_f1 = model_decision_bagging(X_train, y_train, X_test, y_test)
    model_3_acc, model_3_f1 = model_decision_rf(X_train, y_train, X_test, y_test)
    model_4_acc, model_4_f1 = model_decision_gb(X_train, y_train, X_test, y_test)
    model_5_acc, model_5_f1 = model_decision_xgb(X_train, y_train, X_test, y_test)
    model_6_acc, model_6_f1 = model_decision_adaboost(X_train, y_train, X_test, y_test)

    # print the results
    print("DecisionTree", model_1_acc, model_1_f1)
    print("TreeBagging", model_2_acc, model_2_f1)
    print("RandomForest", model_3_acc, model_3_f1)
    print("GradientBoosting", model_4_acc, model_4_f1)
    print("XGBoost", model_5_acc, model_5_f1)
    print("AdaBoost", model_6_acc, model_6_f1)
