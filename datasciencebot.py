import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV


class DataScienceBot:
    """
    A pure-python data scientist :-)
    """
    def __init__(self, categorical_variables=None):
        self.task = None
        self.categorical_variables = categorical_variables
        # data
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        # invariant parameters
        self.numerical_types = ['int32', 'int64', 'float32', 'float64']
        self.fitted_models = []
        self.predictions = []
        self.regression_models = [(LinearRegression, {})]
        self.classification_models =\
            [
            (RandomForestClassifier, {'n_estimators':[10, 50, 100, 250], 'criterion':['gini', 'entropy'],
                                      'max_features':['auto'],  'max_depth':[2, 3, 4, 5, 6, 7, 8],
                                      'n_jobs':[1], 'random_state':[42], 'class_weight':['balanced']})
            ]

    def load_data(self, input_folder):
        """
        Loads pickled data from input folder
        """
        self.X_train = pd.read_pickle(input_folder + '/X_train.pkl')
        self.y_train = pd.read_pickle(input_folder + '/y_train.pkl')
        self.X_test = pd.read_pickle(input_folder + '/X_test.pkl')
        # detects the type of task to perform
        if self.y_train.nunique() / self.y_train.shape[0] < 0.1:
            self.task = 'classification'
        else:
            self.task = 'regression'

    def preprocess_data(self, nan_replacement=None):
        """
        Pre-process the data
        """
        # detects categorical variables
        if self.categorical_variables is None:
            variables = []
            for column in self.X_train.columns:
                if self.X_train[column].dtype not in self.numerical_types:
                    variables.append(column)
            self.categorical_variables = variables

        # fills NaN values
        self.X_train = self.X_train.fillna(nan_replacement)
        self.X_test = self.X_test.fillna(nan_replacement)

        # replaces them by dummy ones
        x = pd.concat((self.X_train[self.categorical_variables], self.X_test[self.categorical_variables]))
        x_dummies = pd.get_dummies(x)
        self.X_train = self.X_train.drop(self.categorical_variables, axis=1)
        n_cols = self.X_train.shape[1]
        self.X_train = pd.concat((self.X_train, x_dummies.iloc[:self.X_train.shape[0]]), axis=1)
        self.X_test = self.X_test.drop(self.categorical_variables, axis=1)
        self.X_test = pd.concat((self.X_test, x_dummies.iloc[self.X_train.shape[0]:]), axis=1)
        self.categorical_variables = self.X_train.columns[n_cols:].tolist()

    def split_train(self, train_size=0.75, seed=42):
        """
        Splits the train set into a train and validation sets
        """
        n_examples = self.X_train.shape[0]
        n_train_examples = int(train_size * n_examples)
        np.random.seed(seed)
        train_idx = np.random.choice(list(range(n_examples)), n_train_examples, replace=False).tolist()
        valid_idx = list(set(range(n_examples)).difference(set(train_idx)))
        self.X_valid = self.X_train.iloc[valid_idx, :].copy()
        self.X_train = self.X_train.iloc[train_idx, :]
        self.y_valid = self.y_train.iloc[valid_idx]
        self.y_train = self.y_train.iloc[train_idx]

    def fit(self):
        """
        Fit various models to the training set
        """
        self.fitted_models = []
        if self.task == 'classification':
            for (model, parameters) in self.classification_models:
                clf = GridSearchCV(model(), parameters, n_jobs=1, cv=2, verbose=10)
                clf.fit(self.X_train, self.y_train)
                self.fitted_models.append(clf)

    def predict(self, X):
        """
        Makes predictions on the test set
        """
        self.predictions = []
        for model in self.fitted_models:
            self.predictions.append(model.predict(X))


if __name__ == '__main__':
    cat_variables = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6',
                     'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5',
                     'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5',
                     'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2',
                     'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8',
                     'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3',
                     'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7',
                     'Medical_History_8', 'Medical_History_9', 'Medical_History_10', 'Medical_History_11',
                     'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16',
                     'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20',
                     'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25',
                     'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29',
                     'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34',
                     'Medical_History_35', 'Medical_History_36', 'Medical_History_37']
    dsb = DataScienceBot(categorical_variables=cat_variables)
    dsb.load_data(input_folder='/Users/roms/GitHub/Automated-Data-Scientist/Data_sample')
    print(dsb.categorical_variables)
    dsb.preprocess_data(nan_replacement=-1)
    print(dsb.categorical_variables)
    dsb.split_train(train_size=.75)
    dsb.fit()
    dsb.predict(dsb.X_test)
    dsb.predictions[0].shape
    pd.DataFrame(dsb.predictions[0], index=dsb.X_test.Id, columns=['Response']).to_csv('submission.csv')
    #predictions.to_csv('predictions.csv')

"""
TO DO: prendre en compte l'Id au chargement du fichier!!
"""
