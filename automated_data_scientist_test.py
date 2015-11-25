# imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# loads data
input_folder = '/Users/roms/GitHub/Automated-Data-Scientist/Data_sample'
output_folder = ''
X_train = pd.read_pickle(input_folder + '/X_train.pkl')
y_train = pd.read_pickle(input_folder + '/y_train.pkl')
X_test = pd.read_pickle(input_folder + '/X_test.pkl')

# deals with NaN values
nan_replacement = -1
X_train = X_train.fillna(nan_replacement)
X_test = X_test.fillna(nan_replacement)

# detects the type of task to perform
if y_train.nunique() / y_train.shape[0] < 0.1:
	task = 'classification'
else:
	task = 'regression'

# preprocessing
def detect_categorical_variables(X, numerical_types):
	detected = []
	for column in X.columns:
		if X[column].dtype not in numerical_types:
			detected.append(column)
	return detected

numerical_types = ['int32', 'int64', 'float32', 'float64']
categorical_variables = detect_categorical_variables(X_train, numerical_types)
categorical_variables = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_10', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37']

def get_dummies(X1, X2, categorical_variables):
	X = pd.concat((X1[categorical_variables], X2[categorical_variables]))
	X_dummies = pd.get_dummies(X)
	X1 = X1.drop(categorical_variables, axis=1)
	X1 = pd.concat((X1, X_dummies.iloc[:X1.shape[0]]), axis=1)
	X2 = X2.drop(categorical_variables, axis=1)
	X2 = pd.concat((X2, X_dummies.iloc[X1.shape[0]:]), axis=1)
	return X1, X2

X_train, X_test = get_dummies(X_train, X_test, categorical_variables)



# applies models
models = 0




knn = KNeighborsClassifier(n_neighbors=3, algorithm='brute', )
knn.fit(X_train, y_train)
knn.score(X_train.iloc[:200], y_train[:200])






"""
# target code:
import datasciencebot as dsb
dsb = datasciencebot(input_folder)
dsb.load_data()
dsb.preprocess_data()
dsb.models = [..., ..., ...]
dsb.fit()
output = dsb.predict()
output.to_csv()

