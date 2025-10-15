import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
import seaborn as sns
import xgboost as xgb
SEED = 123

#Cleaning, input, and output
class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    #Read data
    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    #Drop column target
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

    #Check NA
    def check_missing_values(self):
        return self.data.isnull().sum()

    #Check duplicated
    def check_duplicate_rows(self):
        return self.data.duplicated().sum()

    #Check structure from data
    def check_structure(self):
        return self.data.info()

    #Check total unique value
    def check_unique_values(self):
        return self.data.nunique()

    #Check distribution from numeric columns
    def check_outlier(self):
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            plt.figure(figsize=(5, 3))
            self.data.boxplot(column=[col])
            plt.title(f"Boxplot for {col}")
            plt.show()

    #Visualization from categorical data
    def check_categorical_columns(self, columns):
        for col in columns:
            plt.figure(figsize=(12, 5))

            #Barplot for distribution
            value_counts = self.data[col].value_counts()
            sns.barplot(
                x=value_counts.index,
                y=value_counts.values,
                hue=value_counts.index,
                dodge=False,
                legend=False,
                palette="viridis"
            )
            plt.title(f'Distribution of {col}')
            plt.ylabel('Frequency')
            plt.xlabel(col)
            plt.xticks(rotation=45)
            plt.show()

            #Stats
            print(f"Column: {col}")
            print(value_counts)
            print(f"Unique Categories: {self.data[col].nunique()}")
            print('-' * 80)

    #Fill NA
    def fillingNAWithNumbers(self, columns, number):
        self.data[columns].fillna(number, inplace=True)

    #Search median
    def createMedianFromColumn(self, kolom):
        numeric_col = pd.to_numeric(self.data[kolom], errors='coerce')
        return np.nanmedian(self.data[kolom])

class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel() 
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    #Replace incorrect or inconsistent values in categorical columns
    def correct_categorical_values(self, column, corrections):
        self.input_data[column] = self.input_data[column].replace(corrections)

    #Visualization from categorical data
    def check_categorical_columns(self, columns):
        for col in columns:
            plt.figure(figsize=(12, 5))

            #Barplot for distribution
            value_counts = self.input_data[col].value_counts()
            sns.barplot(
                x=value_counts.index,
                y=value_counts.values,
                hue=value_counts.index,
                dodge=False,
                legend=False,
                palette="viridis"
            )
            plt.title(f'Distribution of {col}')
            plt.ylabel('Frequency')
            plt.xlabel(col)
            plt.xticks(rotation=45)
            plt.show()

            #Stats
            print(f"Column: {col}")
            print(value_counts)
            print(f"Unique Categories: {self.input_data[col].nunique()}")
            print('-' * 80)

    #Split Data
    def split_data(self, test_size = 0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.input_data, self.output_data, test_size=test_size, random_state=random_state)
        print(self.x_train.shape,self.x_test.shape), print(self.y_train.shape,self.y_test.shape)
        print(self.y_train.value_counts())

    def preprocess_data(self, ohe_cols, label_cols, rob_cols, ord_cols):
        self.label_encoders = {}

        for col in label_cols:
          le = LabelEncoder()
          self.x_train[col] = le.fit_transform(self.x_train[col])
          self.x_test[col] = le.transform(self.x_test[col])
          self.label_encoders[col] = le 

        ordinal_cols = list(ord_cols.keys())
        ordinal_categories = list(ord_cols.values())

        self.transformer = ColumnTransformer(
            transformers=[
                ('robust', RobustScaler(), rob_cols),
                ('ohe', OneHotEncoder(handle_unknown='ignore'), ohe_cols),
                ('ordinal', OrdinalEncoder(categories=ordinal_categories), ordinal_cols)
            ]
        )

        #Fit on training, transform both train and test
        self.x_train = self.transformer.fit_transform(self.x_train)
        self.x_test = self.transformer.transform(self.x_test)

        #Get feature names
        ohe_feature_names = self.transformer.named_transformers_['ohe'].get_feature_names_out(ohe_cols)
        self.feature_names = label_cols + rob_cols + list(ohe_feature_names) + ordinal_cols

        print("Preprocessing complete. Transformed feature names:")
        print(self.feature_names)


    #Modeling
    def createModel(self, n_estimators=100, learning_rate=0.01, max_depth=None):
        self.model = xgb.XGBClassifier(
            random_state=SEED,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test)
        self.y_predict_proba = self.model.predict_proba(self.x_test)
        self.y_train_predict = self.model.predict(self.x_train)

    def get_prediction_df(self):
        class_labels = self.model.classes_
        probabilities_df = pd.DataFrame(self.y_predict_proba, columns=class_labels)

        probabilities_df = probabilities_df.round(4)

        return probabilities_df

    #Evaluation
    def createReport(self):
        print("\n Classification Report Train:\n")
        print(classification_report(self.y_train, self.y_train_predict, target_names=["0", "1"]))

        print("\n Classification Report Test:\n")
        print(classification_report(self.y_test, self.y_predict, target_names=["0", "1"]))

    def evaluate_model(self):
        return accuracy_score(self.y_test, self.y_predict)

    #Gridsearch
    def tuningParameter(self):
        parameters = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [None, 1, 2, 3]
        }


        grid_search = GridSearchCV(estimator=self.model, param_grid=parameters, cv=5, scoring='accuracy', verbose=3, n_jobs=-1)

        grid_search.fit(self.x_train, self.y_train)

        print("Tuned HyperParameters ", grid_search.best_params_)
        print("Accuracy: ", grid_search.best_score_)

        self.createModel(n_estimators=grid_search.best_params_['n_estimators'], learning_rate=grid_search.best_params_['learning_rate'], max_depth=grid_search.best_params_['max_depth'])

    #Save model
    def save_model_to_file(self, filename): 
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

    #Save transformer
    def save_transformer_to_file(self, filename):
      with open(filename, 'wb') as file:
          pickle.dump(self.transformer, file)

    #Save label encoder
    def save_label_encoders(self, filename):
      with open(filename, 'wb') as file:
          pickle.dump(self.label_encoders, file)

#1. Load data
file_path = 'UTS MD/Dataset_A_loan.csv'
data_handler = DataHandler(file_path)
data_handler.load_data()

#2. Check missing values
data_handler.check_missing_values()

#3. Outlier check
data_handler.check_outlier()

#4. Impute with median because skewed data
person_income_replace = data_handler.createMedianFromColumn('person_income')
data_handler.fillingNAWithNumbers('person_income', person_income_replace)

#5. Check after impute
data_handler.check_missing_values()

#7. Check duplicates rows
data_handler.check_duplicate_rows()

#8. Count unique data
data_handler.check_unique_values()

#9. Check categorical distribution
data_handler.check_categorical_columns(['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file', 'loan_status'])

#10. Split input and output
data_handler.check_structure()
data_handler.create_input_output("loan_status")

#11. Split
input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)

#12. Replace inconsistent data
corrections = {
    'Male': 'male',
    'fe male': 'female'
}

model_handler.correct_categorical_values('person_gender', corrections)

#Check after replace
model_handler.check_categorical_columns(['person_gender'])

model_handler.split_data()

#Before encode
model_handler.x_train.head()

#13. Encode :
label = ['person_gender','previous_loan_defaults_on_file']

ord = {
    "person_home_ownership": ['OTHER','RENT', 'MORTGAGE', 'OWN'],
    "person_education": ['High School','Associate','Bachelor', 'Master', 'Doctorate']
    }

ohe = ["loan_intent"]

rob = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']

model_handler.preprocess_data(ohe, label, rob, ord)

#Check after encode
model_handler.x_train

#Train and predict 
print("Before Tuning Parameter")
model_handler.train_model()
model_handler.makePrediction()

#Evaluation
print("Model Accuracy: ", model_handler.evaluate_model())
model_handler.createReport()

#Tuning
print("After Tuning Parameter")
model_handler.tuningParameter()
model_handler.train_model()
model_handler.makePrediction()

#Evaluation
print("\nModel Accuracy After Tuning: ", model_handler.evaluate_model())
model_handler.createReport()

#Save
model_handler.save_model_to_file("best_model.pkl")
model_handler.save_transformer_to_file("transformer.pkl")
model_handler.save_label_encoders("label_encoders.pkl")
