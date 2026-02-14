import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def load_data(path: str):
    return pd.read_csv(path)

def split_data(df, target_col='class', test_size=0.2, random_state=42):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def get_preprocessor(X_train):
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
    numerical_cols = X_train.select_dtypes(include='int64').columns.tolist()
    categorical_cols.remove('education')

    # Pipelines
    num_pipeline = Pipeline([('scaler', StandardScaler())])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    education_order = [
        "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
        "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters",
        "Prof-school", "Doctorate"
    ]
    edu_pipeline = Pipeline([('ordinalenc', OrdinalEncoder(categories=[education_order]))])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols),
        ('edu', edu_pipeline, ['education'])
    ])
    return preprocessor
