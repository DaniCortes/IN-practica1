import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
url = "datasets/heart.csv"
data = pd.read_csv(url)

# Split features and target variable
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

# Define numeric and categorical features
numeric_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
categorical_features = ["Sex", "ChestPainType",
                        "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]

# Create preprocessor
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define classifiers
classifiers = [
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC()),
    ('K-Nearest Neighbors', KNeighborsClassifier())
]

# Perform 5-fold cross-validation for each classifier
for name, classifier in classifiers:
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', classifier)])

    # Use cross_val_score for cross-validation
    scores = cross_val_score(model, X, y, cv=5)

    # Print the results
    print(f"\nClassifier: {name}")
    print(
        f"Cross-Validated Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
