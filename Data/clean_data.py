import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 510

data = pd.read_csv("Data/water_potability.csv")
X = data.drop("Potability", axis=1)
y = data['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 1. Impute and add indicators to ALL features first
# 2. Use ColumnTransformer to scale ONLY the original numeric columns, and 'passthrough' the new indicator columns
preprocessor = Pipeline(steps=[
    ('impute_and_flag', SimpleImputer(strategy='median', add_indicator=True)),
    ('scale_numerics_only', ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), slice(0, len(numeric_features)))
        ],
        remainder='passthrough' # Leaves the binary indicator columns unscaled (0/1)
    ))
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

X_train_processed_df['Potability'] = y_train.values
X_test_processed_df['Potability'] = y_test.values

os.makedirs('Data', exist_ok=True)
X_train_processed_df.to_csv('Data/train.csv', index=False)
X_test_processed_df.to_csv('Data/test.csv', index=False)

print("Successfully preprocessed and exported data to Data/train.csv and Data/test.csv")