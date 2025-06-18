import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

data_path = r'C:\Users\schintapenta\OneDrive - Enova Facilities Management\Desktop\training-set-no-fir-no-worktype.csv'
df = pd.read_csv(data_path)
columns_to_ignore_during_training = ['WO#', 'work-type', 'FIR']
df = df.drop(columns=[col for col in columns_to_ignore_during_training if col in df.columns])
df = df.dropna(subset=['Complexity-Weight'])
df['Description'] = df['Description'].fillna('')
df['Log Details'] = df['Log Details'].fillna('')
text_columns = ['Description', 'Log Details']
categorical_columns = ['Service']
target_column = 'Complexity-Weight'
X = df[text_columns + categorical_columns]
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
original_test_df = df.loc[X_test.index].copy()
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2)))
])
category_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessing = ColumnTransformer([
    ('description_tfidf', text_pipeline, 'Description'),
    ('logdetails_tfidf', text_pipeline, 'Log Details'),
    ('service_onehot', category_pipeline, categorical_columns)
])
model = RandomForestClassifier(random_state=42, class_weight='balanced')
full_pipeline = Pipeline([
    ('preprocessor', preprocessing),
    ('classifier', model)
])
full_pipeline.fit(X_train, y_train)
predicted_values = full_pipeline.predict(X_test)
print(f"Training Accuracy: {accuracy_score(y_train, full_pipeline.predict(X_train)):.3f}")
print(f"Testing Accuracy: {accuracy_score(y_test, predicted_values):.3f}")
print("Predicted values:", predicted_values)
original_test_df['predicted complexity'] = predicted_values

match_status = []
for actual, predicted in zip(original_test_df['Complexity-Weight'], original_test_df['predicted complexity']):
    if actual == predicted:
        match_status.append('matching')
    else:
        match_status.append('not matching')
original_test_df['match_status'] = match_status
output_csv = r'C:\Users\schintapenta\OneDrive - Enova Facilities Management\Desktop\tested-data.csv'
original_test_df.to_csv(output_csv, index=False)

print(f"\nTest data with predictions and match info saved to:\n{output_csv}")