import pandas as pd

df = pd.read_csv('/content/parkinsons.csv')
print(df.head())

input_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)']

output_feature = 'status'

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[input_features] = scaler.fit_transform(df[input_features])
print(df.head())


from sklearn.model_selection import train_test_split


X = df[input_features]
y = df[output_feature]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) 

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

if accuracy < 0.8:
    print("Accuracy below 0.8, retraining with different parameters")
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Retrained Validation Accuracy: {accuracy}")

while accuracy < 0.8:
    print("Accuracy still below 0.8, trying again...")
 
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Retrained Validation Accuracy: {accuracy}")

if accuracy >= 0.8:
        print("Accuracy reached 0.8.")





