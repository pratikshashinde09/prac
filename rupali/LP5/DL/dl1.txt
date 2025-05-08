import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('BostonHousing.csv')
print(df.head(n=10))  # Added print to see the output

# Data cleaning
df.dropna(inplace=True)  # Remove rows with missing values
print(df.isnull().sum())  # Added print to see the output
print(df.info())  # Added print to see the output
print(df.describe())  # Added print to see the output

# Check correlation with target variable (use correct column name)
# First check which column name exists in your dataset ('medv' or 'MEDV')
target_col = 'medv' if 'medv' in df.columns else 'MEDV'
print(df.corr()[target_col].sort_values())

# Prepare features and target
X = df.loc[:, df.columns != target_col].values
y = df.loc[:, df.columns == target_col].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Added missing split

# Feature scaling
scaler = StandardScaler()
scaler.fit(X_train)  # Fit only on training data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Model definition - removed duplicate model definition
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu', name='dense_1'),
    Dense(64, activation='relu', name='dense_2'),
    Dense(1, activation='linear', name='dense_output')
])
model.summary()

# Compile the model (added missing compilation step)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.05, verbose=1)

# Evaluation
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
mse_nn, mae_nn = model.evaluate(X_test, y_test, verbose=0)  # Added verbose=0 for cleaner output
r2 = r2_score(y_test, y_pred)

print('Mean squared error on test data: ', mse_nn)
print('Mean absolute error on test data: ', mae_nn)
print('R-squared score:', r2*100)

# Plotting
plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, c='green')
p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()