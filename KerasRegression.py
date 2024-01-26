import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from keras import layers
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import joblib


# Create a ModelCheckpoint callback
model_checkpoint = ModelCheckpoint(
    'SAVED_MODELS\Kerasbest_model.h5',
    monitor='loss',  # Monitor validation loss
    save_best_only=True,  # Save only the best model
    mode='min',  # Save the model when MAE is minimized
    verbose=1
)

#training model function
def trainmodel(model, X_train, y_train, X_test, y_test, epochs_in=2000, batch_size_in=500):
    
    # define model training properties
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics = ['accuracy'])

    # Check if a GPU is available
    if tf.test.is_gpu_available(cuda_only=True):
        print("GPU is available. Using GPU(s) for training.")
        with tf.device('/device:GPU:0'): #run the code using GPU
            history = model.fit(X_train, y_train, epochs=epochs_in, batch_size=batch_size_in, validation_data=(X_test, y_test),
            callbacks=[model_checkpoint])
            model.summary()
    else:
        print("GPU is not available. Using CPU for training.")

    model = LinearRegression()

    #Plotting learning history
    plt.title('Deep learning training with 100 epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(history.history['loss'])
    plt.show()

#------------Main----------------

# Load data from a CSV file
data = pd.read_csv('priceChange.csv') 

# Extract independent and dependent variables from input data
X = data.drop(columns=['new_agreement_price'])
y = data['new_agreement_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#create a sequential model
model = keras.Sequential()

#prints input shape
print(X.shape[1],)

# Add an Input layer with X input units (nodes)
model.add(layers.Input(shape=(X.shape[1],)))

# Add 7 Hidden layers
model.add(layers.Dense(25, activation='relu'))  # Layer 1 with 25 nodes and relu activation
model.add(layers.Dense(60, activation='relu'))  # Layer 2 with 60 nodes and relu activation
model.add(layers.Dense(100, activation='relu')) # Layer 3 with 100 nodes and relu activation
model.add(layers.Dense(200, activation='relu')) # Layer 3 with 200 nodes and relu activation
model.add(layers.Dense(100, activation='relu')) # Layer 3 with 100 nodes and relu activation
model.add(layers.Dense(60, activation='relu')) # Layer 3 with 60 nodes and relu activation
model.add(layers.Dense(25, activation='relu')) # Layer 3 with 25 nodes and relu activation

#Add an Output Layer with 1 node
model.add(layers.Dense(1, activation='linear'))

#trains model
trainmodel(model, X_train, y_train, X_test, y_test)

#saves scaler
joblib.dump(scaler, 'SAVED_MODELS\Kerasbest_model.joblib')

# Evaluate the model
loss = model.evaluate(X_test, y_test)

# Make predictions on the test data
y_pred = model.predict(X_test)

#calculate and print mean squeared error
mse_lin = mean_squared_error(y_test, y_pred)
print('mean_squared_error : ', mse_lin)

# Calculate and print R-squared (R^2) score
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R^2) Score: {r2}")

# Calculate and print Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")