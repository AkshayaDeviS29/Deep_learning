import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime,timedelta
import matplotlib.pyplot as plt


import warnings
warnings = warnings.filterwarnings('ignore')
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Load data
df = pd.read_csv('TSLA.csv')
df.head()
print("********************", df)


# Data information
null = df.info()
print("*null************", null)

sum_null = df.isnull().sum()
print("#########sum null#############", sum_null)

shape = df.shape
print("##########shape##########", shape)

describe = df.describe()
print("######discribe#########", describe)

# Reset index and get 'Close' prices
DF = df.reset_index()['Close']
print("############DF##########", DF)

# Filter data for the year 2020 and earlier
df = df[pd.to_datetime(df['Date']).dt.year <= 2020]



# Scale the 'Close' column
data = df['Close'].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
scale_data = scaler.fit_transform(data)
print("##########saclerdata#########", scale_data[:5])




# Prepare training data
X_train = []
Y_train = []
for i in range(60, len(scale_data)):
   X_train.append(scale_data[i-60:i,0])
   Y_train.append(scale_data[i,0])
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(X_train.shape)
print(Y_train.shape)

# Reshape X_train for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
A = X_train.shape
print("####AA####", A)


# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2,
          callbacks=[early_stopping, model_checkpoint])


# Prepare test data for the one month prediction
test = scale_data[-20:]
X_test = test.reshape((1, 20, 1))
predicted = model.predict(X_test)
predicted_prices = []
for _ in range(31):
    predicted_price = model.predict(X_test)
    predicted_prices.append(predicted_price[0,0])
    X_test = np.append(X_test[:,1:,:],predicted_price.reshape((1,1,1)),axis=1)

predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1,1))

current_date = datetime.now()
future_dates = [current_date + timedelta(days=i) for i in range(1, 32)]
future = pd.DataFrame({
    'date':future_dates,
    'predicted_price':predicted_prices.flatten()
})

print("####future prediction####",future)



plt.figure(figsize=(14, 5))
plt.plot(future['date'], future['predicted_price'], label='Predicted Price for Next 30 Days')
plt.title('Stock Market Prediction')
plt.xlabel('Date')
plt.ylabel('Predicted Price')
plt.legend()
plt.show()