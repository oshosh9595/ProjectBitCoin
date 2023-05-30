import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from evaluate_ts import evaluate_ts
from tools import fetch_cosine_values, format_dataset

tf.config.run_functions_eagerly(True)
tf.random.set_seed(101)

feat_dimension = 20
train_size = 250
test_size = 250

# 텐서플로우 매개변수 정의
learning_rate = 0.1
optimizer = tf.keras.optimizers.Adam
n_epochs = 1000

# 관측 행렬 준비(float32)
cos_values = fetch_cosine_values(train_size + test_size + feat_dimension)
minibatch_cos_X, minibatch_cos_y = format_dataset(cos_values, feat_dimension)
train_X = minibatch_cos_X[:train_size, :].astype(np.float32)
train_y = minibatch_cos_y[:train_size].reshape((-1, 1)).astype(np.float32)
test_X = minibatch_cos_X[train_size:, :].astype(np.float32)
test_y = minibatch_cos_y[train_size:].reshape((-1, 1)).astype(np.float32)

# Define the regression model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(feat_dimension,)))

# Define the loss function
loss_fn = tf.keras.losses.MeanSquaredError()

# Create an optimizer
optimizer = optimizer(learning_rate)

# Compile the model
model.compile(optimizer="adam", loss=loss_fn)

# Perform training
mse_values = []
mae_values = []
for epoch in range(n_epochs):
    model.fit(train_X, train_y, batch_size=train_size, verbose=0)
    train_loss = model.evaluate(train_X, train_y, verbose=0)
    test_loss = model.evaluate(test_X, test_y, verbose=0)
    y_pred = model.predict(test_X)
    mse = np.mean((y_pred - test_y) ** 2)
    mae = np.mean(np.abs(y_pred - test_y))
    mse_values.append(mse)
    mae_values.append(mae)
    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - MSE: {mse:.4f} - MAE: {mae:.4f}")

# Evaluate on test dataset
test_loss = model.evaluate(test_X, test_y)
y_pred = model.predict(test_X)

# Evaluate the results
evaluate_ts(test_X, test_y, y_pred)

# Plot the predicted and true values
plt.plot(range(len(cos_values)), cos_values, 'b')
plt.plot(range(len(cos_values) - test_size, len(cos_values)), y_pred, 'r--')
plt.xlabel("Days")
plt.ylabel("Predicted and true values")
plt.title("Predicted (Red) VS Real (Blue)")
plt.show()