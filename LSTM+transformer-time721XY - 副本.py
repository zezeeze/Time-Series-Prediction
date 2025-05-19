import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from hydroeval import nse
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import os

data = pd.read_excel('XY station.xlsx')  
time = data['Time']
features = data.drop('Time', axis=1)


features_clean = features.copy()
for col in features.columns:
    Q1 = features[col].quantile(0.25)
    Q3 = features[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    features_clean[col] = features[col].mask((features[col] < lower_bound) | (features[col] > upper_bound))
features_clean = features_clean.dropna() 
time = time[features_clean.index] 

scalers = {}
features_scaled = pd.DataFrame()
for col in features_clean.columns:
    scaler = MinMaxScaler()
    features_scaled[col] = scaler.fit_transform(features_clean[[col]]).flatten()
    scalers[col] = scaler 

def augment_data(data, noise_level=0.01, scale_range=(0.9, 1.1)):
    augmented_data = data.copy()

    noise = np.random.normal(0, noise_level, augmented_data.shape)
    augmented_data += noise

    scale = np.random.uniform(scale_range[0], scale_range[1], augmented_data.shape[1])
    augmented_data *= scale
    return augmented_data

def create_sequences(data, time_data, input_length, output_length, augment=False):
    X, y, time_sequences = [], [], []
    for i in range(len(data) - input_length - output_length + 1):
        seq = data[i:i + input_length]
        if augment:
            seq = augment_data(seq)
        X.append(seq)
        y.append(data[i + input_length:i + input_length + output_length])
        time_sequences.append(time_data.iloc[i + input_length:i + input_length + output_length])
    return np.array(X), np.array(y), np.array(time_sequences)


input_length = 48
output_length = 48
X, y, time_sequences = create_sequences(features_scaled, time, input_length, output_length, augment=False)

total_size = len(X)
train_size = int(total_size * 0.6)
val_size = int(total_size * 0.2)
test_size = total_size - train_size - val_size

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
time_train, time_val, time_test = time_sequences[:train_size], time_sequences[train_size:train_size+val_size], time_sequences[train_size+val_size:]

train_original_end = (train_size - 1) + input_length + output_length
if train_original_end > len(features_scaled):
    train_original_end = len(features_scaled)
train_features = features_scaled[:train_original_end]
train_time = time[:train_original_end]



X_train_augmented, y_train_augmented, time_train_augmented = create_sequences(train_features, train_time, input_length, output_length, augment=True)


X_train = np.concatenate([X_train, X_train_augmented], axis=0)
y_train = np.concatenate([y_train, y_train_augmented], axis=0)
time_train = np.concatenate([time_train, time_train_augmented], axis=0)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len  
        self.embed_dim = embed_dim  
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)

    def get_angles(self, position, i, embed_dim):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        return position * angles

    def positional_encoding(self, max_len, embed_dim):
        angle_rads = self.get_angles(
            np.arange(max_len)[:, np.newaxis],
            np.arange(embed_dim)[np.newaxis, :],
            embed_dim
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "embed_dim": self.embed_dim,
        })
        return config


embed_dim = features_scaled.shape[1]
num_heads = 2
ff_dim = embed_dim * 4 
rate = 0.2

inputs = Input(shape=(input_length, embed_dim))
x = PositionalEncoding(input_length, embed_dim)(inputs)


lstm_output = LSTM(30, return_sequences=True)(x)
lstm_output = Dropout(rate)(lstm_output) 


lstm_output = Dense(embed_dim)(lstm_output)
residual = x + lstm_output  

transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
transformer_output = transformer_block(residual)


flatten_output = tf.keras.layers.Flatten()(transformer_output)
outputs = Dense(output_length * embed_dim)(flatten_output)
outputs = tf.keras.layers.Reshape((output_length, embed_dim))(outputs)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')


early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)


model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping],
          verbose=1)


train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)


train_pred_rescaled = np.zeros_like(train_pred.reshape(-1, embed_dim))
val_pred_rescaled = np.zeros_like(val_pred.reshape(-1, embed_dim))
test_pred_rescaled = np.zeros_like(test_pred.reshape(-1, embed_dim))
y_train_rescaled = np.zeros_like(y_train.reshape(-1, embed_dim))
y_val_rescaled = np.zeros_like(y_val.reshape(-1, embed_dim))
y_test_rescaled = np.zeros_like(y_test.reshape(-1, embed_dim))

for i, col in enumerate(features_scaled.columns):
    train_pred_rescaled[:, i] = scalers[col].inverse_transform(train_pred.reshape(-1, embed_dim)[:, i].reshape(-1, 1)).flatten()
    val_pred_rescaled[:, i] = scalers[col].inverse_transform(val_pred.reshape(-1, embed_dim)[:, i].reshape(-1, 1)).flatten()
    test_pred_rescaled[:, i] = scalers[col].inverse_transform(test_pred.reshape(-1, embed_dim)[:, i].reshape(-1, 1)).flatten()
    y_train_rescaled[:, i] = scalers[col].inverse_transform(y_train.reshape(-1, embed_dim)[:, i].reshape(-1, 1)).flatten()
    y_val_rescaled[:, i] = scalers[col].inverse_transform(y_val.reshape(-1, embed_dim)[:, i].reshape(-1, 1)).flatten()
    y_test_rescaled[:, i] = scalers[col].inverse_transform(y_test.reshape(-1, embed_dim)[:, i].reshape(-1, 1)).flatten()


def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nse_score = nse(y_pred.flatten(), y_true.flatten())
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, nse_score, r2


column_names = ['WT', 'pH', 'DO', 'EC', 'NTU', 'PPI', 'NH3-N', 'TN', 'TP'] 


def create_metrics_df(y_true, y_pred):
    metrics = {
        'Indicator': ['RMSE', 'MAE', 'NSE', 'R2']
    }
    for col in column_names:
        rmse_col, mae_col, nse_col, r2_col = evaluate_model(y_true[:, features.columns.get_loc(col)],
                                                          y_pred[:, features.columns.get_loc(col)])
        metrics[col] = [rmse_col, mae_col, nse_col, r2_col]
    return pd.DataFrame(metrics).round(3)


train_metrics_df = create_metrics_df(y_train_rescaled, train_pred_rescaled)
val_metrics_df = create_metrics_df(y_val_rescaled, val_pred_rescaled)
test_metrics_df = create_metrics_df(y_test_rescaled, test_pred_rescaled)


result_folder = f'result_{output_length}'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

model.save(os.path.join(result_folder, 'trained_model.h5'))


def save_results_to_excel(y_true, y_pred, time_data, metrics_df, file_name):
    with pd.ExcelWriter(file_name) as writer:
        for i, col in enumerate(column_names):
            true_col = y_true[:, features.columns.get_loc(col)]
            pred_col = y_pred[:, features.columns.get_loc(col)]
            df = pd.DataFrame({
                'Time': time_data.flatten(),
                'True': true_col,
                'Pred': pred_col
            })

            df = df.sort_values(by='时间戳', ascending=False)
            df.to_excel(writer, sheet_name=col, index=False)
        metrics_df.to_excel(writer, sheet_name='Indicator', index=False)


train_file_name = os.path.join(result_folder, f'train_results_{input_length}+{output_length}.xlsx')
val_file_name = os.path.join(result_folder, f'val_results_{input_length}+{output_length}.xlsx')
test_file_name = os.path.join(result_folder, f'test_results_{input_length}+{output_length}.xlsx')


save_results_to_excel(y_train_rescaled, train_pred_rescaled, time_train, train_metrics_df, train_file_name)
save_results_to_excel(y_val_rescaled, val_pred_rescaled, time_val, val_metrics_df, val_file_name)
save_results_to_excel(y_test_rescaled, test_pred_rescaled, time_test, test_metrics_df, test_file_name)
