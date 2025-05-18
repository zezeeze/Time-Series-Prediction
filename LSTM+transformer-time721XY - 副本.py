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

# 读取数据
data = pd.read_excel('XY station.xlsx')  # 修改文件名
time = data['Time']
features = data.drop('Time', axis=1)

# 异常值处理：使用四分位距法（按特征独立处理）
features_clean = features.copy()
for col in features.columns:
    Q1 = features[col].quantile(0.25)
    Q3 = features[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    features_clean[col] = features[col].mask((features[col] < lower_bound) | (features[col] > upper_bound))
features_clean = features_clean.dropna()  # 删除包含任何异常值的行
time = time[features_clean.index]  # 同步处理时间数据

# 针对每个特征独立归一化
scalers = {}
features_scaled = pd.DataFrame()
for col in features_clean.columns:
    scaler = MinMaxScaler()
    features_scaled[col] = scaler.fit_transform(features_clean[[col]]).flatten()
    scalers[col] = scaler  # 保存每个特征的归一化器

# 数据增强函数
def augment_data(data, noise_level=0.01, scale_range=(0.9, 1.1)):
    augmented_data = data.copy()
    # 添加随机噪声
    noise = np.random.normal(0, noise_level, augmented_data.shape)
    augmented_data += noise
    # 随机缩放
    scale = np.random.uniform(scale_range[0], scale_range[1], augmented_data.shape[1])
    augmented_data *= scale
    return augmented_data

# 数据预处理
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

# 按时间顺序分割数据
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


# 仅在训练集对应的原始数据上生成增强序列
X_train_augmented, y_train_augmented, time_train_augmented = create_sequences(train_features, train_time, input_length, output_length, augment=True)

# 合并增强数据到训练集
X_train = np.concatenate([X_train, X_train_augmented], axis=0)
y_train = np.concatenate([y_train, y_train_augmented], axis=0)
time_train = np.concatenate([time_train, time_train_augmented], axis=0)

# 定义Transformer层
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
        self.max_len = max_len  # 保存 max_len 参数
        self.embed_dim = embed_dim  # 保存 embed_dim 参数
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

# 构建LSTM + Transformer模型
embed_dim = features_scaled.shape[1]
num_heads = 2
ff_dim = embed_dim * 4  # 动态调整隐藏层维度为输入维度的4倍
rate = 0.2

inputs = Input(shape=(input_length, embed_dim))
x = PositionalEncoding(input_length, embed_dim)(inputs)

# LSTM层，添加Dropout正则化
lstm_output = LSTM(30, return_sequences=True)(x)
lstm_output = Dropout(rate)(lstm_output)  # 添加Dropout层

# 调整LSTM输出维度，并添加残差连接
lstm_output = Dense(embed_dim)(lstm_output)
residual = x + lstm_output  # 残差连接

# Transformer模块
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
transformer_output = transformer_block(residual)

# 展平并输出
flatten_output = tf.keras.layers.Flatten()(transformer_output)
outputs = Dense(output_length * embed_dim)(flatten_output)
outputs = tf.keras.layers.Reshape((output_length, embed_dim))(outputs)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# 早停策略
early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping],
          verbose=1)

# 预测
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

# 反归一化
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


# 评价指标
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nse_score = nse(y_pred.flatten(), y_true.flatten())
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, nse_score, r2


# 构建评价指标数据框
column_names = ['WT', 'pH', 'DO', 'EC', 'NTU', 'PPI', 'NH3-N', 'TN', 'TP']  # 修改列名


def create_metrics_df(y_true, y_pred):
    metrics = {
        '指标': ['RMSE', 'MAE', 'NSE', 'R2']
    }
    for col in column_names:
        rmse_col, mae_col, nse_col, r2_col = evaluate_model(y_true[:, features.columns.get_loc(col)],
                                                          y_pred[:, features.columns.get_loc(col)])
        metrics[col] = [rmse_col, mae_col, nse_col, r2_col]
    return pd.DataFrame(metrics).round(3)


train_metrics_df = create_metrics_df(y_train_rescaled, train_pred_rescaled)
val_metrics_df = create_metrics_df(y_val_rescaled, val_pred_rescaled)
test_metrics_df = create_metrics_df(y_test_rescaled, test_pred_rescaled)




# 创建保存结果的文件夹
result_folder = f'result_{output_length}'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)


# 导入SHAP库
import shap
import matplotlib.pyplot as plt

# ---------------------------
# SHAP分析部分
# ---------------------------
# 初始化SHAP解释器（使用训练数据作为背景）
explainer = shap.DeepExplainer(model, X_train[:100])  # 使用前100个训练样本作为参考

# 计算测试集的SHAP值（为节省时间可采样部分数据）
shap_values = explainer.shap_values(X_test[:200])  # 取前200个测试样本

# 计算每个特征在所有时间步和样本上的平均SHAP绝对值
# shap_values形状: (样本数, 时间步, 特征数)
average_shap = np.mean(np.abs(shap_values), axis=(0, 1))  # 对样本和时间步求平均

# 构建特征重要性数据
feature_importance = pd.DataFrame({
    '特征': features.columns,
    'SHAP重要性': average_shap
}).sort_values(by='SHAP重要性', ascending=False)

# 绘制特征重要性柱状图
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['特征'], feature_importance['SHAP重要性'], color='skyblue')
plt.title('特征重要性（SHAP值）', fontsize=14)
plt.xlabel('特征', fontsize=12)
plt.ylabel('平均SHAP绝对值', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 保存图片
shap_plot_path = os.path.join(result_folder, 'shap_feature_importance.png')
plt.savefig(shap_plot_path, dpi=300)
print(f"SHAP特征重要性图已保存至: {shap_plot_path}")
plt.close()

# 保存SHAP分析结果到Excel
shap_metrics_df = feature_importance.reset_index(drop=True)
shap_file_name = os.path.join(result_folder, 'shap_analysis.xlsx')

with pd.ExcelWriter(shap_file_name) as writer:
    shap_metrics_df.to_excel(writer, sheet_name='特征重要性', index=False)
print(f"SHAP分析结果已保存至: {shap_file_name}")

# 保存模型
model.save(os.path.join(result_folder, 'trained_model.h5'))


# 保存训练集，验证集和测试集的结果到Excel文件
def save_results_to_excel(y_true, y_pred, time_data, metrics_df, file_name):
    with pd.ExcelWriter(file_name) as writer:
        for i, col in enumerate(column_names):
            true_col = y_true[:, features.columns.get_loc(col)]
            pred_col = y_pred[:, features.columns.get_loc(col)]
            df = pd.DataFrame({
                '时间戳': time_data.flatten(),
                'True': true_col,
                'Pred': pred_col
            })
            # 按照时间戳降序排列
            df = df.sort_values(by='时间戳', ascending=False)
            df.to_excel(writer, sheet_name=col, index=False)
        metrics_df.to_excel(writer, sheet_name='评价指标', index=False)


print("训练集时间戳范围：", time_train.min(), "到", time_train.max())
print("验证集时间戳范围：", time_val.min(), "到", time_val.max())
print("测试集时间戳范围：", time_test.min(), "到", time_test.max())

train_file_name = os.path.join(result_folder, f'train_results_{input_length}+{output_length}.xlsx')
val_file_name = os.path.join(result_folder, f'val_results_{input_length}+{output_length}.xlsx')
test_file_name = os.path.join(result_folder, f'test_results_{input_length}+{output_length}.xlsx')


save_results_to_excel(y_train_rescaled, train_pred_rescaled, time_train, train_metrics_df, train_file_name)
save_results_to_excel(y_val_rescaled, val_pred_rescaled, time_val, val_metrics_df, val_file_name)
save_results_to_excel(y_test_rescaled, test_pred_rescaled, time_test, test_metrics_df, test_file_name)
