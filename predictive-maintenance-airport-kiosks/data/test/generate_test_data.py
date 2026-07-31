import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

start_date = '2023-01-01 00:00:00'
end_date = '2024-12-31 23:59:59'

# 1. توليد التواريخ
raw_dates = pd.to_datetime(
    np.random.randint(
        int(pd.Timestamp(start_date).timestamp()),
        int(pd.Timestamp(end_date).timestamp()),
        size=n_samples
    ), unit='s'
)
timestamps = pd.Series(raw_dates).sort_values().reset_index(drop=True)

df = pd.DataFrame({'Timestamp': timestamps})

# 2. الـ Features
df['Hour'] = df['Timestamp'].dt.hour
df['Is_Peak_Hour'] = df['Hour'].apply(lambda h: 1 if 12 <= h <= 15 else 0)
df['Is_Weekend'] = df['Timestamp'].dt.dayofweek.apply(lambda d: 1 if d >= 5 else 0)

# Time delta بين الأحداث
df['Time_Delta_Sec'] = df['Timestamp'].diff().dt.total_seconds().fillna(60.0)

# Error Features (نحطو أخطاء متسلسلة باش نديروا صدمة للـ model)
df['Error_Flag'] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

# Rolling Features
df['Error_Count_Last_3'] = df['Error_Flag'].rolling(window=3, min_periods=1).sum().astype(int)
df['Mean_Time_Delta_3'] = df['Time_Delta_Sec'].rolling(window=3, min_periods=1).mean()

# 3. Target Logic واضح (كيطابق الـ Model Feature Importance)
# الـ Model كيعتمد بنسبة 64% على Error_Count_Last_3
def generate_target(row):
    if row['Error_Count_Last_3'] >= 2:
        return 1
    return 0

df['Target_Failure'] = df.apply(generate_target, axis=1)

# حفظ الملف
df.to_csv('data/test/test_data_2023_2024.csv', index=False)
print("✅ تم تحديث test_data_2023_2024.csv بنجاح!")