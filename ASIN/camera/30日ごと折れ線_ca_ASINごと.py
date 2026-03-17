import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

# ==========================================
# 0. 表示・フォントの設定
# ==========================================
plt.rcParams["font.family"] = "MS Gothic"
font_size_global = 12
plt.rcParams.update({'font.size': font_size_global, 'axes.titlesize': 14, 'axes.labelsize': 12})

# ==========================================
# 1. 設定項目
# ==========================================
base_dir = r'C:\Users\admin\Desktop\4年\本実験\ASIN\camera'
spec_file_name = 'ASIN+spec.xlsx'
trends_file_name = 'trends_2023_2025.csv' 
trends_2012_file = 'trends_ca_2012.csv' 

target_lags = [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] 
lag_vars = [f'ln_y_lag{i}' for i in target_lags]

asins = ["B00IDV4ACC", "B00S7LBA08", "B00XXCOJLS", "B0B1LVFFB5", "B01ALAZOOK",
         "B01FH4H5LA", "B01LXIND6Y", "B07FBJ9WNB", "B07G28MFV5", "B07GZGNTXZ",
         "B07JNL93SQ", "B07NVXM29C", "B07SLXZCBW", "B07V6DHHRX", "B07VYHDDF9",
         "B08N4FKFCG", "B09NDF5BN7", "B09ZRN1N3Z", "B072J97HT1", "B083K3SHCK",
         "B097C55DCP", "B099RKG1FN", "B007410K80"]

inter_vars_m1 = ['inter_res_season', 'inter_vol_season', 'inter_pri_season', 'inter_zoo_season']
inter_vars_m2 = ['search_2012_impact']
category_var = ['ln_z_category']

target_vars = lag_vars + category_var + inter_vars_m1 + inter_vars_m2

# ==========================================
# 2. データの読み込みと前処理
# ==========================================
trends_path = os.path.join(base_dir, trends_file_name)
trends_raw = pd.read_csv(trends_path, encoding='cp932')
trends_df = pd.DataFrame({
    'date': pd.to_datetime(trends_raw.iloc[:, 0]),
    'z_res': trends_raw.iloc[:, 1],
    'z_vol': trends_raw.iloc[:, 2],
    'z_pri': trends_raw.iloc[:, 3],
    'z_zoo': trends_raw.iloc[:, 4],
    'z_cat': trends_raw.iloc[:, 6]
})

trends_2012_path = os.path.join(base_dir, trends_2012_file)
t12_raw = pd.read_csv(trends_2012_path, encoding='cp932')
trends_12_df = pd.DataFrame({
    'date': pd.to_datetime(t12_raw.iloc[:, 0]),
    'search_b': t12_raw.iloc[:, 1], 
    'seasonal_d': t12_raw.iloc[:, 3]
})

spec_file_path = os.path.join(base_dir, spec_file_name)
specs_master = pd.read_excel(spec_file_path, usecols=[0, 2, 3, 4, 5])
specs_master.columns = ['asin', 'f_res', 'f_vol', 'f_pri', 'f_zoo']

all_data = []
eps = 1e-6

for asin in asins:
    file_path = os.path.join(base_dir, f'salesrank_daily_2023_2025_{asin}.csv')
    if not os.path.exists(file_path): continue
    
    df_p = pd.read_csv(file_path)
    df_p['date'] = pd.to_datetime(df_p['date'])
    
    df = pd.merge(df_p, trends_df, on='date', how='left')
    df = pd.merge(df, trends_12_df, on='date', how='left')
    df = df.sort_values('date')
    df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2025-12-31')]
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method='linear', limit_direction='both')
    
    spec_match = specs_master[specs_master['asin'] == asin]
    if spec_match.empty: continue
    spec = spec_match.iloc[0]
    
    k_c, A_c = 0.713, 7.831
    df['q_it'] = A_c * (df['daily_salesrank'].clip(lower=1) ** (-k_c))
    df['ln_y'] = np.log(df['q_it'] + eps) 

    new_cols = {}
    for i in target_lags:
        new_cols[f'ln_y_lag{i}'] = df['ln_y'].shift(i)
    
    ln_cat = np.log(df['z_cat'] + eps)
    new_cols['ln_z_category'] = ln_cat
    
    spec_map = {'res': 'f_res', 'vol': 'f_vol', 'pri': 'f_pri', 'zoo': 'f_zoo'}
    trend_map = {'res': 'z_res', 'vol': 'z_vol', 'pri': 'z_pri', 'zoo': 'z_zoo'}
    for key in ['res', 'vol', 'pri', 'zoo']:
        f_val = pd.to_numeric(spec[spec_map[key]], errors='coerce')
        ln_f_val = np.log(f_val + eps)
        ln_z_val = np.log(df[trend_map[key]] + eps)
        new_cols[f'inter_{key}_season'] = ln_f_val * ln_z_val * ln_cat

    new_cols['search_2012_impact'] = np.log(df['search_b'] + eps) * np.log(df['seasonal_d'] + eps)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df['asin'] = asin 
    all_data.append(df)

full_df = pd.concat(all_data).dropna().sort_values('date')
asin_dummies = pd.get_dummies(full_df['asin'], drop_first=True).astype(float)
full_df = pd.concat([full_df, asin_dummies], axis=1)

# ==========================================
# 3. 特徴量の正規化（標準化）
# ==========================================
train_mask = full_df['date'].dt.year == 2023
test_mask = full_df['date'].dt.year >= 2024

scaler = StandardScaler()
full_df.loc[train_mask, target_vars] = scaler.fit_transform(full_df.loc[train_mask, target_vars])
full_df.loc[test_mask, target_vars] = scaler.transform(full_df.loc[test_mask, target_vars])

# ==========================================
# 4. 学習
# ==========================================
train_data = full_df[train_mask]
test_overall = full_df[test_mask].copy()
dummy_cols = asin_dummies.columns.tolist()

X_m0_cols = lag_vars + category_var + dummy_cols
X_m1_cols = lag_vars + category_var + inter_vars_m1 + dummy_cols
X_m2_cols = lag_vars + category_var + inter_vars_m2 + dummy_cols

model_m0 = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[X_m0_cols])).fit()
model_m1 = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[X_m1_cols])).fit()
model_m2 = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[X_m2_cols])).fit()

# ==========================================
# 5. ASINごとのMAE算出と表示
# ==========================================
asin_mae_results = []
unique_test_asins = test_overall['asin'].unique()

for asin in unique_test_asins:
    asin_subset = test_overall[test_overall['asin'] == asin]
    
    def get_pred_subset(model, cols, data):
        exog = sm.add_constant(data[cols], has_constant='add')[model.params.index]
        return model.predict(exog)

    p0 = get_pred_subset(model_m0, X_m0_cols, asin_subset)
    p1 = get_pred_subset(model_m1, X_m1_cols, asin_subset)
    p2 = get_pred_subset(model_m2, X_m2_cols, asin_subset)
    
    mae0 = mean_absolute_error(asin_subset['ln_y'], p0)
    mae1 = mean_absolute_error(asin_subset['ln_y'], p1)
    mae2 = mean_absolute_error(asin_subset['ln_y'], p2)
    
    asin_mae_results.append({
        'ASIN': asin,
        'MAE_M0': mae0, 'MAE_M1': mae1, 'MAE_M2': mae2,
        'M2改善率(%)': (1 - mae2/mae0)*100
    })

asin_mae_df = pd.DataFrame(asin_mae_results).sort_values('ASIN')

print("\n" + "="*80)
print(f"{'ASIN':<15} | {'MAE_M0':>8} | {'MAE_M1':>8} | {'MAE_M2':>8} | {'M2改善率':>8}")
print("-" * 80)
for _, row in asin_mae_df.iterrows():
    print(f"{row['ASIN']:<15} | {row['MAE_M0']:8.4f} | {row['MAE_M1']:8.4f} | {row['MAE_M2']:8.4f} | {row['M2改善率(%)']:7.2f}%")
print("-" * 80)
print(f"{'全ASIN平均':<15} | {asin_mae_df['MAE_M0'].mean():8.4f} | {asin_mae_df['MAE_M1'].mean():8.4f} | {asin_mae_df['MAE_M2'].mean():8.4f} | {asin_mae_df['M2改善率(%)'].mean():7.2f}%")
print("="*80)

# ==========================================
# 6. 時系列グラフの表示
# ==========================================
mae_time_series = []
current_date = test_overall['date'].min()
while current_date <= test_overall['date'].max():
    window_end = current_date + pd.Timedelta(days=29)
    batch = test_overall[(test_overall['date'] >= current_date) & (test_overall['date'] <= window_end)]
    if not batch.empty:
        def get_pred_batch(model, cols):
            exog = sm.add_constant(batch[cols], has_constant='add')[model.params.index]
            return model.predict(exog)
        mae_time_series.append({
            'テスト開始': current_date, 
            'MAE_M0': mean_absolute_error(batch['ln_y'], get_pred_batch(model_m0, X_m0_cols)),
            'MAE_M1': mean_absolute_error(batch['ln_y'], get_pred_batch(model_m1, X_m1_cols)),
            'MAE_M2': mean_absolute_error(batch['ln_y'], get_pred_batch(model_m2, X_m2_cols))
        })
    current_date += pd.Timedelta(days=30)
err_df = pd.DataFrame(mae_time_series)

if not err_df.empty:
    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=150)
    daily_avg = full_df.groupby('date')['ln_y'].mean().reset_index()
    ax1.plot(daily_avg['date'], daily_avg['ln_y'], color='green', linewidth=0.8, label='相対売上', alpha=0.5)
    ax1.set_xlabel('日付')
    ax1.set_ylabel('ln_y (相対売上)')
    
    ax2 = ax1.twinx()
    ax2.plot(err_df['テスト開始'], err_df['MAE_M0'], color='#7F7F7F', marker='s', label='MAE: M0(Base)', linestyle='--', alpha=0.7)
    ax2.plot(err_df['テスト開始'], err_df['MAE_M1'], color='#1F77B4', marker='^', label='MAE: M1')
    ax2.plot(err_df['テスト開始'], err_df['MAE_M2'], color='#D62728', marker='o', label='MAE: M2', linewidth=2)
    ax2.set_ylabel('MAE')
    
    ax1.set_title('デジカメ：相対売上と精度推移（全ASIN平均）', pad=15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.show()
