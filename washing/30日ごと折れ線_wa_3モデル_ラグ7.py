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
font_size_global = 26
title_size = font_size_global
label_size = font_size_global
tick_size = font_size_global - 2
legend_size = font_size_global - 10

# ==========================================
# 1. 設定項目
# ==========================================
base_dir = r'C:\Users\admin\Desktop\4年\本実験\ASIN\washing'
spec_file_name = 'ASIN+spec_wa.xlsx'
wa_daily_path = os.path.join(base_dir, 'wa_daily.csv')
wa_weekly_path = os.path.join(base_dir, 'wa_weekly.csv')
spec_file_path = os.path.join(base_dir, spec_file_name)
trends_2012_file = 'trends_wa_2012.csv' 

# --- 【変更点】ラグ7のみを指定（ラグ1は削除） ---
target_lags = [7] 
lag_vars = [f'ln_y_lag{i}' for i in target_lags]
# -------------------------------------------

asins = list(dict.fromkeys(["B0BP6ZCQC5", "B09TSBGDN4", "B08Q7Q29CQ", "B08B1GLD8L", "B0B313BKHK",
           "B0B31621RB", "B084K7PM6J", "B07JR174ZT", "B09J8MF636", "B0BQQKTMVD",
           "B07MZLN15F", "B07RWR7BX6", "B084K7PM6J", "B07MZLN15F", "B07RWR7BX6", 
           "B08T1DM8KQ", "B0815PXPKZ", "B08KH6QFYZ", "B01C10VSM0", "B08B13BXQL", 
           "B094VFZXMG", "B01G58K3KM", "B0BP6ZS2TX", "B0BP6VPD4K", "B09N71PGTB", 
           "B09N73WQ2V", "B0BD3GJGLF", "B0BD3FYJ3S", "B0BH8JRJCQ", "B075B5WR85", 
           "B083N9LZ6Y", "B0BP6J3SQX", "B07LBWQ3BF", "B01C10VTE2", "B07YLRMKRG", 
           "B07JQX29XR", "B0BLMJJ9TV", "B09DKK6VCL", "B09P31T6GF", "B083PR3Y2K", 
           "B07KCNJY79", "B09DSQQGT8", "B07SQCZRCG", "B09DKKZHFC", "B07YC9Z3N1", 
           "B07YCBRJFQ", "B07YC9Z3MV", "B09NNH112Q", "B0956GJ3CL", "B0BPM6K84W", 
           "B0BPM8TQLZ", "B09571SNLK", "B0956PSK2H", "B0956MW64T", "B09G2L68H2", 
           "B09G2KS69R", "B09G2LJQ98", "B09G2LNCXG", "B09DKGXYCV", "B01HUJES06", 
           "B0162EV9QQ", "B07ZPSWGYN", "B07CDRZYM5"]))

market_main_vars = ['ln_f_sum'] 
inter_vars_m1 = ['inter_cap', 'inter_aut', 'inter_pri', 'inter_siz']
inter_vars_m2 = ['search_2012_impact']

target_vars = lag_vars + market_main_vars + inter_vars_m1 + inter_vars_m2

# ==========================================
# 2. データの読み込み
# ==========================================
wa_weekly = pd.read_csv(wa_weekly_path, encoding='cp932')
wa_weekly.columns = wa_weekly.columns.str.strip() 
wa_weekly['date'] = pd.to_datetime(wa_weekly['date'])
wa_weekly = wa_weekly.sort_values('date').set_index('date')

daily_range = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
wa_weekly_interp = wa_weekly.reindex(daily_range).interpolate(method='linear').reset_index().rename(columns={'index': 'date'})

wa_daily = pd.read_csv(wa_daily_path, encoding='cp932')
wa_daily.columns = wa_daily.columns.str.strip()
wa_daily['date'] = pd.to_datetime(wa_daily['date'])

trends_2012_path = os.path.join(base_dir, trends_2012_file)
t12_raw = pd.read_csv(trends_2012_path, encoding='cp932')
trends_12_df = pd.DataFrame({
    'date': pd.to_datetime(t12_raw.iloc[:, 0]),
    'search_b': t12_raw.iloc[:, 1], 
    'seasonal_d': t12_raw.iloc[:, 3]
})

search_adj = wa_daily.merge(wa_weekly_interp[['date', 'f_sum']], on='date', how='left')
search_adj = search_adj.merge(trends_12_df, on='date', how='left')
num_cols_adj = search_adj.select_dtypes(include=[np.number]).columns
search_adj[num_cols_adj] = search_adj[num_cols_adj].interpolate(method='linear', limit_direction='both')

for fk in ['f_cap', 'f_aut', 'f_pri', 'f_siz']:
    search_adj[f'adj_{fk}'] = search_adj[fk] * search_adj['f_sum']

search_final = search_adj[['date', 'adj_f_cap', 'adj_f_aut', 'adj_f_pri', 'adj_f_siz', 'f_sum', 'search_b', 'seasonal_d']]

# ==========================================
# 3. データ統合と前処理
# ==========================================
specs_master = pd.read_excel(spec_file_path, usecols=[0, 1, 2, 3, 4])
specs_master.columns = ['asin', 'cap_spec', 'aut_spec', 'pri_spec', 'siz_spec']

def get_numeric_scalar(val):
    num = pd.to_numeric(val, errors='coerce')
    return float(num) if (pd.notna(num) and num > 0) else np.nan

all_data = []
eps = 1e-6

for asin in asins:
    file_path = os.path.join(base_dir, f'salesrank_daily_{asin}.csv')
    if not os.path.exists(file_path): continue
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(search_final, on='date', how='left')
    df = df.sort_values('date')
    df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2025-12-31')]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    
    spec_match = specs_master[specs_master['asin'] == asin]
    if spec_match.empty: continue
    spec_row = spec_match.iloc[0]
    
    k_c, A_c = 0.684, 11.74
    df['q_it'] = A_c * (df['daily_salesrank'].clip(lower=1) ** (-k_c))
    df['ln_y'] = np.log(df['q_it'] + eps)

    new_cols = {}
    for i in target_lags:
        new_cols[f'ln_y_lag{i}'] = df['ln_y'].shift(i)
    
    new_cols['ln_f_sum'] = np.log(df['f_sum'] + eps)
    
    ln_z_cap = np.log(df['adj_f_cap'] + eps)
    ln_z_aut = np.log(df['adj_f_aut'] + eps)
    ln_z_pri = np.log(df['adj_f_pri'] + eps)
    ln_z_siz = np.log(df['adj_f_siz'] + eps)

    f_cap_s = np.log(get_numeric_scalar(spec_row['cap_spec'])) if pd.notna(get_numeric_scalar(spec_row['cap_spec'])) else 0.0
    f_pri_s = np.log(get_numeric_scalar(spec_row['pri_spec'])) if pd.notna(get_numeric_scalar(spec_row['pri_spec'])) else 0.0
    f_siz_s = np.log(get_numeric_scalar(spec_row['siz_spec'])) if pd.notna(get_numeric_scalar(spec_row['siz_spec'])) else 0.0
    d_aut_s = pd.to_numeric(spec_row['aut_spec'], errors='coerce') if pd.notna(pd.to_numeric(spec_row['aut_spec'], errors='coerce')) else 0.0

    new_cols['inter_cap'] = ln_z_cap * f_cap_s
    new_cols['inter_aut'] = ln_z_aut * d_aut_s 
    new_cols['inter_pri'] = ln_z_pri * f_pri_s
    new_cols['inter_siz'] = ln_z_siz * f_siz_s

    new_cols['search_2012_impact'] = np.log(df['search_b'] + eps) * np.log(df['seasonal_d'] + eps)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    all_data.append(df)

full_df = pd.concat(all_data).dropna().sort_values('date')
asin_dummies = pd.get_dummies(full_df['asin'], drop_first=True).astype(float)
full_df = pd.concat([full_df, asin_dummies], axis=1)

# ==========================================
# 4. 特徴量の正規化（標準化）
# ==========================================
train_mask = full_df['date'].dt.year == 2023
test_mask = full_df['date'].dt.year >= 2024

scaler = StandardScaler()
full_df.loc[train_mask, target_vars] = scaler.fit_transform(full_df.loc[train_mask, target_vars])
full_df.loc[test_mask, target_vars] = scaler.transform(full_df.loc[test_mask, target_vars])

# ==========================================
# 5. 学習と予測
# ==========================================
train_data = full_df[train_mask].copy()
test_overall = full_df[test_mask].copy()
dummy_cols = asin_dummies.columns.tolist()

X_m0_cols = lag_vars + market_main_vars + dummy_cols
X_m1_cols = lag_vars + market_main_vars + inter_vars_m1 + dummy_cols
X_m2_cols = lag_vars + market_main_vars + inter_vars_m2 + dummy_cols

model_m0 = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[X_m0_cols])).fit()
model_m1 = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[X_m1_cols])).fit()
model_m2 = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[X_m2_cols])).fit()

# --- 重みの表示 ---
print("\n" + "="*60)
print(f"{'変数名':<25} | {'M1 (従来)':>10} | {'M2 (2012)':>10}")
print("-" * 60)
display_vars = lag_vars + market_main_vars

for var in display_vars:
    if var in model_m1.params:
        print(f"{var:<25} | {model_m1.params[var]:10.4f} | {model_m2.params[var]:10.4f}")

print("-" * 60)
for var in inter_vars_m1:
    print(f"{var:<25} | {model_m1.params[var]:10.4f} | {'-':>10}")
for var in inter_vars_m2:
    print(f"{var:<25} | {'-':>10} | {model_m2.params[var]:10.4f}")

# --- 予測評価 ---
mae_results = []
current_date = test_overall['date'].min()
while current_date <= test_overall['date'].max(): # 修正済み
    window_end = current_date + pd.Timedelta(days=29)
    batch = test_overall[(test_overall['date'] >= current_date) & (test_overall['date'] <= window_end)]
    if not batch.empty:
        def get_pred(model, cols):
            exog = sm.add_constant(batch[cols], has_constant='add')[model.params.index]
            return model.predict(exog)
        mae_results.append({
            'テスト開始': current_date, 
            'MAE_M0': mean_absolute_error(batch['ln_y'], get_pred(model_m0, X_m0_cols)),
            'MAE_M1': mean_absolute_error(batch['ln_y'], get_pred(model_m1, X_m1_cols)),
            'MAE_M2': mean_absolute_error(batch['ln_y'], get_pred(model_m2, X_m2_cols))
        })
    current_date += pd.Timedelta(days=30)

err_df = pd.DataFrame(mae_results)

# --- 平均MAEと平均改善率の算出 ---
avg_mae_m0 = err_df['MAE_M0'].mean()
avg_mae_m1 = err_df['MAE_M1'].mean()
avg_mae_m2 = err_df['MAE_M2'].mean()

m1_imp = (1 - avg_mae_m1/avg_mae_m0)*100
m2_imp = (1 - avg_mae_m2/avg_mae_m0)*100

print("\n" + "="*60)
print(f"MAE 平均 (M0:ベースライン) : {avg_mae_m0:.4f}")
print(f"MAE 平均 (M1:従来モデル)   : {avg_mae_m1:.4f}")
print(f"MAE 平均 (M2:2012モデル)   : {avg_mae_m2:.4f}")
print("-" * 60)
print(f"M1 平均改善率 : {m1_imp:.2f}%")
print(f"M2 平均改善率 : {m2_imp:.2f}%")
print("="*60)

# ==========================================
# 6. 可視化
# ==========================================
if not err_df.empty:
    fig, ax1 = plt.subplots(figsize=(16, 9), dpi=100)
    daily_avg = full_df.groupby('date')['ln_y'].mean().reset_index()
    
    ax1.plot(daily_avg['date'], daily_avg['ln_y'], color='green', linewidth=0.8, label='相対売上', alpha=1.0)
    ax1.set_xlabel('日付', fontsize=label_size)
    ax1.set_ylabel('相対売上（ln_y）', fontsize=label_size)
    
    ax2 = ax1.twinx()
    ax2.plot(err_df['テスト開始'], err_df['MAE_M0'], color='#7F7F7F', marker='s', label='MAE: M0 (Baseline)', linestyle='--', alpha=0.7)
    ax2.plot(err_df['テスト開始'], err_df['MAE_M1'], color='#1F77B4', marker='^', label='MAE: M1')
    ax2.plot(err_df['テスト開始'], err_df['MAE_M2'], color='#D62728', marker='o', label='MAE: M2', linewidth=3)
    ax2.set_ylabel('予測誤差（MAE）', fontsize=label_size)
    ax2.set_ylim(0, 0.35)

    ax1.grid(True, which='both', linestyle='--', alpha=0.3, color='gray')
    ax1.set_title('洗濯機：相対売上と精度推移（ラグ7のみ）', fontsize=title_size, pad=20)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=legend_size, framealpha=0.8)
    plt.tight_layout()
    plt.show()
