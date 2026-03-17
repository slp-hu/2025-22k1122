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

max_lag = 30
asins = ["B00IDV4ACC", "B00S7LBA08", "B00XXCOJLS", "B0B1LVFFB5", "B01ALAZOOK",
         "B01FH4H5LA", "B01LXIND6Y", "B07FBJ9WNB", "B07G28MFV5", "B07GZGNTXZ",
         "B07JNL93SQ", "B07NVXM29C", "B07SLXZCBW", "B07V6DHHRX", "B07VYHDDF9",
         "B08N4FKFCG", "B09NDF5BN7", "B09ZRN1N3Z", "B072J97HT1", "B083K3SHCK",
         "B097C55DCP", "B099RKG1FN", "B007410K80"]

lag_vars = [f'ln_y_lag{i}' for i in range(1, max_lag + 1)]
inter_vars = ['inter_res_season', 'inter_vol_season', 'inter_pri_season', 'inter_zoo_season']
category_var = ['ln_z_category']

# 標準化の対象とする連続変数リスト
target_vars = lag_vars + category_var + inter_vars

# ==========================================
# 2. データの読み込みと前処理
# ==========================================
spec_file_path = os.path.join(base_dir, spec_file_name)
specs_master = pd.read_excel(spec_file_path, usecols=[0, 2, 3, 4, 5])
specs_master.columns = ['asin', 'f_res', 'f_vol', 'f_pri', 'f_zoo']

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

all_data = []
for asin in asins:
    file_path = os.path.join(base_dir, f'salesrank_daily_2023_2025_{asin}.csv')
    if not os.path.exists(file_path): continue
    
    df_product = pd.read_csv(file_path)
    df_product['date'] = pd.to_datetime(df_product['date'])
    
    df = pd.merge(df_product, trends_df, on='date', how='left')
    df = df.sort_values('date')
    df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2025-12-31')]
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method='linear', limit_direction='both')
    
    spec_match = specs_master[specs_master['asin'] == asin]
    if spec_match.empty: continue
    spec = spec_match.iloc[0]
    
    # ランクから販売量への変換
    k_c, A_c = 0.713, 7.831
    df['q_it'] = A_c * (df['daily_salesrank'].clip(lower=1) ** (-k_c))
    df['ln_y'] = np.log(df['q_it']) 

    new_cols = {}
    for i in range(1, max_lag + 1):
        new_cols[f'ln_y_lag{i}'] = df['ln_y'].shift(i)
    
    eps = 1e-6
    ln_cat = np.log(df['z_cat'] + eps)
    new_cols['ln_z_category'] = ln_cat
    
    spec_map = {'res': 'f_res', 'vol': 'f_vol', 'pri': 'f_pri', 'zoo': 'f_zoo'}
    trend_map = {'res': 'z_res', 'vol': 'z_vol', 'pri': 'z_pri', 'zoo': 'z_zoo'}
    
    for key in ['res', 'vol', 'pri', 'zoo']:
        f_val = pd.to_numeric(spec[spec_map[key]], errors='coerce')
        ln_f_val = np.log(f_val + eps)
        ln_z_val = np.log(df[trend_map[key]] + eps)
        # 修正箇所: 変数名のクォートとf-stringを正しく記述
        new_cols[f'inter_{key}_season'] = ln_f_val * ln_z_val * ln_cat

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    all_data.append(df)

full_df = pd.concat(all_data).dropna().sort_values('date')

# ASINダミー変数の作成（これは標準化しない）
asin_dummies = pd.get_dummies(full_df['asin'], drop_first=True).astype(float)
full_df = pd.concat([full_df, asin_dummies], axis=1)

# ==========================================
# 3. 特徴量の正規化（標準化）
# ==========================================
train_mask = full_df['date'].dt.year == 2023
test_mask = full_df['date'].dt.year >= 2024

scaler = StandardScaler()

# 2023年のデータで学習し適用
full_df.loc[train_mask, target_vars] = scaler.fit_transform(full_df.loc[train_mask, target_vars])
# 2024年以降に適用
full_df.loc[test_mask, target_vars] = scaler.transform(full_df.loc[test_mask, target_vars])

# ==========================================
# 4. 学習と評価
# ==========================================
train_data = full_df[train_mask]
test_overall = full_df[test_mask].copy()
dummy_cols = asin_dummies.columns.tolist()

X_proposed_cols = target_vars + dummy_cols
X_baseline_cols = lag_vars + category_var + dummy_cols

model_p = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[X_proposed_cols])).fit()
model_b = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[X_baseline_cols])).fit()

# --- 重み（標準化回帰係数）の出力 ---
print("\n" + "="*50)
print("提案モデル：標準化回帰係数（影響度の比較用）")
print("="*50)

print(f"[自己回帰項 (AR components)]")
for lag in ['ln_y_lag1', 'ln_y_lag7', 'ln_y_lag30']:
    print(f"  {lag:15}: {model_p.params[lag]:8.4f}")

print(f"\n[外部需要・相互作用項]")
print(f"  市場トレンド (ln_z_category): {model_p.params['ln_z_category']:.4f}")
for var in inter_vars:
    print(f"  {var:15}: {model_p.params[var]:8.4f}")

# --- 予測評価ループ ---
mae_results = []
current_date = test_overall['date'].min()
end_date = test_overall['date'].max()

while current_date <= end_date:
    window_end = current_date + pd.Timedelta(days=29)
    batch = test_overall[(test_overall['date'] >= current_date) & (test_overall['date'] <= window_end)]
    if not batch.empty:
        pred_p = model_p.predict(sm.add_constant(batch[X_proposed_cols], has_constant='add'))
        pred_b = model_b.predict(sm.add_constant(batch[X_baseline_cols], has_constant='add'))
        mae_results.append({
            'テスト開始': current_date, 
            'MAE_ベースライン': mean_absolute_error(batch['ln_y'], pred_b),
            'MAE_提案モデル': mean_absolute_error(batch['ln_y'], pred_p)
        })
    current_date += pd.Timedelta(days=30)

err_df = pd.DataFrame(mae_results)
print("\n" + "="*50)
print(f"平均改善率: {(1 - err_df['MAE_提案モデル'].mean()/err_df['MAE_ベースライン'].mean())*100:.2f}%")
print("="*50)

# ==========================================
# 5. 可視化
# ==========================================
if not err_df.empty:
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
    
    daily_lny_avg = full_df.groupby('date')['ln_y'].mean().reset_index()
    ax1.plot(daily_lny_avg['date'], daily_lny_avg['ln_y'], color='#FFCCCC', linewidth=0.8, label='相対売上(実績平均)', alpha=0.8)
    ax1.set_xlabel('日付', fontsize=12)
    ax1.set_ylabel('相対売上（ln_y）', fontsize=12)
    
    ax2 = ax1.twinx()
    # 凡例用にそれぞれのプロットを変数に格納
    line_p, = ax2.plot(err_df['テスト開始'], err_df['MAE_提案モデル'], color='#D62728', marker='o', markersize=4, label='MAE：提案モデル', linewidth=1.5)
    line_b, = ax2.plot(err_df['テスト開始'], err_df['MAE_ベースライン'], color='#1F77B4', marker='s', markersize=4, label='MAE：ベースライン', linewidth=1.5)
    
    ax2.set_ylabel('予測誤差（MAE）', fontsize=12)
    ax2.set_ylim(0, 0.5) 

    ax1.grid(True, which='both', linestyle='--', alpha=0.3, color='gray')
    ax1.set_title('標準化後のモデルによる予測誤差推移', fontsize=14, pad=15)

    # 凡例をまとめて表示
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # ここを修正：lines1(リスト) と lines2(リスト) を結合
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.show()
