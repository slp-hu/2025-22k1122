import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
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
base_dir = r'C:\Users\admin\Desktop\4年\本実験\ASIN\camera'
spec_file_name = 'ASIN+spec.xlsx'

max_lag = 30

asins = ["B00IDV4ACC", "B00S7LBA08", "B00XXCOJLS", "B0B1LVFFB5", "B01ALAZOOK",
         "B01FH4H5LA", "B01LXIND6Y", "B07FBJ9WNB", "B07G28MFV5", "B07GZGNTXZ",
         "B07JNL93SQ", "B07NVXM29C", "B07SLXZCBW", "B07V6DHHRX", "B07VYHDDF9",
         "B08N4FKFCG", "B09NDF5BN7", "B09ZRN1N3Z", "B072J97HT1", "B083K3SHCK",
         "B097C55DCP", "B099RKG1FN", "B007410K80"]

# 変数リストの定義
lag_vars = [f'ln_y_lag{i}' for i in range(1, max_lag + 1)]
inter_vars = ['inter_res', 'inter_vol', 'inter_pri', 'inter_zoo']
category_var = ['ln_z_category'] # H列のカテゴリトレンド

# 提案モデルの全変数
target_vars = lag_vars + category_var + inter_vars

# ==========================================
# 2. データの読み込みと前処理
# ==========================================
spec_file_path = os.path.join(base_dir, spec_file_name)
specs_master = pd.read_excel(spec_file_path, usecols=[0, 2, 3, 4, 5])
specs_master.columns = ['asin', 'f_res', 'f_vol', 'f_pri', 'f_zoo']

all_data = []
for asin in asins:
    file_path = os.path.join(base_dir, f'salesrank_daily_2023_2025_{asin}.csv')
    if not os.path.exists(file_path): continue
    
    # CSV読み込み（H列がカテゴリ検索数と想定）
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2025-12-31')]
    
    # H列（インデックス7）をカテゴリトレンドとして取得
    # カラム名が不明な場合を考慮し、列番号で指定
    category_col_name = df.columns[7] 
    
    df['daily_salesrank'] = df['daily_salesrank'].interpolate(method='linear', limit_direction='both')
    spec_match = specs_master[specs_master['asin'] == asin]
    if spec_match.empty: continue
    spec = spec_match.iloc[0]
    
    # パレート変換
    k_c, A_c = 0.713, 1.0
    df['q_it'] = A_c * (df['daily_salesrank'].clip(lower=1) ** (-k_c))
    df['ln_y'] = np.log(df['q_it']) 

    new_cols = {}
    for i in range(1, max_lag + 1):
        new_cols[f'ln_y_lag{i}'] = df['ln_y'].shift(i)
    
    eps = 1e-6
    # カテゴリ全体のトレンド（季節性）を対数変換
    new_cols['ln_z_category'] = np.log(df[category_col_name] + eps)
    
    # 相互作用項の作成
    features = {'res': 'f_res', 'vol': 'f_vol', 'pri': 'f_pri', 'zoo': 'f_zoo'}
    search_keywords = {'res': '解像度', 'vol': 'サイズ', 'pri': '価格', 'zoo': 'ズーム性能'}
    
    for key, f_col in features.items():
        z_col = search_keywords[key]
        f_val = pd.to_numeric(spec[f_col], errors='coerce')
        ln_f_val = np.log(f_val + eps)
        
        if z_col in df.columns:
            ln_z_val = np.log(df[z_col] + eps)
        else:
            ln_z_val = 0
            
        new_cols[f'inter_{key}'] = ln_z_val * ln_f_val

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    all_data.append(df)

full_df = pd.concat(all_data).dropna().sort_values('date')
asin_dummies = pd.get_dummies(full_df['asin'], drop_first=True).astype(float)
full_df = pd.concat([full_df, asin_dummies], axis=1)

# ==========================================
# 3. 学習と評価
# ==========================================
train_data = full_df[full_df['date'].dt.year == 2023]
test_overall = full_df[full_df['date'].dt.year >= 2024].copy()

dummy_cols = asin_dummies.columns.tolist()

# 提案モデル：自己回帰 + カテゴリ（季節性） + 相互作用項
X_proposed_cols = target_vars + dummy_cols
# ベースライン：自己回帰 + カテゴリ（季節性）のみ
X_baseline_cols = lag_vars + category_var + dummy_cols

model_p = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[X_proposed_cols])).fit()
model_b = sm.OLS(train_data['ln_y'], sm.add_constant(train_data[X_baseline_cols])).fit()

# --- 評価ループ ---
mae_results = []
current_date = test_overall['date'].min()
end_date = test_overall['date'].max()

while current_date <= end_date:
    window_end = current_date + pd.Timedelta(days=29)
    batch = test_overall[(test_overall['date'] >= current_date) & (test_overall['date'] <= window_end)]
    
    if not batch.empty:
        pred_p = model_p.predict(sm.add_constant(batch[X_proposed_cols], has_constant='add'))
        pred_b = model_b.predict(sm.add_constant(batch[X_baseline_cols], has_constant='add'))
        
        m_p = mean_absolute_error(batch['ln_y'], pred_p)
        m_b = mean_absolute_error(batch['ln_y'], pred_b)
        
        mae_results.append({
            'テスト開始': current_date.strftime('%Y-%m-%d'),
            'MAE_ベースライン': m_b,
            'MAE_提案モデル': m_p,
            '改善率(%)': (m_b - m_p) / m_b * 100
        })
    current_date += pd.Timedelta(days=30)
# ==========================================
# 4. 結果の出力 (自己回帰の重みを追加)
# ==========================================
err_df = pd.DataFrame(mae_results)
print(err_df.to_string(index=True))

print("\n========================================")
print("提案モデルの主要な回帰係数（Weight）")
print("========================================")

# --- 自己回帰項 (ラグ変数) の重み ---
print("[自己回帰項 (AR components)]")
ar_params = model_p.params[lag_vars]
print(f"  ラグ1日 (ln_y_lag1)  : {ar_params['ln_y_lag1']:.4f}")
print(f"  ラグ7日 (ln_y_lag7)  : {ar_params['ln_y_lag7']:.4f}")
print(f"  ラグ30日 (ln_y_lag30): {ar_params['ln_y_lag30']:.4f}")
print(f"  ラグ30日間の平均重み  : {ar_params.mean():.4f}")

# --- カテゴリトレンドと相互作用項 ---
print("\n[外部変数・相互作用項]")
print(f"  カテゴリトレンド(ln_z_category): {model_p.params['ln_z_category']:.4f}")
for var in inter_vars:
    print(f"  {var:15}: {model_p.params[var]:.4f}")

print(f"\n全期間の平均改善率: {err_df['改善率(%)'].mean():.2f}%")

# (以下、可視化コードは前回と同様)
# ==========================================
# 5. 可視化
# ==========================================
fig, ax1 = plt.subplots(figsize=(16, 9))
history_df = full_df.groupby('date')['ln_y'].mean().reset_index()
ax1.plot(history_df['date'], history_df['ln_y'], color='red', alpha=0.3, label='相対売上実績')
ax1.set_ylabel('ln(q_it)', fontsize=label_size)

ax2 = ax1.twinx()
ax2.plot(pd.to_datetime(err_df['テスト開始']), err_df['MAE_提案モデル'], color='tab:red', marker='o', label='MAE: 提案モデル')
ax2.plot(pd.to_datetime(err_df['テスト開始']), err_df['MAE_ベースライン'], color='tab:blue', marker='s', label='MAE: ベースライン')
ax2.set_ylabel('MAE', fontsize=label_size)

plt.title('カテゴリ季節性を考慮した予測誤差推移', fontsize=title_size)
ax1.legend(loc='upper left', fontsize=legend_size)
ax2.legend(loc='upper right', fontsize=legend_size)
plt.tight_layout()
plt.show()
