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
font_size_global = 20
title_size = font_size_global - 5
label_size = font_size_global + 15
tick_size = font_size_global - 8
legend_size = font_size_global + 2
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': label_size})

# ==========================================
# 1. 設定項目
# ==========================================
base_dir = r'C:\Users\admin\Desktop\4年\本実験\ASIN\camera'
spec_file_name = 'ASIN+spec.xlsx'
trends_file_name = 'trends_2023_2025.csv'
trends_2012_file = 'trends_ca_2012.csv'

# ラグの設定
target_lags = list(range(7, 31)) 
lag_vars = [f'ln_y_lag{i}' for i in target_lags]

asins = ["B00IDV4ACC", "B00S7LBA08", "B00XXCOJLS", "B0B1LVFFB5", "B01ALAZOOK",
         "B01FH4H5LA", "B01LXIND6Y", "B07FBJ9WNB", "B07G28MFV5", "B07GZGNTXZ",
         "B07JNL93SQ", "B07NVXM29C", "B07SLXZCBW", "B07V6DHHRX", "B07VYHDDF9",
         "B08N4FKFCG", "B09NDF5BN7", "B09ZRN1N3Z", "B072J97HT1", "B083K3SHCK",
         "B097C55DCP", "B099RKG1FN", "B007410K80"]

inter_vars_m1 = ['inter_res_season', 'inter_vol_season', 'inter_pri_season', 'inter_zoo_season']
inter_vars_m2 = ['search_2012_impact']
category_var = ['ln_z_category']

# ==========================================
# 2. データの読み込みと前処理
# ==========================================
trends_df = pd.DataFrame({
    'date': pd.to_datetime(pd.read_csv(os.path.join(base_dir, trends_file_name), encoding='cp932').iloc[:, 0]),
    'z_res': pd.read_csv(os.path.join(base_dir, trends_file_name), encoding='cp932').iloc[:, 1],
    'z_vol': pd.read_csv(os.path.join(base_dir, trends_file_name), encoding='cp932').iloc[:, 2],
    'z_pri': pd.read_csv(os.path.join(base_dir, trends_file_name), encoding='cp932').iloc[:, 3],
    'z_zoo': pd.read_csv(os.path.join(base_dir, trends_file_name), encoding='cp932').iloc[:, 4],
    'z_cat': pd.read_csv(os.path.join(base_dir, trends_file_name), encoding='cp932').iloc[:, 6]
})

t12_raw = pd.read_csv(os.path.join(base_dir, trends_2012_file), encoding='cp932')
trends_12_df = pd.DataFrame({'date': pd.to_datetime(t12_raw.iloc[:, 0]), 'search_b': t12_raw.iloc[:, 1], 'seasonal_d': t12_raw.iloc[:, 3]})

specs_master = pd.read_excel(os.path.join(base_dir, spec_file_name), usecols=[0, 2, 3, 4, 5])
specs_master.columns = ['asin', 'f_res', 'f_vol', 'f_pri', 'f_zoo']

all_data_list = []
eps = 1e-6

for asin in asins:
    file_path = os.path.join(base_dir, f'salesrank_daily_2023_2025_{asin}.csv')
    if not os.path.exists(file_path): continue
    df = pd.merge(pd.read_csv(file_path).assign(date=lambda x: pd.to_datetime(x['date'])), trends_df, on='date', how='left')
    df = pd.merge(df, trends_12_df, on='date', how='left').sort_values('date')
    df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).interpolate(limit_direction='both')
    df['q_it'] = 7.831 * (df['daily_salesrank'].clip(lower=1) ** (-0.713))
    df['ln_y'] = np.log(df['q_it'] + eps)
    
    new_cols = {f'ln_y_lag{i}': df['ln_y'].shift(i) for i in target_lags}
    new_cols['ln_z_category'] = np.log(df['z_cat'] + eps)
    
    spec = specs_master[specs_master['asin'] == asin].iloc[0] if asin in specs_master['asin'].values else None
    if spec is not None:
        for k in ['res', 'vol', 'pri', 'zoo']:
            new_cols[f'inter_{k}_season'] = np.log(pd.to_numeric(spec[f'f_{k}']) + eps) * np.log(df[f'z_{k}'] + eps) * new_cols['ln_z_category']
    new_cols['search_2012_impact'] = np.log(df['search_b'] + eps) * np.log(df['seasonal_d'] + eps)
    all_data_list.append(pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1))

full_df = pd.concat(all_data_list).dropna().sort_values('date')

# ==========================================
# 3. 学習・予測メインロジック
# ==========================================
def get_top_10(df, year):
    return df[df['date'].dt.year == year].groupby('asin')['q_it'].mean().nlargest(10).index.tolist()

top_10_2023 = get_top_10(full_df, 2023)
top_10_2024 = get_top_10(full_df, 2024)
top_10_2025 = get_top_10(full_df, 2025)

def run_session(train_years, test_year, train_asins, test_asins):
    # 追加：使用ASINの表示
    print(f"\n===== Session: Test Year {test_year} =====")
    print(f"学習用ASIN (Top10 {train_years}年ベース): {train_asins}")
    print(f"予測対象ASIN (Top10 {test_year}年ベース): {test_asins}")
    
    target_vars = lag_vars + category_var + inter_vars_m1 + inter_vars_m2
    train_df = full_df[full_df['date'].dt.year.isin(train_years) & full_df['asin'].isin(train_asins)].copy()
    test_df = full_df[(full_df['date'].dt.year == test_year) & full_df['asin'].isin(test_asins)].copy()
    
    sc = StandardScaler()
    train_df[target_vars] = sc.fit_transform(train_df[target_vars])
    test_df[target_vars] = sc.transform(test_df[target_vars])
    
    dummies = pd.get_dummies(train_df['asin'], prefix='d', drop_first=True).astype(float)
    train_df = pd.concat([train_df, dummies], axis=1)
    d_cols = dummies.columns.tolist()
    
    models = {
        'M0:ベースラインモデル': lag_vars + category_var + d_cols,
        'M1:製品特徴モデル': lag_vars + category_var + inter_vars_m1 + d_cols,
        'M2:カテゴリ検索モデル': lag_vars + category_var + inter_vars_m2 + d_cols
    }
    
    results = {}
    for name, cols in models.items():
        model_fit = sm.OLS(train_df['ln_y'], sm.add_constant(train_df[cols])).fit()
        exog_test = sm.add_constant(test_df, has_constant='add')
        for c in model_fit.params.index:
            if c not in exog_test.columns: exog_test[c] = 0
        
        preds = model_fit.predict(exog_test[model_fit.params.index])
        mae_series = np.abs(test_df['ln_y'].values - preds.values)
        
        results[name] = {'mae': mean_absolute_error(test_df['ln_y'], preds),
                         'pred': preds,
                         'mae_series': mae_series,
                         'params': model_fit.params}
    return results, test_df['date'], test_df['ln_y']

res24, d24, y24 = run_session([2023], 2024, top_10_2023, top_10_2024)
res25, d25, y25 = run_session([2023, 2024], 2025, list(set(top_10_2023 + top_10_2024)), top_10_2025)

# ==========================================
# 4. 統計の表示
# ==========================================
def print_summary(year, res):
    print(f"\n--- {year}年 評価結果 ---")
    m0_k, m1_k, m2_k = 'M0:ベースラインモデル', 'M1:製品特徴モデル', 'M2:カテゴリ検索モデル'
    m0_mae, m1_mae, m2_mae = res[m0_k]['mae'], res[m1_k]['mae'], res[m2_k]['mae']
    print(f"MAE | M0: {m0_mae:.4f}, M1: {m1_mae:.4f}, M2: {m2_mae:.4f}")
    print(f"改善率 vs M0 | M1: {(1-m1_mae/m0_mae)*100:+.2f}%, M2: {(1-m2_mae/m0_mae)*100:+.2f}%")
    
    p0, p1, p2 = res[m0_k]['params'], res[m1_k]['params'], res[m2_k]['params']
    
    first_lag_name = lag_vars[0]
    rest_lag_names = lag_vars[1:]
    
    rest_avg_m0 = p0[rest_lag_names].mean()
    rest_avg_m1 = p1[rest_lag_names].mean()
    rest_avg_m2 = p2[rest_lag_names].mean()
    
    min_l = target_lags[0]
    rest_l_range = f"{target_lags[1]}-{target_lags[-1]}" if len(target_lags) > 1 else "N/A"
    
    print(f"[代表重み] {first_lag_name: <10} | M0: {p0[first_lag_name]:8.4f}, M1: {p1[first_lag_name]:8.4f}, M2: {p2[first_lag_name]:8.4f}")
    print(f"[代表重み] Lag{rest_l_range}Avg | M0: {rest_avg_m0:8.4f}, M1: {rest_avg_m1:8.4f}, M2: {rest_avg_m2:8.4f}")
    
    # --- 追加：全変数の重み一覧表示 ---
    print(f"\n<<< {year}年 全変数重み（係数）一覧 >>>")
    all_vars_list = p1.index.union(p2.index).union(p0.index).tolist()
    # ソート順: const, lags, category, inter_m1, inter_m2, dummies
    print(f"{'Variable':<25} | {'M0':>10} | {'M1':>10} | {'M2':>10}")
    print("-" * 65)
    for v in all_vars_list:
        v0 = f"{p0[v]:10.4f}" if v in p0 else f"{'-':>10}"
        v1 = f"{p1[v]:10.4f}" if v in p1 else f"{'-':>10}"
        v2 = f"{p2[v]:10.4f}" if v in p2 else f"{'-':>10}"
        print(f"{v:<25} | {v0} | {v1} | {v2}")
    print("-" * 65)

print_summary("2024", res24)
print_summary("2025", res25)

# ==========================================
# 5. 二軸グラフ描画
# ==========================================
y23_avg = full_df[full_df['date'].dt.year == 2023].groupby('date')['ln_y'].mean()
y24_avg = pd.DataFrame({'Actual': y24, 'date': d24}).groupby('date').mean()
y25_avg = pd.DataFrame({'Actual': y25, 'date': d25}).groupby('date').mean()
all_actual = pd.concat([y23_avg, y24_avg['Actual'], y25_avg['Actual']])

def get_mae_30d(res, dates):
    df = pd.DataFrame({
        'date': dates,
        'M0': res['M0:ベースラインモデル']['mae_series'],
        'M1': res['M1:製品特徴モデル']['mae_series'],
        'M2': res['M2:カテゴリ検索モデル']['mae_series']
    }).groupby('date').mean()
    return df.resample('30D').mean()

mae_plot_data = pd.concat([get_mae_30d(res24, d24), get_mae_30d(res25, d25)])

fig, ax1 = plt.subplots(figsize=(15, 7))
ax1.plot(all_actual.index, all_actual.values, color='green', label='相対売上(平均)', linewidth=1.2, alpha=0.7)
ax1.set_ylabel('相対売上 (ln_y)')
ax1.tick_params(axis='y',labelsize=tick_size)
ax1.tick_params(axis='x', labelsize=tick_size+14)

ax2 = ax1.twinx()
ax2.plot(mae_plot_data.index, mae_plot_data['M0'], color='gray', marker='s', markersize=4, linestyle=':', label='MAE: M0(ベースラインモデル)', alpha=0.6)
ax2.plot(mae_plot_data.index, mae_plot_data['M1'], color='blue', marker='o', markersize=5, label='MAE: M1(製品特徴モデル)', linewidth=1.5)
ax2.plot(mae_plot_data.index, mae_plot_data['M2'], color='red', marker='^', markersize=5, label='MAE: M2(カテゴリ検索モデル)', linewidth=1.2, alpha=0.8)
ax2.set_ylabel('予測誤差 (30日平均MAE)')
ax2.tick_params(axis='y', labelsize=tick_size)

plt.axvline(pd.to_datetime('2024-01-01'), color='black', linestyle='--', alpha=0.3)
plt.axvline(pd.to_datetime('2025-01-01'), color='black', linestyle='--', alpha=0.3)
plt.title('デジカメ：相対売上とモデル別MAEの推移', fontsize=title_size)
ax1.grid(axis='y', alpha=0.2)

handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()
ax1.legend(handler1 + handler2, label1 + label2, loc='upper left', fontsize=legend_size)

plt.tight_layout()
plt.show()
