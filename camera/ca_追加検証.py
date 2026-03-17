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
plt.rcParams.update({'font.size': 12})

# ==========================================
# 1. 設定項目
# ==========================================
base_dir = r'C:\Users\admin\Desktop\4年\本実験\ASIN\camera'
spec_file_name = 'ASIN+spec.xlsx'
trends_file_name = 'trends_2023_2025.csv'
trends_2012_file = 'trends_ca_2012.csv'

# 実験のパラメータ
max_lag = 30
start_lag_range = list(range(2, 15))  # 2から14まで

asins = ["B00IDV4ACC", "B00S7LBA08", "B00XXCOJLS", "B0B1LVFFB5", "B01ALAZOOK",
         "B01FH4H5LA", "B01LXIND6Y", "B07FBJ9WNB", "B07G28MFV5", "B07GZGNTXZ",
         "B07JNL93SQ", "B07NVXM29C", "B07SLXZCBW", "B07V6DHHRX", "B07VYHDDF9",
         "B08N4FKFCG", "B09NDF5BN7", "B09ZRN1N3Z", "B072J97HT1", "B083K3SHCK",
         "B097C55DCP", "B099RKG1FN", "B007410K80"]

inter_vars_m1 = ['inter_res_season', 'inter_vol_season', 'inter_pri_season', 'inter_zoo_season']
inter_vars_m2 = ['search_2012_impact']
category_var = ['ln_z_category']

# ==========================================
# 2. データの読み込みと前処理（全ラグを事前に生成）
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

# 全てのラグ(1～max_lag)を先に作っておく
for asin in asins:
    file_path = os.path.join(base_dir, f'salesrank_daily_2023_2025_{asin}.csv')
    if not os.path.exists(file_path): continue
    df = pd.merge(pd.read_csv(file_path).assign(date=lambda x: pd.to_datetime(x['date'])), trends_df, on='date', how='left')
    df = pd.merge(df, trends_12_df, on='date', how='left').sort_values('date')
    df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).interpolate(limit_direction='both')
    df['q_it'] = 7.831 * (df['daily_salesrank'].clip(lower=1) ** (-0.713))
    df['ln_y'] = np.log(df['q_it'] + eps)
    
    # 1から30までのラグを生成
    lags_df = pd.DataFrame({f'ln_y_lag{i}': df['ln_y'].shift(i) for i in range(1, max_lag + 1)})
    
    new_cols_basic = pd.DataFrame()
    new_cols_basic['ln_z_category'] = np.log(df['z_cat'] + eps)
    new_cols_basic['search_2012_impact'] = np.log(df['search_b'] + eps) * np.log(df['seasonal_d'] + eps)
    
    spec = specs_master[specs_master['asin'] == asin].iloc[0] if asin in specs_master['asin'].values else None
    if spec is not None:
        for k in ['res', 'vol', 'pri', 'zoo']:
            new_cols_basic[f'inter_{k}_season'] = np.log(pd.to_numeric(spec[f'f_{k}']) + eps) * np.log(df[f'z_{k}'] + eps) * new_cols_basic['ln_z_category']
            
    all_data_list.append(pd.concat([df, lags_df, new_cols_basic], axis=1))

# dropnaは最大ラグ(30)に合わせて一括で行う（比較条件を揃えるため）
full_df_master = pd.concat(all_data_list).dropna().sort_values('date')

# ==========================================
# 3. 実験用ループ関数
# ==========================================
def run_experiment(start_lag, train_years, test_year, train_asins, test_asins):
    current_lags = [f'ln_y_lag{i}' for i in range(start_lag, max_lag + 1)]
    target_vars = current_lags + category_var + inter_vars_m1 + inter_vars_m2
    
    train_df = full_df_master[full_df_master['date'].dt.year.isin(train_years) & full_df_master['asin'].isin(train_asins)].copy()
    test_df = full_df_master[(full_df_master['date'].dt.year == test_year) & full_df_master['asin'].isin(test_asins)].copy()
    
    sc = StandardScaler()
    train_df[target_vars] = sc.fit_transform(train_df[target_vars])
    test_df[target_vars] = sc.transform(test_df[target_vars])
    
    dummies = pd.get_dummies(train_df['asin'], prefix='d', drop_first=True).astype(float)
    train_df = pd.concat([train_df, dummies], axis=1)
    d_cols = dummies.columns.tolist()
    
    models = {
        'M0': current_lags + category_var + d_cols,
        'M1': current_lags + category_var + inter_vars_m1 + d_cols,
        'M2': current_lags + category_var + inter_vars_m2 + d_cols
    }
    
    summary = {}
    for name, cols in models.items():
        fit = sm.OLS(train_df['ln_y'], sm.add_constant(train_df[cols])).fit()
        ex_test = sm.add_constant(test_df, has_constant='add')
        for c in fit.params.index:
            if c not in ex_test.columns: ex_test[c] = 0
        preds = fit.predict(ex_test[fit.params.index])
        
        summary[name] = {
            'mae': mean_absolute_error(test_df['ln_y'], preds),
            'params': fit.params,
            'first_lag_name': current_lags[0]
        }
    
    # 指標の集計
    m0_mae = summary['M0']['mae']
    m1_mae, m2_mae = summary['M1']['mae'], summary['M2']['mae']
    
    # 改善率
    m1_imp = (1 - m1_mae/m0_mae) * 100
    m2_imp = (1 - m2_mae/m0_mae) * 100
    
    # 検索重みの自己回帰に対する割合 (%)
    f_lag = current_lags[0]
    m1_ratio = (summary['M1']['params'][inter_vars_m1].abs().sum() / abs(summary['M1']['params'][f_lag])) * 100
    m2_ratio = (summary['M2']['params'][inter_vars_m2].abs().sum() / abs(summary['M2']['params'][f_lag])) * 100
    
    return {
        'StartLag': start_lag,
        'M1_Imp%': m1_imp,
        'M2_Imp%': m2_imp,
        'AvgProposedMAE': (m1_mae + m2_mae) / 2,
        'M1_WeightRatio%': m1_ratio,
        'M2_WeightRatio%': m2_ratio,
        'M1_AR1_Weight': summary['M1']['params'][f_lag],
        'M1_Search_WeightAbs': summary['M1']['params'][inter_vars_m1].abs().sum()
    }

# ==========================================
# 4. 一括実行と表示
# ==========================================
def get_top_10(df, year):
    return df[df['date'].dt.year == year].groupby('asin')['q_it'].mean().nlargest(10).index.tolist()

top23, top24, top25 = get_top_10(full_df_master, 2023), get_top_10(full_df_master, 2024), get_top_10(full_df_master, 2025)

for test_year, train_yrs, t_asins, v_asins in [(2024, [2023], top23, top24), (2025, [2023, 2024], list(set(top23+top24)), top25)]:
    results = []
    print(f"\n>>> {test_year}年 ラグ実験実行中...")
    for s in start_lag_range:
        results.append(run_experiment(s, train_yrs, test_year, t_asins, v_asins))
    
    # DataFrameにまとめて綺麗に表示
    res_df = pd.DataFrame(results).set_index('StartLag')
    
    print(f"\n--- {test_year}年 ラグ別サマリーテーブル ---")
    # 表示する列を整理
    display_cols = ['M1_Imp%', 'M2_Imp%', 'AvgProposedMAE', 'M1_WeightRatio%', 'M2_WeightRatio%']
    print(res_df[display_cols].round(4))
    
    # グラフ：開始ラグと改善率の関係
    plt.figure(figsize=(10, 5))
    plt.plot(res_df.index, res_df['M1_Imp%'], marker='o', label='M1 Improvement %')
    plt.plot(res_df.index, res_df['M2_Imp%'], marker='s', label='M2 Improvement %')
    plt.title(f'{test_year}: Start Lag vs Improvement Rate')
    plt.xlabel('Start Lag (Upper fixed at 30)')
    plt.ylabel('Improvement Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    #plt.show()
