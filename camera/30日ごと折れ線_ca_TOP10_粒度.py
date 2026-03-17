import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import matplotlib

# ==========================================
# 0. 日本語フォント・表示の設定
# ==========================================
# フォントの優先順位を設定（MS Gothic, Yu Gothic, Meiryoの順）
plt.rcParams["font.family"] = ["MS Gothic", "Yu Gothic", "Meiryo", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止

FONT_SIZE_GLOBAL = 15
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': FONT_SIZE_GLOBAL + 3,
    'axes.titlesize': FONT_SIZE_GLOBAL + 5,
    'xtick.labelsize': FONT_SIZE_GLOBAL - 2,
    'ytick.labelsize': FONT_SIZE_GLOBAL - 2,
    'legend.fontsize': FONT_SIZE_GLOBAL - 1
})

# ==========================================
# 1. 設定項目
# ==========================================
base_dir = r'C:\Users\admin\Desktop\4年\本実験\ASIN\camera'
spec_file_name = 'ASIN+spec.xlsx'
trends_file_name = 'trends_2023_2025.csv'
trends_2012_file = 'trends_ca_2012.csv'

asins = ["B00IDV4ACC", "B00S7LBA08", "B00XXCOJLS", "B0B1LVFFB5", "B01ALAZOOK",
         "B01FH4H5LA", "B01LXIND6Y", "B07FBJ9WNB", "B07G28MFV5", "B07GZGNTXZ",
         "B07JNL93SQ", "B07NVXM29C", "B07SLXZCBW", "B07V6DHHRX", "B07VYHDDF9",
         "B08N4FKFCG", "B09NDF5BN7", "B09ZRN1N3Z", "B072J97HT1", "B083K3SHCK",
         "B097C55DCP", "B099RKG1FN", "B007410K80"]

target_lags = [1]
lag_vars = ['ln_y_lag1']
inter_vars_m1 = ['inter_res_season', 'inter_vol_season', 'inter_pri_season', 'inter_zoo_season']
inter_vars_m2 = ['search_2012_impact']
category_var = ['ln_z_category']
eps = 1e-6

# ==========================================
# 2. 共通データの読み込み
# ==========================================
trends_raw = pd.read_csv(os.path.join(base_dir, trends_file_name), encoding='cp932')
trends_df = pd.DataFrame({
    'date': pd.to_datetime(trends_raw.iloc[:, 0]),
    'z_res': trends_raw.iloc[:, 1], 'z_vol': trends_raw.iloc[:, 2],
    'z_pri': trends_raw.iloc[:, 3], 'z_zoo': trends_raw.iloc[:, 4],
    'z_cat': trends_raw.iloc[:, 6]
})

t12_raw = pd.read_csv(os.path.join(base_dir, trends_2012_file), encoding='cp932')
trends_12_df = pd.DataFrame({
    'date': pd.to_datetime(t12_raw.iloc[:, 0]), 
    'search_b': t12_raw.iloc[:, 1], 'seasonal_d': t12_raw.iloc[:, 3]
})

specs_master = pd.read_excel(os.path.join(base_dir, spec_file_name), usecols=[0, 2, 3, 4, 5])
specs_master.columns = ['asin', 'f_res', 'f_vol', 'f_pri', 'f_zoo']

# ==========================================
# 3. 解析用関数
# ==========================================
def get_top_10(df, year):
    return df[df['date'].dt.year == year].groupby('asin')['q_it'].mean().nlargest(10).index.tolist()

def run_experiment(full_df):
    top_10_2023 = get_top_10(full_df, 2023)
    top_10_2024 = get_top_10(full_df, 2024)
    top_10_2025 = get_top_10(full_df, 2025)
    
    def run_session(train_years, test_year, train_asins, test_asins):
        target_vars = lag_vars + category_var + inter_vars_m1 + inter_vars_m2
        train_df = full_df[full_df['date'].dt.year.isin(train_years) & full_df['asin'].isin(train_asins)].copy()
        test_df = full_df[(full_df['date'].dt.year == test_year) & full_df['asin'].isin(test_asins)].copy()
        
        sc = StandardScaler()
        train_df[target_vars] = sc.fit_transform(train_df[target_vars])
        test_df[target_vars] = sc.transform(test_df[target_vars])
        
        dummies = pd.get_dummies(train_df['asin'], prefix='d', drop_first=True).astype(float)
        train_df = pd.concat([train_df, dummies], axis=1)
        d_cols = dummies.columns.tolist()
        
        models = {'M0': lag_vars + category_var + d_cols,
                  'M1': lag_vars + category_var + inter_vars_m1 + d_cols,
                  'M2': lag_vars + category_var + inter_vars_m2 + d_cols}
        
        res = {}
        for name, cols in models.items():
            model_fit = sm.OLS(train_df['ln_y'], sm.add_constant(train_df[cols])).fit()
            exog_test = sm.add_constant(test_df, has_constant='add')
            for c in model_fit.params.index:
                if c not in exog_test.columns: exog_test[c] = 0
            preds = model_fit.predict(exog_test[model_fit.params.index])
            res[name] = mean_absolute_error(test_df['ln_y'], preds)
        return res

    res24 = run_session([2023], 2024, top_10_2023, top_10_2024)
    res25 = run_session([2023, 2024], 2025, list(set(top_10_2023 + top_10_2024)), top_10_2025)
    return res24, res25

# ==========================================
# 4. 週次・月次の一括実行
# ==========================================
final_comparison = []

for freq in ['W', 'MS']:
    freq_name = "週次" if freq == 'W' else "月次"
    all_data_list = []
    
    for asin in asins:
        file_path = os.path.join(base_dir, f'salesrank_daily_2023_2025_{asin}.csv')
        if not os.path.exists(file_path): continue
        
        df_d = pd.merge(pd.read_csv(file_path).assign(date=lambda x: pd.to_datetime(x['date'])), trends_df, on='date', how='left')
        df_d = pd.merge(df_d, trends_12_df, on='date', how='left').set_index('date').sort_index()
        df_d[df_d.select_dtypes(include=[np.number]).columns] = df_d.select_dtypes(include=[np.number]).interpolate(limit_direction='both')
        df_d['q_it'] = 7.831 * (df_d['daily_salesrank'].clip(lower=1) ** (-0.713))
        
        # 集約処理
        resampled = pd.DataFrame()
        resampled['q_it'] = df_d['q_it'].resample(freq).sum()
        for col in ['z_res', 'z_vol', 'z_pri', 'z_zoo', 'z_cat', 'search_b', 'seasonal_d']:
            resampled[col] = df_d[col].resample(freq).mean()
        
        resampled['ln_y'] = np.log(resampled['q_it'] + eps)
        resampled['ln_y_lag1'] = resampled['ln_y'].shift(1)
        resampled['ln_z_category'] = np.log(resampled['z_cat'] + eps)
        
        spec = specs_master[specs_master['asin'] == asin].iloc[0] if asin in specs_master['asin'].values else None
        if spec is not None:
            for k in ['res', 'vol', 'pri', 'zoo']:
                resampled[f'inter_{k}_season'] = np.log(pd.to_numeric(spec[f'f_{k}']) + eps) * np.log(resampled[f'z_{k}'] + eps) * resampled['ln_z_category']
        resampled['search_2012_impact'] = np.log(resampled['search_b'] + eps) * np.log(resampled['seasonal_d'] + eps)
        resampled['asin'] = asin
        all_data_list.append(resampled.reset_index())

    full_df = pd.concat(all_data_list).dropna()
    res24, res25 = run_experiment(full_df)
    
    for year, res in [("2024", res24), ("2025", res25)]:
        final_comparison.append({
            '分析単位': freq_name, '対象年': year,
            'M0_MAE': res['M0'], 'M1_MAE': res['M1'], 'M2_MAE': res['M2'],
            'M1改善率(%)': (1 - res['M1']/res['M0']) * 100,
            'M2改善率(%)': (1 - res['M2']/res['M0']) * 100
        })

# ==========================================
# 5. 結果表示と可視化
# ==========================================
summary_df = pd.DataFrame(final_comparison)
print("\n===== 精度比較サマリー (Lag1) =====")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# グラフ描画
# インデックスを「分析単位」と「対象年」のマルチインデックスにしてプロット
plot_data = summary_df.set_index(['分析単位', '対象年'])[['M1改善率(%)', 'M2改善率(%)']]

ax = plot_data.plot(kind='bar', figsize=(12, 7), width=0.7)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('分析単位別のモデル改善率比較 (週次 vs 月次)')
plt.ylabel('M0に対する精度改善率 (%)')
plt.xlabel('分析の粒度と対象年')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.legend(title='モデル種類')

plt.tight_layout()
plt.show()
