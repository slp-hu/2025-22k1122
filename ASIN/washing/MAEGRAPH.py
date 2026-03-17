import matplotlib.pyplot as plt

# --- 設定：ここを調整してください ---
LABEL_SIZE = 20   # 軸ラベル (StartLag, AvgProposedMAE)
TITLE_SIZE = 22   # タイトル
TICK_SIZE = 30    # 軸の数値 (2, 3, 4...)
LEGEND_SIZE = 30  # 凡例の文字サイズ
LINE_WIDTH = 2    # 線の太さ
MARKER_SIZE = 8   # マーカーの大きさ
# ------------------------------

# データの準備
start_lags = list(range(2, 15))
mae_2024_all = [0.3314, 0.3550, 0.3708, 0.3845, 0.3975, 0.4123, 0.4312, 0.4443, 0.4533, 0.4599, 0.4642, 0.4693, 0.4826]
mae_2025_all = [0.2616, 0.2873, 0.3019, 0.3132, 0.3231, 0.3334, 0.3455, 0.3561, 0.3628, 0.3686, 0.3734, 0.3783, 0.3862]
mae_2024_washing = [0.2111, 0.2495, 0.2787, 0.3020, 0.3204, 0.3389, 0.3572, 0.3706, 0.3809, 0.3889, 0.3960, 0.4026, 0.4098]
mae_2025_washing = [0.2117, 0.2851, 0.3564, 0.4209, 0.4686, 0.5022, 0.5362, 0.5791, 0.6165, 0.6477, 0.6847, 0.7134, 0.7361]

plt.figure(figsize=(12, 8))

# 各ラインのプロット（linewidthとmarkersizeを追加）
plt.plot(start_lags, mae_2024_all, marker='o', label='2024 Digital Camera', color='blue', lw=LINE_WIDTH, ms=MARKER_SIZE)
plt.plot(start_lags, mae_2025_all, marker='s', label='2025 Digital Camera', color='skyblue', lw=LINE_WIDTH, ms=MARKER_SIZE)
plt.plot(start_lags, mae_2024_washing, marker='^', linestyle='--', label='2024 Washing Machine', color='green', lw=LINE_WIDTH, ms=MARKER_SIZE)
plt.plot(start_lags, mae_2025_washing, marker='v', linestyle='--', label='2025 Washing Machine', color='orange', lw=LINE_WIDTH, ms=MARKER_SIZE)

# 軸ラベルの設定
plt.xlabel('Start_Lags', fontsize=LABEL_SIZE)
plt.ylabel('Proposed_Model_MAE', fontsize=LABEL_SIZE)

# メモリ（Ticks）のサイズ変更
plt.xticks(start_lags, fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)

# 凡例（Legend）の設定
# prop={'size': ...} でサイズを指定、frameon=Trueで枠線を表示（論文向け）
plt.legend(fontsize=LEGEND_SIZE, loc='upper left', frameon=True, edgecolor='black')

# 補助線
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('mae_analysis_high_res.png', dpi=300)
plt.show()
