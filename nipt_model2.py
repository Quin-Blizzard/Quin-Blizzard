#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型二：最优分箱 + 分位数稳健时点（BMI→达标时间的分位函数）

脚本目的：
1) 构造“最早达标时间 T*”的区间删失数据（若无外部CSV则合成数据）；
2) 使用 Weibull AFT 区间删失模型拟合 T* 与 BMI 的条件分布；
3) 在给定可靠度 pi0 下，得到 BMI→达标时间分位曲线，并做单调回归（isotonic）；
4) 以可靠度约束为前提，对 BMI 进行最优（简单）分箱，输出每组的推荐时点 t_j；
5) 生成高质量可视化：分位曲线 + 分组阶梯、以及 K 选择的代价曲线。

说明：
- 代码遵循“朴素直观”的脚本风格，避免 OOP 和复杂封装；
- 各段均含中文注释，便于阅读、复用与复制；
- 依赖：numpy、pandas、matplotlib、seaborn、scikit-learn、lifelines。
"""

# ========================
# 导入依赖与基础设置
# ========================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# 设置 Matplotlib 与 Seaborn 的全局风格，使图表美观、清晰
sns.set_theme(context="talk", style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# ========================
# 超参数与风险函数设定（符合模型二假设）
# ========================
# 可靠度阈值：组内平均达标比例需≥ pi0（对应 F(T<=t)≥pi0）
pi0 = 0.9

# 分组复杂度惩罚系数（越大越偏向较少的组数）
gamma_penalty = 0.15

# 风险函数 r(t)：反映不同发现时点的潜在风险（早/中/晚）
# 这里使用分段常数近似：早期（<=12）=1，中期（13-27）=3，晚期（>=28）=5
def r_of_t(t):
    if t <= 12:
        return 1.0
    elif t < 28:
        return 3.0
    else:
        return 5.0


# ========================
# 数据读取或合成：构造区间删失的 T*（首次达标时间）
# ========================
# 说明：若 /workspace/nipt_data.csv 存在且包含 BMI、可能的多次检测记录，
# 则可在此处改写为读取并构造区间删失的逻辑。为可直接运行，本脚本默认合成数据。

np.random.seed(42)

# 合成样本量（孕妇数，男胎）
n_subjects = 800

# 合成 BMI（偏重高 BMI 人群）
bmi = np.clip(np.random.normal(loc=30.0, scale=4.0, size=n_subjects), 18, 45)

# 合成年龄（可作为协变量，但模型二强调 BMI 主导）
age = np.clip(np.random.normal(loc=30.5, scale=4.5, size=n_subjects), 18, 45)

# 合成“真实”最早达标时间 T*：随 BMI 单调不减（临床上 BMI 高者达标更晚）
# 这里用线性趋势 + 随机扰动，并截断到合理孕周区间
eps = np.random.normal(loc=0.0, scale=1.0, size=n_subjects)
T_star_true = 10.5 + 0.22 * (bmi - 25.0) + 0.4 * (age - 30.0) / 10.0 + eps
T_star_true = np.clip(T_star_true, 9.0, 24.0)

# 合成每个个体的检测次数与检测时点（更贴近实际流程）
min_tests, max_tests = 1, 3
test_counts = np.random.choice([1, 2, 3], size=n_subjects, p=[0.25, 0.5, 0.25])

test_times_list = []
for i in range(n_subjects):
    k = test_counts[i]
    # 第一检多在 9~13 周，之后每次+1~3周
    t1 = np.random.uniform(9.0, 13.0)
    times = [t1]
    for _ in range(k - 1):
        times.append(times[-1] + np.random.uniform(1.0, 3.5))
    test_times_list.append(np.array(times))

# 根据 T* 生成观测（是否达标）并构造区间删失 [lower, upper]
# 若有检测阳性（达标），取最早阳性时间为上界；若之前有阴性，最后阴性为下界
# 情况：
# - 仅阴性 -> 右删失：lower = 最后阴性时间, upper = +inf
# - 首检阳性 -> 左删失：lower = 0, upper = 首检阳性时间
# - 阴后阳 -> 区间删失：lower = 最后阴性时间, upper = 首阳性时间

lower_bounds = []
upper_bounds = []

for i in range(n_subjects):
    times = test_times_list[i]
    Tstar = T_star_true[i]
    # 生成是否达标（忽略测量误差，以符合模型二“事件时间”设定）
    is_pos = times >= Tstar
    if is_pos.any():
        first_pos_time = times[is_pos][0]
        neg_times = times[times < first_pos_time]
        if len(neg_times) == 0:
            # 左删失
            lower_bounds.append(0.0)
            upper_bounds.append(first_pos_time)
        else:
            # 区间删失
            lower_bounds.append(neg_times.max())
            upper_bounds.append(first_pos_time)
    else:
        # 右删失
        lower_bounds.append(times.max())
        upper_bounds.append(np.inf)

shuju = pd.DataFrame({
    "BMI": bmi,
    "age": age,
    "lower": np.array(lower_bounds, dtype=float),
    "upper": np.array(upper_bounds, dtype=float),
})

# 记录用于可视化的观测区间（中点用于散点参考）
mid_obs = shuju[["lower", "upper"]].copy()
mid_obs["upper_finite"] = np.isfinite(mid_obs["upper"]).astype(int)
mid_obs["mid"] = np.where(np.isfinite(mid_obs["upper"]),
                           0.5 * (mid_obs["lower"] + mid_obs["upper"]),
                           mid_obs["lower"] + 1.0)  # 右删失样本中点近似


# ========================
# 拟合 Weibull AFT 区间删失模型（lifelines）
# ========================
from lifelines import WeibullAFTFitter

# 仅使用 BMI（与模型二一致），age 可作为稳健性扩展，这里保留列但不必入模
aft = WeibullAFTFitter()

# lifelines 提供 fit_interval_censoring 来直接拟合区间删失数据
# 该函数需要通过参数 lower_bound_col / upper_bound_col 指明列名
# 将 BMI 与删失边界放在一个 DataFrame 中传入
df_fit = shuju[["BMI", "lower", "upper"]].copy()
aft.fit_interval_censoring(df_fit, lower_bound_col="lower", upper_bound_col="upper", show_progress=False, formula="BMI")


# ========================
# 计算 BMI→分位数达标时间曲线，并做单调回归
# ========================
# 目标：对给定 BMI b，求 t*(b)=最小 t 使得 F(T<=t|b)≥pi0
# 在 AFT 模型下：predict_percentile(X, p_surv) 返回 S(t|X)=p_surv 的时间
# 因 F=1-S，故 p_surv = 1 - pi0

p_surv = 1.0 - pi0

# 在 BMI 网格上计算分位曲线（使用样本 BMI 范围）
bmi_min, bmi_max = float(shuju["BMI"].min()), float(shuju["BMI"].max())
bmi_grid = np.linspace(bmi_min, bmi_max, 120)

# lifelines 接口需要 DataFrame 作为协变量输入
grid_df = pd.DataFrame({"BMI": bmi_grid})
t_quant_grid = aft.predict_percentile(grid_df, p=p_surv).values.flatten()

# 对个体也计算各自的分位数时间（用于单调回归拟合 y=f(BMI)）
t_quant_indiv = aft.predict_percentile(shuju[["BMI"]], p=p_surv).values.flatten()

# 单调回归：保证 t*(b) 随 BMI 非减（符合模型二“单调性”假设）
iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
t_quant_iso = iso.fit_transform(shuju["BMI"].values, t_quant_indiv)

# 在网格上得到单调回归后的平滑曲线
t_quant_iso_grid = iso.predict(bmi_grid)


# ========================
# 在可靠度约束下进行 BMI 分箱，并为每组求最小可行时点 t_j
# ========================
# 方法：
# - 对候选 K（组数）遍历（如 2~6）；
# - 采用等频分箱作为初始简单近似；
# - 对每个组，使用 AFT 预测的个体生存函数 S_i(t)，用网格搜索最小 t，使组内平均 CDF=1-平均S >= pi0；
# - 计算总成本 sum_j(weighted r(t_j)) + gamma*K，选择代价最小的 K；

def compute_group_time_under_pi0(aft_model, X_group, times_grid, pi):
    """
    给定某组的协变量 X_group（仅 BMI），以及时间网格 times_grid，
    计算满足组内平均 CDF >= pi 的最小 t（若全不达标则取最大时间）。
    """
    # 预测该组每个个体在 times_grid 上的生存函数 S(t)
    S_df = aft_model.predict_survival_function(X_group, times=times_grid)
    # S_df: index=times, columns=每个样本
    S_mean = S_df.values.mean(axis=1)  # 组内平均 S(t)
    F_mean = 1.0 - S_mean
    idx = np.argmax(F_mean >= pi)
    if F_mean[idx] >= pi:
        return float(times_grid[idx])
    else:
        return float(times_grid[-1])


# 构建时间网格（覆盖主要达标区间，留有余量）
tmin_guess = max(8.0, float(np.nanmin(t_quant_grid)) - 2.0)
tmax_guess = min(32.0, float(np.nanmax(t_quant_grid)) + 4.0)
times_grid = np.linspace(tmin_guess, tmax_guess, 220)

# 候选组数范围
K_candidates = list(range(2, 7))

best_K = None
best_cost = np.inf
best_edges = None
best_tj = None
cost_list = []

for K in K_candidates:
    # 等频分箱的 BMI 边界
    quantiles = np.linspace(0, 1, K + 1)
    edges = np.quantile(shuju["BMI"].values, quantiles)
    # 修正以避免重复边界导致空组
    edges = np.unique(edges)
    if len(edges) - 1 < K:
        # 如果去重后组数不足，则跳过该 K
        continue

    # 按边界分组
    groups = []
    t_j_list = []
    weight_list = []

    for j in range(len(edges) - 1):
        left, right = edges[j], edges[j + 1]
        mask = (shuju["BMI"] >= left) & (shuju["BMI"] < right if j < len(edges)-2 else shuju["BMI"] <= right)
        X_group = shuju.loc[mask, ["BMI"]]
        if len(X_group) == 0:
            continue
        # 求该组最小可行 t_j（满足平均 CDF>=pi0）
        t_j = compute_group_time_under_pi0(aft, X_group, times_grid, pi0)
        groups.append((left, right))
        t_j_list.append(t_j)
        weight_list.append(len(X_group) / len(shuju))

    # 计算总代价：加权 r(t_j) + gamma*K
    group_risks = [r_of_t(tj) for tj in t_j_list]
    total_cost = float(np.sum(np.array(group_risks) * np.array(weight_list)) + gamma_penalty * K)
    cost_list.append((K, total_cost))

    if total_cost < best_cost:
        best_cost = total_cost
        best_K = K
        best_edges = np.array([g[0] for g in groups] + [groups[-1][1]])
        best_tj = np.array(t_j_list)


# ========================
# 输出与可视化
# ========================
print("\n================ 结果摘要（模型二） ================")
print(f"可靠度阈值 pi0 = {pi0:.2f}")
print(f"复杂度惩罚 gamma = {gamma_penalty:.2f}")
print(f"最优组数 K* = {best_K}")
print(f"总代价（加权风险+惩罚） = {best_cost:.3f}")
print("\nBMI 分组区间与推荐时点（周）：")
for j in range(len(best_edges) - 1):
    left, right = best_edges[j], best_edges[j + 1]
    tj = best_tj[j]
    print(f"第{j+1}组: [{left:.1f}, {right:.1f}] → t_j = {tj:.2f} 周")


# 图1：BMI-达标时间分位曲线（AFT 预测）+ 单调回归 + 分组阶梯
fig, ax = plt.subplots(1, 1, figsize=(9, 6))

# 背景散点：用观测区间中点作为参考（仅作可视化，不参与拟合）
ax.scatter(shuju["BMI"].values, mid_obs["mid"].values,
           s=12, alpha=0.35, color="#9ecae1", label="观测区间中点（参考）")

# AFT 直接分位曲线
ax.plot(bmi_grid, t_quant_grid, color="#2ca25f", lw=2.2, label=f"AFT 分位曲线 (F≥{pi0:.0%})")

# 单调回归后的平滑曲线
ax.plot(bmi_grid, t_quant_iso_grid, color="#de2d26", lw=2.0, linestyle="--", label="单调回归后分位曲线")

# 分组阶梯：横线 + 竖线
for j in range(len(best_edges) - 1):
    left, right = best_edges[j], best_edges[j + 1]
    tj = best_tj[j]
    ax.hlines(y=tj, xmin=left, xmax=right, colors="#756bb1", lw=3.0)
    if j < len(best_edges) - 2:
        ax.vlines(x=right, ymin=min(t_quant_grid)-0.5, ymax=max(t_quant_grid)+0.5,
                  colors="#bdbdbd", linestyles=":", lw=1.5)

ax.set_title("BMI→最早达标时间（分位）与分组推荐时点")
ax.set_xlabel("BMI")
ax.set_ylabel("达标时间（孕周，周）")
ax.legend(loc="best")

plt.tight_layout()
fig_path1 = "/workspace/nipt_model2_fig_curve.png"
plt.savefig(fig_path1)
print(f"\n已保存图像：{fig_path1}")


# 图2：K 候选的总代价曲线（用于选择最优组数）
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
if len(cost_list) > 0:
    K_vals = [k for k, c in cost_list]
    costs = [c for k, c in cost_list]
    ax2.plot(K_vals, costs, marker="o", color="#1f77b4", lw=2.2)
    ax2.set_xticks(K_vals)
    ax2.set_title("分组数 K 的总代价曲线")
    ax2.set_xlabel("K（组数）")
    ax2.set_ylabel("总代价：加权 r(t_j) + gamma*K")
    ax2.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig_path2 = "/workspace/nipt_model2_fig_cost.png"
    plt.savefig(fig_path2)
    print(f"已保存图像：{fig_path2}")
else:
    ax2.text(0.5, 0.5, "无有效 K 结果", ha="center", va="center")
    plt.tight_layout()
    fig_path2 = "/workspace/nipt_model2_fig_cost.png"
    plt.savefig(fig_path2)
    print(f"已保存图像：{fig_path2}")


# 结束
print("\n脚本执行完毕。")

