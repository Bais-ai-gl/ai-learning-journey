# %% [说明] Day 2 Pandas 5 任务模板
# 用法：
#   1) 在 VSCode 打开本文件，每段 "# %%" 之间是一个单元格，点左侧 "Run Cell" 逐格运行
#   2) 或直接在终端运行整个文件：  py day2_pandas_tasks.py
# 数据说明：
#   默认用内置模拟数据（保证立刻能跑）。真实练习时，去 Kaggle 下载 Ames 房价 train.csv，
#   把下面"加载数据"单元格里 df = ... 换成 pd.read_csv("train.csv") 即可（任务1~3同样适用）。

# %% 加载数据（先装依赖：pip install pandas matplotlib）
import pandas as pd
import numpy as np

np.random.seed(42)
n = 200

# 主表：房屋成交记录（含一个"字符串格式的日期列"，供任务4练习解析）
df = pd.DataFrame({
    "Id": range(1, n + 1),
    "MSZoning": np.random.choice(["RL", "RM", "FV", "RH", "C"], n),   # 分区
    "SalePrice": np.random.randint(80000, 500000, n).astype(float),    # 房价
    "GrLivArea": np.random.randint(500, 4000, n),                      # 居住面积
    "YearBuilt": np.random.randint(1900, 2020, n),                     # 建造年份
    "NeighborhoodId": np.random.randint(1, 6, n),                      # 社区编号(供任务5合并)
})

# 故意注入一些缺失值（供任务2练习）
df.loc[df.sample(12, random_state=1).index, "GrLivArea"] = np.nan
df.loc[df.sample(6, random_state=2).index, "YearBuilt"] = np.nan

# 加一列"字符串日期"（注意：这里是 str，不是日期类型，任务4要解析它）
date_start = pd.Timestamp("2020-01-01")
df["SaleDate"] = (date_start + pd.to_timedelta(np.arange(n) * 7, unit="D")).strftime("%Y-%m-%d")

# 副表：社区信息（供任务5合并）
neighborhood = pd.DataFrame({
    "NeighborhoodId": [1, 2, 3, 4, 5],
    "NeighborhoodName": ["北区", "南区", "东区", "西区", "中心区"],
    "City": ["Ames"] * 5,
})

print("主表 df 形状：", df.shape)
print("副表 neighborhood 形状：", neighborhood.shape)

# %% 任务1：查看数据基本结构
print("=== shape / dtypes / describe ===")
print("行数×列数：", df.shape)
print("\n各列类型：")
print(df.dtypes)
print("\n数值列统计：")
print(df.describe().round(1))

# %% 任务2：缺失值统计 + 条形图
import matplotlib
import matplotlib.pyplot as plt

# 解决中文显示为方块的问题（Windows 用微软雅黑/黑体）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False   # 让负号正常显示

missing = df.isnull().sum()                 # 每列缺失值数量
missing = missing[missing > 0]              # 只看有缺失的列
print("各列缺失值数量：")
print(missing)

if len(missing) > 0:
    missing.plot(kind="bar", color="steelblue", figsize=(6, 3))
    plt.title("各列缺失值数量")
    plt.ylabel("缺失个数")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("missing_values.png", dpi=100)   # 图片保存到当前目录
    print("\n[已保存] 缺失值条形图 -> missing_values.png")
    plt.show()
else:
    print("（当前数据没有缺失值）")

# %% 任务3：按 MSZoning 分组求 SalePrice 均值
grouped = df.groupby("MSZoning")["SalePrice"].agg(["count", "mean", "max"]).round(0)
grouped = grouped.sort_values("mean", ascending=False)
print("按分区统计房价：")
print(grouped)

# %% 任务4：把字符串日期解析成 datetime 并提取年份/月份
print("解析前 SaleDate 类型：", df["SaleDate"].dtype)

df["SaleDate_dt"] = pd.to_datetime(df["SaleDate"])   # 字符串 -> 日期类型
df["SaleYear"] = df["SaleDate_dt"].dt.year           # 提取年份
df["SaleMonth"] = df["SaleDate_dt"].dt.month         # 提取月份

print("解析后类型：", df["SaleDate_dt"].dtype)
print("\n按年份统计成交笔数：")
print(df["SaleYear"].value_counts().sort_index())

# %% 任务5：两张表 merge
merged = df.merge(neighborhood, on="NeighborhoodId", how="left")
print("合并后形状：", merged.shape)
print("合并后前 5 行（关键列）：")
print(merged[["Id", "SalePrice", "NeighborhoodId", "NeighborhoodName", "City"]].head())

# 验证：按社区名统计平均房价
print("\n按社区名统计平均房价：")
print(merged.groupby("NeighborhoodName")["SalePrice"].mean().round(0).sort_values(ascending=False))

# %% 自测清单（全部能答上来 = Day 2 过关）
print("""
【Day 2 自测】
□ 能写出 df.shape / df.dtypes / df.describe 各自的作用
□ 能写出 df.isnull().sum() 统计缺失值
□ 能写出 groupby().agg() 分组聚合
□ 能写出 pd.to_datetime() 把字符串转日期、.dt.year 提取年份
□ 能写出 merge(on=..., how='left') 并说出 left/right/inner 区别
""")
