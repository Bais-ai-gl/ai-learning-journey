# Day 5：机器学习最小集 · 分类模型对比
# 用法：py -m pip install scikit-learn  然后  py 01_classification.py
# 用 sklearn 自带的乳腺癌数据集（无需手动下载）

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. 加载数据
data = load_breast_cancer()
X, y = data.data, data.target            # X 特征，y 标签（0=恶性 1=良性）
print("特征形状：", X.shape, " 标签取值：", set(y))

# 2. 划分训练/测试集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 定义两个模型做对比
models = {
    "逻辑回归": LogisticRegression(max_iter=10000),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
}

# 4. 训练 + 评估
for name, model in models.items():
    model.fit(X_train, y_train)                 # 训练
    y_pred = model.predict(X_test)              # 预测
    acc = accuracy_score(y_test, y_pred)        # 准确率
    print(f"\n===== {name} =====")
    print(f"准确率：{acc:.3f}")
    print("混淆矩阵（行=真实，列=预测）：")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=data.target_names))

# 思考题：精确率(precision)和召回率(recall)在这个"癌症检测"场景里，哪个更重要？为什么？
