import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor

from skopt import gp_minimize
from skopt.space import Real

st.title("发光材料智能优化系统（贝叶斯 + 多目标）")

# 上传数据
file = st.file_uploader("上传CSV数据", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df)

    features = st.multiselect("选择输入特征", df.columns)
    
    target1 = st.selectbox("目标1（如ML）", df.columns)
    target2 = st.selectbox("目标2（如LPL）", df.columns)

    if len(features) > 0:

        X = df[features].values
        y1 = df[target1].values
        y2 = df[target2].values

        # 模型
        model1 = GaussianProcessRegressor()
        model2 = GaussianProcessRegressor()

        model1.fit(X, y1)
        model2.fit(X, y2)

        st.success("模型训练完成（GPR）")

        # 定义搜索空间
        space = []
        for col in features:
            space.append(Real(df[col].min(), df[col].max(), name=col))

        # 🎯 多目标合并（加权）
        w1 = st.slider("目标1权重", 0.0, 1.0, 0.5)
        w2 = 1 - w1

        def objective(params):
            params = np.array(params).reshape(1, -1)

            pred1, std1 = model1.predict(params, return_std=True)
            pred2, std2 = model2.predict(params, return_std=True)

            # 贝叶斯思想：考虑不确定性
            score = w1 * (pred1 + std1) + w2 * (pred2 + std2)

            return -score[0]  # 最小化

        st.subheader("贝叶斯优化中...")

        res = gp_minimize(objective, space, n_calls=30, random_state=0)

        best_params = res.x

        st.subheader("最优参数（贝叶斯）")

        result_df = pd.DataFrame([best_params], columns=features)
        st.dataframe(result_df)

        # 预测最优值
        best_params_np = np.array(best_params).reshape(1, -1)
        pred1 = model1.predict(best_params_np)[0]
        pred2 = model2.predict(best_params_np)[0]

        st.write(f"{target1} 预测最优: {pred1:.3f}")
        st.write(f"{target2} 预测最优: {pred2:.3f}")

        # 🔥 Pareto前沿（随机采样）
        st.subheader("Pareto前沿")

        samples = 2000
        rand_X = np.random.uniform(
            [df[col].min() for col in features],
            [df[col].max() for col in features],
            (samples, len(features))
        )

        p1 = model1.predict(rand_X)
        p2 = model2.predict(rand_X)

        # Pareto筛选
        pareto = []
        for i in range(len(rand_X)):
            dominated = False
            for j in range(len(rand_X)):
                if (p1[j] >= p1[i] and p2[j] >= p2[i]) and (j != i):
                    dominated = True
                    break
            if not dominated:
                pareto.append(i)

        pareto_x = p1[pareto]
        pareto_y = p2[pareto]

        fig, ax = plt.subplots()
        ax.scatter(p1, p2, alpha=0.3, label="样本")
        ax.scatter(pareto_x, pareto_y, color='red', label="Pareto前沿")

        ax.set_xlabel(target1)
        ax.set_ylabel(target2)
        ax.legend()

        st.pyplot(fig)