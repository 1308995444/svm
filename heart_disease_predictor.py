import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 模型加载
try:
    model = joblib.load('svm.pkl')
except Exception as e:
    st.error(f"模型加载失败: {str(e)}")
    st.stop()

# 原始特征定义
feature_ranges = {
    'gender': {"type": "categorical", "options": [1, 2], "desc": "性别 (1:男 2:女)"},
    'srh': {"type": "categorical", "options": [1,2,3,4,5], "desc": "自评健康 (1-5)"},
    # ...（其他特征定义保持不变）
}

# 输入界面
feature_values = []
for feature, prop in feature_ranges.items():
    if prop["type"] == "numerical":
        value = st.number_input(
            prop["desc"],
            min_value=float(prop["min"]),
            max_value=float(prop["max"]),
            value=float(prop["default"]),
            step=prop.get("step", 1.0),
            key=feature
        )
    else:
        value = st.selectbox(
            prop["desc"],
            options=prop["options"],
            key=feature
        )
    feature_values.append(value)

# 预测执行
if st.button("预测"):
    features = np.array([feature_values])
    
    try:
        # 预测结果
        pred = model.predict(features)[0]
        try:
            proba = model.predict_proba(features)[0][pred]
            confidence = f"{proba*100:.1f}%"
        except:
            decision = model.decision_function(features)[0]
            confidence = f"决策值: {decision:.2f}"
        
        st.write(f"预测结果: {'高风险' if pred == 1 else '低风险'}")
        st.write(f"置信度: {confidence}")
        
        # SHAP解释（动态调整版）
        st.subheader("特征影响分析")
        try:
            # 1. 创建解释器
            background = np.tile(features.mean(axis=0), (10, 1))
            explainer = shap.KernelExplainer(
                model.predict, 
                background,
                feature_names=list(feature_ranges.keys())
            )
            
            # 2. 计算SHAP值
            shap_values = explainer.shap_values(features)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 取正类SHAP值
            sv = shap_values[0]  # 获取当前样本的SHAP值
            
            # 3. 动态计算坐标范围（核心逻辑）
            max_abs_impact = max(0.2, np.abs(sv).max() * 1.2)  # 保证最小0.2范围，留20%余量
            tick_interval = max(0.05, round(max_abs_impact/4, 2))  # 动态刻度间隔
            
            # 4. 创建图形
            plt.figure(figsize=(12, 4))
            force_plot = shap.force_plot(
                explainer.expected_value,
                sv,
                features[0],
                feature_names=list(feature_ranges.keys()),
                matplotlib=True,
                show=False
            )
            
            # 5. 动态调整图形属性
            ax = plt.gca()
            ax.set_xlim(-max_abs_impact, max_abs_impact)
            
            # 设置专业刻度（保证刻度数为奇数，包含0）
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5]))
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # 添加参考线
            ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
            
            # 6. 显示图形
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close()
            
            # 显示当前范围信息
            st.caption(f"动态坐标范围: [-{max_abs_impact:.2f}, {max_abs_impact:.2f}]")
            
        except Exception as e:
            st.warning(f"SHAP解释生成失败: {str(e)}")
            
    except Exception as e:
        st.error(f"预测错误: {str(e)}")
