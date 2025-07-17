import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

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

# 输入界面（保持不变）
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
        # 预测结果（保持不变）
        pred = model.predict(features)[0]
        try:
            proba = model.predict_proba(features)[0][pred]
            confidence = f"{proba*100:.1f}%"
        except:
            decision = model.decision_function(features)[0]
            confidence = f"决策值: {decision:.2f}"
        
        st.write(f"预测结果: {'高风险' if pred == 1 else '低风险'}")
        st.write(f"置信度: {confidence}")
        
        # SHAP解释（关键修改部分）
        st.subheader("特征影响")
        try:
            # 1. 创建解释器
            background = shap.utils.sample(features, 10) if features.shape[0] > 10 else features
            explainer = shap.KernelExplainer(
                model.predict, 
                background,
                feature_names=list(feature_ranges.keys())
            )
            
            # 2. 计算SHAP值
            shap_values = explainer.shap_values(features)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 取正类SHAP值
            
            # 3. 创建自定义力导向图（核心修改）
            plt.figure(figsize=(12, 3))
            force_plot = shap.force_plot(
                base_value=explainer.expected_value,
                shap_values=shap_values[0],
                features=features[0],
                feature_names=list(feature_ranges.keys()),
                matplotlib=True,
                show=False
            )
            
            # 4. 强制设置坐标轴范围
            ax = plt.gca()
            ax.set_xlim(-0.2, 0.2)  # 关键修改：限制x轴范围
            
            # 5. 显示图形
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close()
            
        except Exception as e:
            st.warning(f"SHAP解释生成失败: {str(e)}")
            
    except Exception as e:
        st.error(f"预测错误: {str(e)}")
