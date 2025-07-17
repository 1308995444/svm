import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 模型加载 - 替换为SVM模型
model = joblib.load('svm.pkl')  # 确保这是您训练好的SVM模型文件

# 特征定义（完全保留原始定义）
feature_ranges = {
    'gender': {"type": "categorical", "options": [1, 2], "desc": "性别/Gender (1:男/Male, 2:女/Female)"},
    'srh': {"type": "categorical", "options": [1,2,3,4,5], "desc": "自评健康/Self-rated health (1-5: 很差/Very poor 到 很好/Very good)"},
    'adlab_c': {"type": "categorical", "options": [0,1,2,3,4,5,6], "desc": "日常活动能力/Activities of daily living (0-6: 无/None 到 完全依赖/Complete dependence)"},
    'arthre': {"type": "categorical", "options": [0, 1], "desc": "关节炎/Arthritis (0:无/No, 1:有/Yes)"},
    'digeste': {"type": "categorical", "options": [0, 1], "desc": "消化系统问题/Digestive issues (0:无/No, 1:有/Yes)"},
    'retire': {"type": "categorical", "options": [0, 1], "desc": "退休状态/Retirement status (0:未退休/Not retired, 1:已退休/Retired)"},
    'satlife': {"type": "categorical", "options": [1,2,3,4,5], "desc": "生活满意度/Life satisfaction (1-5: 非常不满意/Very dissatisfied 到 非常满意/Very satisfied)"},
    'sleep': {
        "type": "numerical", 
        "min": 0.0, 
        "max": 24.0, 
        "default": 8.0, 
        "desc": "睡眠时长/Sleep duration (小时/hours)",
        "step": 0.1,
        "format": "%.1f"
    },
    'disability': {"type": "categorical", "options": [0, 1], "desc": "残疾/Disability (0:无/No, 1:有/Yes)"},
    'internet': {"type": "categorical", "options": [0, 1], "desc": "互联网使用/Internet use (0:不使用/No, 1:使用/Yes)"},
    'hope': {"type": "categorical", "options": [1,2,3,4], "desc": "希望程度/Hope level (1-4: 很低/Very low 到 很高/Very high)"},
    'fall_down': {"type": "categorical", "options": [0, 1], "desc": "跌倒史/Fall history (0:无/No, 1:有/Yes)"},
    'eyesight_close': {"type": "categorical", "options": [1,2,3,4,5], "desc": "视力/Near vision (1-5: 很差/Very poor 到 很好/Very good)"},
    'hear': {"type": "categorical", "options": [1,2,3,4,5], "desc": "听力/Hearing (1-5: 很差/Very poor 到 很好/Very good)"},
    'edu': {"type": "categorical", "options": [1,2,3,4], "desc": "教育程度/Education level (1:小学以下/Below Primary, 2:小学/Primary, 3:中学/Secondary, 4:中学以上/Above Secondary)"},
    'pension': {"type": "categorical", "options": [0, 1], "desc": "养老保险/Pension (0:无/No, 1:有/Yes)"},
    'pain': {"type": "categorical", "options": [0, 1], "desc": "慢性疼痛/Chronic pain (0:无/No, 1:有/Yes)"}
}

# 界面布局（完全保留原始流程）
st.title("Depression Risk-Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

# 输入表单（完全保留）
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=properties["desc"],
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
            step=properties.get("step", 1.0),
            format=properties.get("format", "%f"),
            key=f"num_{feature}"
        )
    else:
        value = st.selectbox(
            label=properties["desc"],
            options=properties["options"],
            key=f"cat_{feature}"
        )
    feature_values.append(value)

features = np.array([feature_values])

# 预测与解释（适配SVM模型）
if st.button("Predict"):
    # SVM预测
    predicted_class = model.predict(features)[0]
    
    # 获取概率或决策分数
    try:
        predicted_proba = model.predict_proba(features)[0]
        probability = predicted_proba[predicted_class] * 100
        prob_text = f"Predicted probability: {probability:.2f}%"
    except AttributeError:
        decision_score = model.decision_function(features)[0]
        prob_text = f"Decision score: {decision_score:.2f}"
    
    # 结果显示（完全保留原始格式）
    text_en = f"{prob_text} ({'High risk' if predicted_class == 1 else 'Low risk'})"
    fig, ax = plt.subplots(figsize=(10,2))
    ax.text(0.5, 0.7, text_en, 
            fontsize=14, ha='center', va='center', fontname='Arial')
    ax.axis('off')
    st.pyplot(fig)

    # SHAP解释（适配SVM的KernelExplainer）
    # 替换原SHAP解释部分为以下代码：

# SHAP解释（SVM专用版）
try:
    # 1. 创建适合SVM的解释器
    explainer = shap.KernelExplainer(
        model.predict, 
        shap.sample(features, 5),  # 小样本背景数据
        feature_names=list(feature_ranges.keys())
    )
    
    # 2. 计算SHAP值（确保输出维度正确）
    shap_values = explainer.shap_values(features)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # 二分类取正类SHAP值
    
    # 3. 规范化图形输出
    plt.figure(figsize=(12, 3))
    shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values[0],
        features=features[0],
        feature_names=list(feature_ranges.keys()),
        matplotlib=True,
        show=False,
        plot_cmap="coolwarm"  # 明确指定颜色映射
    )
    
    # 4. 强制设置合理坐标轴范围
    ax = plt.gca()
    ax.set_xlim(-1, 1)  # 固定SHAP值显示范围
    plt.xticks(fontsize=10)
    plt.tight_layout()
    
    # 5. 显示图形
    st.pyplot(plt.gcf(), clear_figure=True)
    plt.close()
    
except Exception as e:
    st.warning(f"SHAP解释生成失败: {str(e)}")
    st.info("建议：检查模型预测输出是否在[0,1]范围内")
