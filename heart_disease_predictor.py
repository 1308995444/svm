import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 模型加载 - Now loading SVM model
model = joblib.load('svm.pkl')

# 特征定义 (same as before)
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

# 界面布局
st.title("Depression Risk-Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

# 输入表单
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

# 预测与解释
if st.button("Predict"):
    # SVM prediction
    predicted_class = model.predict(features)[0]
    
    # For SVM, we need decision function or predict_proba if available
    try:
        # Try to get probabilities first (if SVM has probability=True)
        predicted_proba = model.predict_proba(features)[0]
        probability = predicted_proba[predicted_class] * 100
        prob_text = f"Predicted probability: {probability:.2f}%"
    except AttributeError:
        # Fall back to decision function if predict_proba not available
        decision_score = model.decision_function(features)[0]
        prob_text = f"Decision score: {decision_score:.2f}"
    
    risk_text = f"({'High risk' if predicted_class == 1 else 'Low risk'})"
    
    # 结果显示
    text_en = f"{prob_text} {risk_text}"
    fig, ax = plt.subplots(figsize=(10,2))
    ax.text(0.5, 0.7, text_en, 
            fontsize=14, ha='center', va='center', fontname='Arial')
    ax.axis('off')
    st.pyplot(fig)

    # SHAP解释 - Using KernelExplainer for SVM
   # 在SHAP可视化部分替换为以下代码：

# SHAP解释 - 使用更稳定的可视化参数
st.subheader("SHAP Explanation")

# 创建解释器（使用训练数据样本作为背景更佳）
try:
    # 如果有训练数据，推荐这样初始化（示例）
    # background = shap.sample(X_train, 100) 
    # 没有训练数据时使用均值作为背景
    background = shap.utils.sample(features, 10) if features.shape[0] > 10 else features
    
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(features)
    
    # 创建更清晰的force plot
    plt.figure(figsize=(12, 3))
    force_plot = shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values[0] if isinstance(shap_values, list) else shap_values,
        features=features,
        feature_names=list(feature_ranges.keys()),
        matplotlib=True,
        show=False,
        plot_cmap="coolwarm"  # 使用更清晰的配色
    )
    
    # 调整图形显示
    plt.tight_layout()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    st.pyplot(plt.gcf())
    plt.close()
    
except Exception as e:
    st.error(f"SHAP visualization error: {str(e)}")
    st.warning("Please ensure your model supports SHAP explanation and input data is valid.")
