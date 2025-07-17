import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. 模型加载
try:
    model = joblib.load('svm.pkl')
except Exception as e:
    st.error(f"模型加载失败: {str(e)}")
    st.stop()

# 2. 原始特征定义（严格保留）
feature_ranges = {
    'gender': {"type": "categorical", "options": [1, 2], "desc": "性别 (1:男 2:女)"},
    'srh': {"type": "categorical", "options": [1,2,3,4,5], "desc": "自评健康 (1-5)"},
    'adlab_c': {"type": "categorical", "options": [0,1,2,3,4,5,6], "desc": "日常活动能力 (0-6)"},
    'arthre': {"type": "categorical", "options": [0, 1], "desc": "关节炎 (0:无 1:有)"},
    'digeste': {"type": "categorical", "options": [0, 1], "desc": "消化问题 (0:无 1:有)"},
    'retire': {"type": "categorical", "options": [0, 1], "desc": "退休状态 (0:未退 1:已退)"},
    'satlife': {"type": "categorical", "options": [1,2,3,4,5], "desc": "生活满意度 (1-5)"},
    'sleep': {"type": "numerical", "min": 0.0, "max": 24.0, "default": 7.0, "step": 0.5, "desc": "睡眠时长(小时)"},
    'disability': {"type": "categorical", "options": [0, 1], "desc": "残疾 (0:无 1:有)"},
    'internet': {"type": "categorical", "options": [0, 1], "desc": "上网 (0:否 1:是)"},
    'hope': {"type": "categorical", "options": [1,2,3,4], "desc": "希望程度 (1-4)"},
    'fall_down': {"type": "categorical", "options": [0, 1], "desc": "跌倒史 (0:无 1:有)"},
    'eyesight_close': {"type": "categorical", "options": [1,2,3,4,5], "desc": "视力 (1-5)"},
    'hear': {"type": "categorical", "options": [1,2,3,4,5], "desc": "听力 (1-5)"},
    'edu': {"type": "categorical", "options": [1,2,3,4], "desc": "教育程度 (1-4)"},
    'pension': {"type": "categorical", "options": [0, 1], "desc": "养老金 (0:无 1:有)"},
    'pain': {"type": "categorical", "options": [0, 1], "desc": "慢性疼痛 (0:无 1:有)"}
}

# 3. 输入界面（原始布局）
st.title("抑郁症风险预测")
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

# 4. 预测执行（原始流程）
if st.button("预测"):
    features = np.array([feature_values])
    
    try:
        # 原始预测逻辑
        pred = model.predict(features)[0]
        try:
            proba = model.predict_proba(features)[0][pred]
            confidence = f"{proba*100:.1f}%"
        except:
            decision = model.decision_function(features)[0]
            confidence = f"决策值: {decision:.2f}"
        
        # 原始结果显示方式
        st.write(f"预测结果: {'高风险' if pred == 1 else '低风险'}")
        st.write(f"置信度: {confidence}")
        
        # SHAP解释（修复空白问题）
        st.subheader("特征影响")
        try:
            # 修复1: 使用更稳定的背景数据生成方式
            background = np.tile(features.mean(axis=0), (10, 1))
            
            # 修复2: 明确指定feature_names
            explainer = shap.KernelExplainer(
                model.predict, 
                background,
                feature_names=list(feature_ranges.keys())
            )
            
            # 修复3: 确保shap_values维度正确
            shap_values = explainer.shap_values(features)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 取正类SHAP值
            
            # 修复4: 显式创建图形对象
            plt.figure(figsize=(12, 3))
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                features[0],
                feature_names=list(feature_ranges.keys()),
                matplotlib=True,
                show=False
            )
            st.pyplot(plt.gcf(), clear_figure=True)  # 修复5: 添加clear_figure
            plt.close()
            
        except Exception as e:
            st.warning(f"SHAP解释生成失败: {str(e)}")
            
    except Exception as e:
        st.error(f"预测错误: {str(e)}")
