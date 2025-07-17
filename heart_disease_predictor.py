import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# --- 强制保留的原始特征变量 ---
feature_ranges = {
    'gender': {"type": "categorical", "options": [1, 2], "desc": "1. 性别 (1:男 2:女)*", "required": True},
    'srh': {"type": "categorical", "options": [1,2,3,4,5], "desc": "2. 自评健康 (1-5: 很差→很好)*", "required": True},
    'adlab_c': {"type": "categorical", "options": [0,1,2,3,4,5,6], "desc": "3. 日常活动能力 (0-6)*", "required": True},
    'arthre': {"type": "categorical", "options": [0, 1], "desc": "4. 关节炎 (0:无 1:有)*", "required": True},
    'digeste': {"type": "categorical", "options": [0, 1], "desc": "5. 消化问题 (0:无 1:有)*", "required": True},
    'retire': {"type": "categorical", "options": [0, 1], "desc": "6. 退休状态 (0:未退 1:已退)*", "required": True},
    'satlife': {"type": "categorical", "options": [1,2,3,4,5], "desc": "7. 生活满意度 (1-5)*", "required": True},
    'sleep': {"type": "numerical", "min": 0.0, "max": 24.0, "default": 7.0, "step": 0.5, 
              "desc": "8. 睡眠时长(小时)*", "required": True},
    'disability': {"type": "categorical", "options": [0, 1], "desc": "9. 残疾 (0:无 1:有)*", "required": True},
    'internet': {"type": "categorical", "options": [0, 1], "desc": "10. 上网 (0:否 1:是)*", "required": True},
    'hope': {"type": "categorical", "options": [1,2,3,4], "desc": "11. 希望程度 (1-4)*", "required": True},
    'fall_down': {"type": "categorical", "options": [0, 1], "desc": "12. 跌倒史 (0:无 1:有)*", "required": True},
    'eyesight_close': {"type": "categorical", "options": [1,2,3,4,5], "desc": "13. 视力 (1-5)*", "required": True},
    'hear': {"type": "categorical", "options": [1,2,3,4,5], "desc": "14. 听力 (1-5)*", "required": True},
    'edu': {"type": "categorical", "options": [1,2,3,4], "desc": "15. 教育程度 (1-4)*", "required": True},
    'pension': {"type": "categorical", "options": [0, 1], "desc": "16. 养老金 (0:无 1:有)*", "required": True},
    'pain': {"type": "categorical", "options": [0, 1], "desc": "17. 慢性疼痛 (0:无 1:有)*", "required": True}
}

# --- 模型加载 ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('svm.pkl')
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()

model = load_model()

# --- 界面构建 ---
st.title("抑郁症风险预测模型 (完整变量版)")
st.markdown("**带*号为必填项**", unsafe_allow_html=True)

# 输入表单（强制完整保留所有变量）
feature_values = {}
missing_fields = []

with st.form("input_form"):
    cols = st.columns(3)
    col_idx = 0
    
    for i, (feature, prop) in enumerate(feature_ranges.items()):
        with cols[col_idx]:
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
                    index=0,
                    key=feature
                )
            feature_values[feature] = value
            
            # 检查必填项
            if prop.get("required", False) and value in (None, ""):
                missing_fields.append(prop["desc"].split(".")[1].split("(")[0].strip())
        
        col_idx = (col_idx + 1) % 3
    
    submitted = st.form_submit_button("提交评估", type="primary")

# --- 预测执行 ---
if submitted:
    if missing_fields:
        st.error(f"缺失必填字段: {', '.join(missing_fields)}")
    else:
        # 转换为模型输入格式
        features = np.array([[feature_values[f] for f in feature_ranges.keys()]])
        
        try:
            # 执行预测
            pred = model.predict(features)[0]
            try:
                proba = model.predict_proba(features)[0][pred]
                confidence = f"{proba*100:.1f}%"
            except:
                decision = model.decision_function(features)[0]
                confidence = f"决策值: {decision:.2f}"
            
            # 显示结果
            st.markdown(f"""
            <div style='border:2px solid #f63366; border-radius:5px; padding:15px; margin:20px 0;'>
                <h2 style='color:{"#ff0000" if pred == 1 else "#00aa00"}'>{"高风险" if pred == 1 else "低风险"}</h2>
                <p>置信度: <strong>{confidence}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # SHAP解释（改进版）
            st.subheader("特征影响分析")
            
            try:
                # 创建解释器
                explainer = shap.KernelExplainer(
                    model.predict, 
                    np.zeros((1, len(feature_ranges)))  # 简化背景数据
                )
                shap_values = explainer.shap_values(features)
                
                # 可视化1：条形图
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values, 
                    features, 
                    feature_names=list(feature_ranges.keys()),
                    plot_type="bar",
                    show=False
                )
                st.pyplot(fig1)
                
                # 可视化2：力导向图
                st.markdown("**各特征贡献力**")
                fig2, ax2 = plt.subplots(figsize=(12, 3))
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[0],
                    features[0],
                    feature_names=list(feature_ranges.keys()),
                    matplotlib=True,
                    show=False
                )
                st.pyplot(fig2)
                
                # 添加颜色说明
                st.markdown("""
                <div style='background-color:#f0f0f0; padding:10px; border-radius:5px; margin-top:10px;'>
                    <span style='color:#ff0000'>红色</span>表示增加风险的因子 | 
                    <span style='color:#00aa00'>绿色</span>表示降低风险的因子
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.warning(f"SHAP解释生成失败: {str(e)}")
                
        except Exception as e:
            st.error(f"预测过程中出错: {str(e)}")

# --- 变量说明 ---
with st.expander("点击查看变量详细说明"):
    st.table(pd.DataFrame.from_dict(feature_ranges, orient='index')[['desc']])
