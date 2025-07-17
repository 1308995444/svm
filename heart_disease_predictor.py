import streamlit as st
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

# 设置页面布局
st.set_page_config(layout="wide")

# 自定义CSS样式
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .result-box {
        border: 2px solid #f63366;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .shap-table {
        width: 100%;
        border-collapse: collapse;
    }
    .shap-table th, .shap-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .shap-table th {
        background-color: #f63366;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 模拟模型预测函数
def predict(features):
    # 这里应该是您的实际模型预测代码
    # 返回 (预测类别, 预测概率, SHAP值)
    return 1, 0.8959, np.random.randn(1, len(feature_ranges)) * 0.1

# 特征定义
feature_ranges = {
    'age': {"type": "numerical", "min": 18, "max": 100, "default": 45, "desc": "年龄/Age"},
    'gender': {"type": "categorical", "options": ["Male", "Female"], "desc": "性别/Gender"},
    'blood_pressure': {"type": "numerical", "min": 80, "max": 200, "default": 120, "desc": "血压/Blood Pressure"},
    # 添加更多特征...
}

# 用户输入
st.title("健康风险评估系统")
st.header("请输入您的健康数据")

input_data = {}
col1, col2 = st.columns(2)
for i, (feature, props) in enumerate(feature_ranges.items()):
    col = col1 if i % 2 == 0 else col2
    if props["type"] == "numerical":
        input_data[feature] = col.number_input(
            props["desc"],
            min_value=props["min"],
            max_value=props["max"],
            value=props["default"]
        )
    else:
        input_data[feature] = col.selectbox(
            props["desc"],
            options=props["options"]
        )

# 预测按钮
if st.button("开始风险评估", use_container_width=True):
    # 转换输入数据为模型需要的格式
    features = np.array([[input_data[f] for f in feature_ranges.keys()]])
    
    # 获取预测结果
    pred_class, pred_prob, shap_values = predict(features)
    
    # 显示预测结果
    with st.container():
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="big-font">Prediction Result</p>', unsafe_allow_html=True)
            st.markdown(f'<h2 style="color:{"red" if pred_class == 1 else "green"}">{"High risk" if pred_class == 1 else "Low risk"}</h2>', 
                        unsafe_allow_html=True)
        
        with col2:
            st.markdown('<p class="big-font">Confidence</p>', unsafe_allow_html=True)
            st.markdown(f'<h2>{pred_prob*100:.2f}%</h2>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # SHAP解释可视化
    st.subheader("Feature Impact Analysis")
    
    # 创建SHAP表格
    feature_names = list(feature_ranges.keys())
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact Value": shap_values[0],
        "Percentage": np.abs(shap_values[0])/np.sum(np.abs(shap_values[0]))*100
    }).sort_values("Impact Value", ascending=False)
    
    # 显示SHAP表格
    st.table(shap_df.style.format({
        "Impact Value": "{:.3f}",
        "Percentage": "{:.1f}%"
    }).apply(lambda x: ['background-color: #ffcccc' if v < 0 else 'background-color: #ccffcc' for v in x] 
             if x.name == "Impact Value" else ['']*len(x)))
    
    # SHAP force plot
    plt.figure(figsize=(12, 3))
    shap.force_plot(
        base_value=0.5,  # 基线值
        shap_values=shap_values[0],
        features=features[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())
    plt.close()
    
    # 添加解释说明
    st.info("""
    **图表说明:**
    - 正值(绿色)表示该特征增加了患病风险
    - 负值(红色)表示该特征降低了患病风险
    - 数值大小表示影响程度
    """)
