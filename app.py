import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Hệ thống Phân tích & Dự báo BĐS",
    page_icon="🏠",
    layout="wide"
)

@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 300
    districts = ["Quận 1", "Quận 3", "Quận 7", "Quận Bình Thạnh", "Quận Thủ Đức"]
    data = {
        "title": [f"Bài đăng #{i+1}" for i in range(n_samples)],
        "district": np.random.choice(districts, n_samples),
        "area_m2": np.random.uniform(30, 150, n_samples).round(1),
        "num_bedrooms": np.random.randint(1, 5, n_samples),
    }
    df = pd.DataFrame(data)
    
    district_coeff = {"Quận 1": 120, "Quận 3": 100, "Quận 7": 70, "Quận Bình Thạnh": 85, "Quận Thủ Đức": 50}
    df["price_billion"] = df.apply(
        lambda row: round((row["area_m2"] * district_coeff[row["district"]] + row["num_bedrooms"] * 200 + np.random.normal(0, 500)) / 1000, 2),
        axis=1
    )
    df["price_billion"] = df["price_billion"].apply(lambda x: max(x, 1.0))
    return df

df = load_data()

@st.cache_resource
def train_model(data):
    df_encoded = pd.get_dummies(data[["district", "area_m2", "num_bedrooms"]], drop_first=True)
    X = df_encoded
    y = data["price_billion"]
    model = LinearRegression()
    model.fit(X, y)
    return model, list(X.columns)

model, feature_columns = train_model(df)

# 3. GIAO DIỆN WEB
st.title("🏠 Hệ thống Thu thập, Phân tích & Dự báo Bất Động Sản")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard Thống kê", "🔍 Tra cứu & Bộ lọc", "🤖 Dự báo Giá (AI/ML)"])

# TAB 1: DASHBOARD
with tab1:
    st.header("Thống kê Thị trường Bất động sản")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số bài đăng", f"{len(df)} tin")
    col2.metric("Giá trung bình", f"{df['price_billion'].mean():.2f} Tỷ VNĐ")
    col3.metric("Diện tích trung bình", f"{df['area_m2'].mean():.1f} m²")
    col4.metric("Giá/m² trung bình", f"{(df['price_billion']*1000/df['area_m2']).mean():.1f} Tr/m²")
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Phân bố Giá theo Khu vực")
        fig_box = px.box(df, x="district", y="price_billion", color="district",
                         labels={"district": "Khu vực", "price_billion": "Giá (Tỷ VNĐ)"})
        st.plotly_chart(fig_box, use_container_width=True)
    with c2:
        st.subheader("Tương quan Diện tích - Giá")
        fig_scatter = px.scatter(df, x="area_m2", y="price_billion", color="district", size="num_bedrooms",
                                 labels={"area_m2": "Diện tích (m²)", "price_billion": "Giá (Tỷ VNĐ)"})
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.header("Tra cứu Bài đăng")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_district = st.multiselect("Chọn Khu vực:", options=df["district"].unique(), default=df["district"].unique())
    with col_f2:
        min_p, max_p = float(df["price_billion"].min()), float(df["price_billion"].max())
        selected_price = st.slider("Khoảng giá (Tỷ VNĐ):", min_value=min_p, max_value=max_p, value=(min_p, max_p))
    
    filtered_df = df[(df["district"].isin(selected_district)) & 
                      (df["price_billion"] >= selected_price[0]) & 
                      (df["price_billion"] <= selected_price[1])]
    st.dataframe(filtered_df, use_container_width=True)

with tab3:
    st.header("Dự báo Giá nhà với Machine Learning")
    with st.form("predict_form"):
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            input_district = st.selectbox("Chọn Quận:", options=df["district"].unique())
        with col_in2:
            input_area = st.number_input("Diện tích (m²):", min_value=10.0, max_value=500.0, value=60.0)
        with col_in3:
            input_rooms = st.number_input("Số phòng ngủ:", min_value=1, max_value=10, value=2)
            
        submit = st.form_submit_button("🚀 Dự báo Giá")
        
    if submit:
        input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
        input_data["area_m2"] = input_area
        input_data["num_bedrooms"] = input_rooms
        district_col = f"district_{input_district}"
        if district_col in input_data.columns:
            input_data[district_col] = 1
            
        predicted = max(model.predict(input_data)[0], 0.5)
        st.success(f"📌 **Kết quả dự báo:** Ước tính **{predicted:.2f} Tỷ VNĐ**")
        st.info(f"💡 Đơn giá: **{(predicted * 1000 / input_area):.1f} Triệu VNĐ / m²**")