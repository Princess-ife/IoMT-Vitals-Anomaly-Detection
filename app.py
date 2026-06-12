import streamlit as st
import joblib as jb
import numpy as np

# --- Page setup must come before any Streamlit UI output ---
st.set_page_config(
    page_title="IoMT Vitals Anomaly Detection",
    page_icon="🫀",
    layout="wide"
)

# --- Load trained model and scaler ---
model = jb.load("isolation_forest_model.pkl")
scaler = jb.load("scaler.pkl")

# --- Custom CSS for dashboard-style UI ---
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

.hero-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    padding: 32px;
    border-radius: 24px;
    color: white;
    margin-bottom: 28px;
}

.hero-card h1 {
    font-size: 38px;
    margin-bottom: 8px;
}

.hero-card p {
    color: #cbd5e1;
    font-size: 16px;
}

.section-card {
    background: #ffffff;
    padding: 26px;
    border-radius: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    margin-bottom: 24px;
}

.metric-card {
    background: #ffffff;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    text-align: center;
    min-height: 180px;
}

.metric-card h4 {
    margin-top: 12px;
    margin-bottom: 6px;
    color: #111827;
}

.metric-card p {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
}
</style>
""", unsafe_allow_html=True)

# --- Header section ---
st.markdown("""
<div class="hero-card">
    <h1>IoMT Vitals Anomaly Detection</h1>
    <p>
        A clinical vitals dashboard powered by an Isolation Forest model for anomaly detection.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Icons as SVG strings ---
body_icon = """<svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">

<path fill-rule="evenodd" clip-rule="evenodd" d="M24 12C25.6569 12 27 10.6569 27 9C27 7.34315 25.6569 6 24 6C22.3431 6 21 7.34315 21 9C21 10.6569 22.3431 12 24 12ZM24 14C26.7614 14 29 11.7614 29 9C29 6.23858 26.7614 4 24 4C21.2386 4 19 6.23858 19 9C19 11.7614 21.2386 14 24 14Z" fill="currentColor"/>

<path fill-rule="evenodd" clip-rule="evenodd" d="M17.374 18.3144C18.3174 18.494 19 19.3188 19 20.2792L19 41C19 41.5523 19.4477 42 20 42H20.0868C20.604 42 21.0359 41.6056 21.0827 41.0905L22 31C22.0029 29.8975 22.8975 29.0052 24 29.0052C25.1025 29.0052 25.9971 29.8974 26 30.9999L26.9173 41.0905C26.9641 41.6056 27.396 42 27.9132 42H28C28.5523 42 29 41.5523 29 41V20.3169C29 19.3564 29.6828 18.5316 30.6264 18.3521C32.3824 18.0182 34.2392 17.5548 36.2798 16.9601C36.8101 16.8056 37.1146 16.2505 36.9601 15.7202C36.8056 15.19 36.2505 14.8855 35.7202 15.04C31.0861 16.3906 27.5307 17.0156 24.0043 16.9998C20.4743 16.9839 16.9146 16.3259 12.2674 15.0365C11.7352 14.8888 11.1841 15.2005 11.0364 15.7327C10.8888 16.2649 11.2005 16.816 11.7327 16.9636C13.7658 17.5278 15.6186 17.9803 17.374 18.3144ZM22.9779 41.8025C22.6245 43.0798 21.4552 44 20.0868 44H20C18.3431 44 17 42.6569 17 41L17 20.2792C15.1742 19.9317 13.2653 19.4645 11.1979 18.8908C9.60138 18.4478 8.66625 16.7945 9.10925 15.1979C9.55225 13.6014 11.2056 12.6663 12.8021 13.1093C17.3715 14.3772 20.7371 14.9851 24.0132 14.9998C27.2788 15.0144 30.6314 14.4399 35.1606 13.1199C36.7513 12.6563 38.4166 13.57 38.8802 15.1606C39.3438 16.7513 38.4301 18.4166 36.8394 18.8802C34.7552 19.4876 32.8345 19.9681 31 20.3169V41C31 42.6569 29.6569 44 28 44H27.9132C26.5448 44 25.3755 43.0798 25.0221 41.8025C24.9897 41.6851 24.9641 41.5647 24.9458 41.4417C24.9375 41.3856 24.9307 41.3288 24.9255 41.2716L24.0082 31.1811C24.0029 31.1223 24.0002 31.0637 24 31.0052C23.9998 31.0637 23.9971 31.1223 23.9918 31.1811L23.0745 41.2716C23.0693 41.3288 23.0625 41.3856 23.0542 41.4417C23.0359 41.5647 23.0103 41.6851 22.9779 41.8025Z" fill="currentColor"/>

</svg>"""



heart_icon = """<svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">

<path fill-rule="evenodd" clip-rule="evenodd" d="M30.0496 9.52364V7C30.0496 6.73478 29.9433 6.48043 29.754 6.29289C29.5648 6.10535 29.3081 6 29.0404 6L24.6966 6.00004C24.1393 6.00005 23.6874 6.44776 23.6874 7.00004V9.43344L22.2014 10.1236L20.2666 7.71934C19.9444 7.31893 19.369 7.22444 18.9336 7.50046L15.5299 9.65838C15.2891 9.81105 15.1249 10.0578 15.0782 10.3372C15.0315 10.6167 15.1065 10.9027 15.2848 11.1242L17.1795 13.4786C17.085 13.5921 16.9853 13.7173 16.885 13.8521C16.6902 14.1137 16.4635 14.4494 16.2719 14.8321C15.5246 14.1605 14.6395 13.6048 13.8044 13.1624C12.8859 12.676 11.9889 12.3063 11.3251 12.0589C10.992 11.9348 10.4234 11.7437 10.2274 11.6797C9.95398 11.6 9.62982 11.6346 9.38664 11.7819C9.14346 11.9292 8.97433 12.1723 8.92128 12.4498L8.01764 17.1765C7.96689 17.4419 8.0267 17.7164 8.18346 17.9375C8.34021 18.1586 8.58057 18.3074 8.84981 18.3501C9.95422 18.5251 10.7886 19.0696 11.36 19.592C11.6445 19.8522 11.8562 20.1003 11.9943 20.2797C12.0134 20.3046 12.0338 20.3323 12.054 20.3603C11.0452 21.885 10.6551 23.6793 10.7178 25.5266C10.8027 28.0283 11.7131 30.6743 13.0789 33.0685C14.4475 35.4676 16.3084 37.6759 18.37 39.2949C20.4174 40.9028 22.7558 42 25.0659 42C27.3703 42 29.7686 40.9085 31.8923 39.2926C34.0322 37.6644 35.9918 35.4312 37.4281 32.9732C40.2505 28.1434 41.2807 21.9328 36.7183 17.934C36.5292 17.7683 36.3537 17.6285 36.1859 17.5069C36.9168 17.4975 37.7563 17.5123 38.734 17.5457L39.7493 17.5805L40 10.8317L38.9965 10.7974C36.4319 10.7097 34.0766 10.8212 31.9386 11.308C31.4066 10.6245 30.8297 10.1119 30.3808 9.76632C30.2613 9.67431 30.1499 9.59342 30.0496 9.52364Z" fill="currentColor"/>

</svg>"""



nose_icon = """<svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">

<path d="M30.4035 5.08497C30.9088 5.30778 31.1378 5.89807 30.915 6.40341L30 5.99998C30.915 6.40341 30.9151 6.40318 30.915 6.40341L30.9145 6.40457L30.9095 6.41599L30.8946 6.44948C30.8816 6.47865 30.8624 6.52139 30.8373 6.57695C30.7871 6.68808 30.7131 6.85056 30.6168 7.05845C30.424 7.47417 30.1417 8.07182 29.7815 8.80394C29.0613 10.2673 28.0274 12.2722 26.7725 14.4375C24.2893 18.7222 20.8478 23.8016 17.1779 26.4468L17.1762 26.4481C13.3194 29.2159 12.5743 32.2502 13.1976 34.4373C13.8393 36.6893 15.9969 38.3725 18.6437 38.3725H19.1156C19.4217 37.306 20.2196 36.275 21.5774 35.6419C23.1162 34.9243 25.2972 34.7469 28.2261 35.4267C28.7641 35.5515 29.099 36.0889 28.9741 36.6268C28.8493 37.1648 28.3119 37.4997 27.7739 37.3749C25.1029 36.7549 23.4172 36.9907 22.4227 37.4545C21.8795 37.7077 21.5215 38.0367 21.2969 38.3725H21.8726C24.4282 38.3725 26.5 40.4443 26.5 43H24.5C24.5 41.5489 23.3237 40.3725 21.8726 40.3725H18.6437C15.139 40.3725 12.1703 38.1304 11.2741 34.9854C10.3595 31.7759 11.6647 27.9421 16.0091 24.8239C19.3008 22.451 22.5625 17.7129 25.0421 13.4346C26.2687 11.3182 27.2812 9.35506 27.987 7.92083C28.3397 7.20413 28.6153 6.62058 28.8022 6.21731C28.8957 6.0157 28.967 5.85922 29.0146 5.75378C29.0384 5.70106 29.0563 5.6611 29.0681 5.63465L29.0812 5.60519L29.0849 5.59672C29.3078 5.09138 29.8981 4.86216 30.4035 5.08497Z" fill="currentColor"/>

<path d="M32.5289 39.8487C33.9392 38.9698 35.2367 37.0814 35.7557 34.9909C36.2866 32.8525 36.0417 30.3276 34.0958 28.3064L32.655 29.6935C34.0101 31.101 34.2236 32.8617 33.8147 34.509C33.3938 36.2042 32.3531 37.6016 31.4711 38.1513L32.5289 39.8487Z" fill="currentColor"/>

</svg>"""

# --- Function to colour SVG icons ---
def colour_icon(svg_code, color):
    return f'<div style="color:{color}; width:64px; margin:auto;">{svg_code}</div>'


# --- Input layout ---
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Patient Vital Inputs")

    # --- Sliders ---
    heart_rate = st.slider("Heart Rate (bpm)", 60.0, 149.0, 105.0)
    spo2_level = st.slider("SpO₂ Level (%)", 80.0, 99.0, 89.5)
    systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", 100.0, 179.0, 140.0)
    diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", 60.0, 99.0, 80.0)
    temperature = st.slider("Body Temperature (°C)", 36.0, 38.0, 37.0)

    st.markdown("</div>", unsafe_allow_html=True)

# --- Derived features ---
pulse_pressure = systolic_bp - diastolic_bp
map_value = (systolic_bp + 2 * diastolic_bp) / 3

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧾 Live Patient Summary")

    # --- Display derived values ---
    st.metric("Pulse Pressure", f"{pulse_pressure:.1f} mmHg")
    st.metric("Mean Arterial Pressure", f"{map_value:.1f} mmHg")

    st.info("These derived features are passed into the anomaly detection model.")

    check_button = st.button("Check My Vitals", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# --- Button and prediction ---
if check_button:

    # --- Prepare features for model prediction ---
    features = np.array([[
        heart_rate,
        spo2_level,
        systolic_bp,
        diastolic_bp,
        temperature,
        pulse_pressure,
        map_value
    ]])

    # --- Scale features and predict anomaly ---
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🤖 Machine Learning Result")

    if prediction[0] == -1:
        st.error("Anomaly Detected. Please consult a healthcare professional.")
    else:
        st.success("No Anomaly Detected. Keep monitoring your health.")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Clinical threshold checks ---
    heart_flag = heart_rate < 60 or heart_rate > 100
    bp_flag = systolic_bp > 130 or diastolic_bp > 80
    spo2_flag = spo2_level < 95
    temp_flag = temperature < 36.1 or temperature > 37.2

    # --- Decide SVG colours based on clinical flags ---
    heart_color = "#ef4444" if heart_flag or bp_flag else "#94a3b8"
    nose_color = "#ef4444" if spo2_flag else "#94a3b8"
    body_color = "#ef4444" if temp_flag else "#94a3b8"

    st.subheader("🩺 Clinical Threshold Check")

    # --- Display SVG cards ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            {colour_icon(body_icon, body_color)}
            <h4>Body Temperature</h4>
            <p>{temperature:.1f} °C</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            {colour_icon(heart_icon, heart_color)}
            <h4>Heart / Blood Pressure</h4>
            <p>{heart_rate:.0f} bpm</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            {colour_icon(nose_icon, nose_color)}
            <h4>Oxygen / SpO₂</h4>
            <p>{spo2_level:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Explanation section ---
    with st.container(border=True):
        st.subheader("Explanation")
        st.caption(
            "This checks your vitals against standard clinical reference ranges, "
            "independent of the machine learning model."
        )

        if heart_flag:
            st.warning("Your heart rate is outside the normal range of 60 to 100 bpm.")

        if bp_flag:
            st.warning("Your blood pressure is above normal. (Systolic > 130 or Diastolic > 80).")

        if spo2_flag:
            st.warning("Your SpO₂ is below the normal range. 95% or higher is usually expected.")

        if temp_flag:
            st.warning("Your body temperature is outside the normal range of 36.1°C to 37.2°C.")

        if not (heart_flag or bp_flag or spo2_flag or temp_flag):
            st.success("All vitals are within normal clinical ranges.")