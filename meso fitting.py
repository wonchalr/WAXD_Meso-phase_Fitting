import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

st.title("WAXD Fitting Program")

# =============================================================================
# 1. Upload Data
# =============================================================================
st.header("1. Upload Data")
st.write("Upload 100% Mesophase Excel file and Sample Data (Excel file or paste text)")

# 100% Mesophase data (จะไม่ถูกลบโดย Clear Data)
meso_file = st.file_uploader("Upload 100% Mesophase Excel File", type=["xlsx", "xls"], key="meso")

# Sample data: file upload และ/หรือ paste text
sample_file = st.file_uploader("Upload Sample Excel File", type=["xlsx", "xls"], key="sample")
sample_text = st.text_area("Or paste sample data (columns: 2θ, I)", value="", height=150)

# ปุ่ม Clear Data สำหรับ sample data
if st.button("Clear Data", key="clear_data"):
    st.session_state["sample_cleared"] = True
    st.session_state["sample_text"] = ""  # เคลียร์ข้อความในกล่อง paste
    st.session_state["sample_x"] = np.array([])
    st.session_state["sample_y"] = np.array([])
    st.success("Sample data cleared.")

# ฟังก์ชันสำหรับโหลด sample data จากไฟล์หรือข้อความที่ paste
def load_sample_data():
    if sample_file is not None:
        try:
            df = pd.read_excel(sample_file)
            return df.iloc[:, 0].values, df.iloc[:, 1].values
        except Exception as e:
            st.error(f"Error reading sample file: {e}")
            return np.array([]), np.array([])
    elif sample_text.strip() != "":
        try:
            data = []
            for line in sample_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                # แบ่งข้อมูลโดยใช้ comma หรือ whitespace
                if "," in line:
                    parts = line.split(",")
                else:
                    parts = line.split()
                if len(parts) >= 2:
                    data.append([float(parts[0]), float(parts[1])])
            data = np.array(data)
            if data.size > 0:
                return data[:, 0], data[:, 1]
            else:
                return np.array([]), np.array([])
        except Exception as e:
            st.error(f"Error parsing sample text: {e}")
            return np.array([]), np.array([])
    else:
        return np.array([]), np.array([])

# โหลด sample data เบื้องต้น (ถ้ามี) และเก็บไว้ใน session_state
if "sample_x" not in st.session_state or "sample_y" not in st.session_state:
    sx, sy = load_sample_data()
    st.session_state["sample_x"] = sx
    st.session_state["sample_y"] = sy

# ปุ่ม Update Data Plot สำหรับ sample data
if st.button("Update Data Plot", key="update_plot_btn"):
    sx, sy = load_sample_data()
    st.session_state["sample_x"] = sx
    st.session_state["sample_y"] = sy
    st.session_state["sample_cleared"] = False
    st.success("Sample data updated.")
    # Plot ข้อมูลดิบใหม่
    if sx.size > 0:
        fig_update = go.Figure()
        fig_update.add_trace(go.Scatter(x=sx, y=sy, mode='markers', name='Sample (Raw)'))
        # สำหรับ 100% Mesophase ให้ใช้ข้อมูลจาก meso_file
        if meso_file is not None:
            try:
                meso_df = pd.read_excel(meso_file)
                meso_x = meso_df.iloc[:, 0].values
                meso_y = meso_df.iloc[:, 1].values
                fig_update.add_trace(go.Scatter(x=meso_x, y=meso_y, mode='lines', name='100% Mesophase (Raw)'))
            except Exception as e:
                st.error(f"Error reading 100% Mesophase file: {e}")
        st.plotly_chart(fig_update, use_container_width=True, key="update_plot")
    else:
        st.info("No sample data available. Please upload or paste sample data.")

# โหลด 100% Mesophase data (ต้องมีไฟล์)
if meso_file is not None:
    try:
        meso_df = pd.read_excel(meso_file)
        meso_x = meso_df.iloc[:, 0].values
        meso_y = meso_df.iloc[:, 1].values
    except Exception as e:
        st.error(f"Error reading 100% Mesophase file: {e}")
        st.stop()
else:
    st.info("Please upload the 100% Mesophase Excel file.")
    st.stop()

# หาก sample data ยังไม่มีการอัพเดท ให้ใช้ค่าจาก session_state
sample_x = st.session_state.get("sample_x", np.array([]))
sample_y = st.session_state.get("sample_y", np.array([]))

# Plot ข้อมูลดิบ (เฉพาะกรณีที่มี sample data)
st.header("Raw Data Plot")
fig_raw = go.Figure()
if sample_x.size > 0:
    fig_raw.add_trace(go.Scatter(x=sample_x, y=sample_y, mode='markers', name='Sample (Raw)'))
else:
    fig_raw.add_annotation(text="No sample data", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
fig_raw.add_trace(go.Scatter(x=meso_x, y=meso_y, mode='lines', name='100% Mesophase (Raw)'))
st.plotly_chart(fig_raw, use_container_width=True, key="raw_plot")

# =============================================================================
# 2. Baseline Correction
# =============================================================================
st.header("2. Baseline Correction")

# กำหนดช่วง slider ตามค่า 2θ ของ meso data
x_min = float(np.min(meso_x))
x_max = float(np.max(meso_x))
bg1 = st.slider("Select baseline point 1 (2θ)", min_value=x_min, max_value=x_max,
                value=x_min + (x_max - x_min) / 4, step=0.1, key="bg1")
bg2 = st.slider("Select baseline point 2 (2θ)", min_value=x_min, max_value=x_max,
                value=x_min + 3*(x_max - x_min) / 4, step=0.1, key="bg2")

# แสดงเส้นแนวตั้งในกราฟ raw data
fig_raw.add_vline(x=bg1, line=dict(color='red', dash='dash'))
fig_raw.add_vline(x=bg2, line=dict(color='red', dash='dash'))
st.plotly_chart(fig_raw, use_container_width=True, key="raw_plot_with_lines")

if st.button("Apply Baseline Correction", key="apply_baseline"):
    # คำนวณค่า baseline จาก meso data โดยใช้ np.interp เพื่อหาค่า y ที่ bg1 และ bg2
    y1 = np.interp(bg1, meso_x, meso_y)
    y2 = np.interp(bg2, meso_x, meso_y)
    slope = (y2 - y1) / (bg2 - bg1)
    intercept = y1 - slope * bg1
    baseline = (slope, intercept)
    # ลบ baseline (สมการเส้นตรง: slope*x + intercept) ออกจากข้อมูล
    meso_y_corr = meso_y - (slope * meso_x + intercept)
    sample_y_corr = sample_y - (slope * sample_x + intercept)
    # เก็บข้อมูลที่แก้ไขแล้วไว้ใน session_state
    st.session_state.meso_y_corr = meso_y_corr
    st.session_state.sample_y_corr = sample_y_corr
    st.session_state.baseline = baseline
    st.success("Baseline correction applied.")
    
    # Plot กราฟข้อมูลหลัง baseline correction
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(x=sample_x, y=sample_y_corr, mode='markers', name='Sample (Corrected)'))
    fig_corr.add_trace(go.Scatter(x=meso_x, y=meso_y_corr, mode='lines', name='100% Mesophase (Corrected)'))
    # แสดงเส้น baseline (สำหรับอ้างอิง)
    baseline_line = slope * meso_x + intercept
    fig_corr.add_trace(go.Scatter(x=meso_x, y=baseline_line, mode='lines', name='Baseline',
                                  line=dict(dash='dot', color='gray')))
    fig_corr.add_vline(x=bg1, line=dict(color='red', dash='dash'))
    fig_corr.add_vline(x=bg2, line=dict(color='red', dash='dash'))
    st.plotly_chart(fig_corr, use_container_width=True, key="corr_plot")

# =============================================================================
# 3. Overlay Alpha Phase
# =============================================================================
st.header("3. Overlay Alpha Phase")
if st.button("Overlay Alpha Phase", key="overlay_alpha"):
    if "meso_y_corr" not in st.session_state or "sample_y_corr" not in st.session_state:
        st.warning("Please apply baseline correction first.")
    else:
        meso_y_corr = st.session_state.meso_y_corr
        sample_y_corr = st.session_state.sample_y_corr
        # กำหนด default amplitude ของ peak (ประมาณ 10% ของ (max-min))
        default_amp = 0.1 * (np.max(sample_y_corr) - np.min(sample_y_corr))
        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Scatter(x=sample_x, y=sample_y_corr, mode='markers',
                                         name='Sample (Corrected)'))
        fig_overlay.add_trace(go.Scatter(x=meso_x, y=meso_y_corr, mode='lines',
                                         name='100% Mesophase (Corrected)'))
        fig_overlay.add_vline(x=bg1, line=dict(color='red', dash='dash'))
        fig_overlay.add_vline(x=bg2, line=dict(color='red', dash='dash'))
        # กำหนดค่า default ของ 5 crystalline peaks (ใช้ Gaussian)
        defaultPeaks = [
            {"center": 12.3, "fwhm": 0.63},
            {"center": 14.8, "fwhm": 0.56},
            {"center": 16.3, "fwhm": 0.84},
            {"center": 18.6, "fwhm": 0.74},
            {"center": 19.2, "fwhm": 0.66},
        ]
        x_fit = np.linspace(x_min, x_max, 1000)
        for i, peak in enumerate(defaultPeaks):
            center = peak["center"]
            fwhm = peak["fwhm"]
            sigma = fwhm / 2.3548
            # Gaussian peak: A*exp(-((x-center)^2/(2*sigma^2)))
            y_peak = default_amp * np.exp(-((x_fit - center)**2) / (2 * sigma**2))
            fig_overlay.add_trace(go.Scatter(x=x_fit, y=y_peak, mode='lines',
                                             name=f"Peak {i+1} (default)",
                                             line=dict(dash='dash')))
        st.plotly_chart(fig_overlay, use_container_width=True, key="overlay_plot")

# =============================================================================
# 4. Fitting (Using Gaussian Peaks)
# =============================================================================
st.header("4. Fitting")
if st.button("Fitting", key="fitting_btn"):
    if "meso_y_corr" not in st.session_state or "sample_y_corr" not in st.session_state:
        st.warning("Please apply baseline correction first.")
    else:
        meso_y_corr = st.session_state.meso_y_corr
        sample_y_corr = st.session_state.sample_y_corr

        # นิยาม model function สำหรับ curve_fit โดยใช้ Gaussian peaks
        def model_func(x, factor,
                       A1, c1, s1,
                       A2, c2, s2,
                       A3, c3, s3,
                       A4, c4, s4,
                       A5, c5, s5):
            # Interpolate ค่า meso จากข้อมูลที่ถูก baseline corrected
            meso_interp = np.interp(x, meso_x, meso_y_corr)
            y = factor * meso_interp
            y += A1 * np.exp(-((x - c1)**2) / (2 * s1**2))
            y += A2 * np.exp(-((x - c2)**2) / (2 * s2**2))
            y += A3 * np.exp(-((x - c3)**2) / (2 * s3**2))
            y += A4 * np.exp(-((x - c4)**2) / (2 * s4**2))
            y += A5 * np.exp(-((x - c5)**2) / (2 * s5**2))
            return y

        # กำหนด initial guess parameters
        factor0 = 1.0
        amp0 = 0.1 * (np.max(sample_y_corr) - np.min(sample_y_corr))
        defaultPeaks = [
            {"center": 12.3, "fwhm": 0.63},
            {"center": 14.8, "fwhm": 0.56},
            {"center": 16.3, "fwhm": 0.84},
            {"center": 18.6, "fwhm": 0.74},
            {"center": 19.2, "fwhm": 0.66},
        ]
        p0 = [factor0]
        for peak in defaultPeaks:
            c0 = peak["center"]
            sigma0 = peak["fwhm"] / 2.3548
            p0.extend([amp0, c0, sigma0])

        try:
            popt, pcov = curve_fit(model_func, sample_x, sample_y_corr, p0=p0)
        except Exception as e:
            st.error(f"Fitting failed: {e}")
        else:
            fitted_y = model_func(sample_x, *popt)
            meso_fit = popt[0] * np.interp(sample_x, meso_x, meso_y_corr)
            peaks = []
            for i in range(5):
                A = popt[1 + 3 * i]
                c = popt[1 + 3 * i + 1]
                s = popt[1 + 3 * i + 2]
                peak_y = A * np.exp(-((sample_x - c)**2) / (2 * s**2))
                peaks.append(peak_y)
            # คำนวณพื้นที่โดยใช้ integration แบบ trapezoidal
            area_meso = np.trapz(meso_fit, sample_x)
            area_alpha = sum(np.trapz(peak, sample_x) for peak in peaks)
            percent_alpha = (area_alpha / (area_meso + area_alpha)) * 100

            # Plot ผลการ fitting
            fig_fit = go.Figure()
            fig_fit.add_trace(go.Scatter(x=sample_x, y=sample_y_corr, mode='markers', name='Data'))
            fig_fit.add_trace(go.Scatter(x=sample_x, y=fitted_y, mode='lines', name='Fitted'))
            fig_fit.add_trace(go.Scatter(x=sample_x, y=meso_fit, mode='lines', name='Scaled 100% Mesophase'))
            for i, peak in enumerate(peaks):
                fig_fit.add_trace(go.Scatter(x=sample_x, y=peak, mode='lines', name=f"Peak {i+1}"))
            st.plotly_chart(fig_fit, use_container_width=True, key="fitting_plot")

            # แสดงผลลัพธ์ของการ fitting
            st.subheader("Fitting Results")
            st.write(f"Optimized factor for 100% Mesophase: {popt[0]:.4f}")
            for i in range(5):
                A = popt[1 + 3 * i]
                c = popt[1 + 3 * i + 1]
                s = popt[1 + 3 * i + 2]
                fwhm_approx = 2.3548 * s
                st.write(f"Peak {i+1}: Amplitude = {A:.4f}, Center = {c:.4f}, Sigma = {s:.4f}, Approx. FWHM = {fwhm_approx:.4f}")
            st.write(f"Alpha-phase content (%): {percent_alpha:.2f}%")
