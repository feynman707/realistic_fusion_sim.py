import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import base64
from fpdf import FPDF

# --- Constants ---
e = 1.6e-19  # elementary charge [C]
k_B = 1.38e-23  # Boltzmann constant [J/K]
keV_to_J = 1.602e-16

# --- Reaction parameters ---
E_alpha = 3.5 * keV_to_J  # alpha energy [J]
E_total = 17.6 * keV_to_J
laser_efficiency = 0.15  # 가정: 레이저 에너지에서 알파 가열로 변환되는 비율

# Bosch-Hale like approximation for <sigma v>
def sigma_v_DT(T_keV):
    A = 1.1e-24  # cm^3/s, approximation
    B = 19.0     # keV^{1/3}
    return A * T_keV**(-2/3) * np.exp(-B / T_keV**(1/3))

# Bremsstrahlung loss [W/cm^3]
def bremsstrahlung_loss(n, T_keV):
    return 1.4e-34 * n**2 * np.sqrt(T_keV)  # n in cm^-3

# Alpha heating power [W/cm^3]
def alpha_heating(n, T_keV):
    sv = sigma_v_DT(T_keV)
    return 0.25 * n**2 * sv * E_alpha  # factor 0.25 for D-T

# Fusion power [W/cm^3]
def fusion_power(n, T_keV):
    sv = sigma_v_DT(T_keV)
    return 0.25 * n**2 * sv * E_total

# Temperature evolution model (multi-compression optional)
def simulate(T0, n0, dt, steps, compression_schedule, pulse_duration):
    Ts = [T0]
    ns = [n0]
    Pfus = []
    Ploss = []
    times = np.arange(steps + 1) * dt
    pulse_steps = int(pulse_duration / dt)

    for i in range(steps):
        T = Ts[-1]
        compression = compression_schedule[i] if i < len(compression_schedule) else 1
        current_n = n0 * compression if i < pulse_steps else n0

        P_fusion = fusion_power(current_n, T)
        P_alpha = alpha_heating(current_n, T)
        P_loss = bremsstrahlung_loss(current_n, T)

        Pfus.append(P_fusion)
        Ploss.append(P_loss)

        dT = (P_alpha - P_loss) * dt / (1.5 * current_n * keV_to_J)
        T_new = max(T + dT, 0.01)
        Ts.append(T_new)
        ns.append(current_n)
    return times, Ts, ns, Pfus, Ploss

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🔥 실험 기반 레이저 핵융합 시뮬레이션 (Q값 + PDF + 최적 조건 분석 포함)")

col1, col2 = st.columns(2)

with col1:
    T0 = st.slider("초기 온도 T₀ (keV)", 1.0, 30.0, 5.0, step=0.5)
    n = st.slider("입자 밀도 n (cm⁻³)", 1e20, 1e22, 1e21, step=1e20)
    compression_max = st.slider("최대 압축률 (배)", 1, 1000, 100, step=10)
    compression_stages = st.slider("압축 단계 수", 1, 10, 3)

with col2:
    dt = st.number_input("시간 간격 dt (ns)", value=0.5, step=0.1) * 1e-9
    steps = st.number_input("총 시간 스텝 수", value=200, step=10, format="%d")
    pulse_duration = st.number_input("레이저 펄스 시간 (ns)", value=50.0, step=5.0) * 1e-9

# 압축 스케줄 생성 (선형 다단계)
compression_schedule = np.ones(int(steps))
if compression_stages > 1:
    stage_len = int(steps / compression_stages)
    for s in range(compression_stages):
        start = s * stage_len
        end = start + stage_len
        compression_schedule[start:end] = 1 + (compression_max - 1) * ((s + 1) / compression_stages)
else:
    compression_schedule[:] = compression_max

if st.button("▶️ 시뮬레이션 시작"):
    t, T_curve, n_curve, P_fus_list, P_loss_list = simulate(T0, n, dt, int(steps), compression_schedule, pulse_duration)

    # 온도 그래프
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t * 1e9, T_curve)
    ax.set_xlabel("시간 (ns)")
    ax.set_ylabel("온도 (keV)")
    ax.set_title("핵융합 플라즈마 온도 변화")
    st.pyplot(fig)

    final_T = T_curve[-1]
    st.success(f"✅ 최종 온도: {final_T:.2f} keV")

    tau = dt * steps
    if n * tau > 1e14:
        st.markdown("### 🌟 핵융합 조건 만족! (n·τ > 10¹⁴)")
    else:
        st.markdown("### ❌ 핵융합 조건 불충분 (n·τ < 10¹⁴)")

    # Q값 계산
    E_output = np.sum(P_fus_list) * dt
    E_input = laser_efficiency * pulse_duration * 1e21  # 가정치
    Q = E_output / E_input if E_input > 0 else 0
    st.metric("Q값 (출력/입력 에너지 비율)", f"{Q:.2f}")

    # 최적 조건 분석 (펄스 지속시간이 Q값 최대가 되는 경우?)
    if st.checkbox("📈 펄스 지속시간 vs Q값 분석"):
        durations = np.linspace(10e-9, 300e-9, 20)
        qvals = []
        for d in durations:
            _, _, _, pf, _ = simulate(T0, n, dt, int(steps), compression_schedule, d)
            q = np.sum(pf)*dt / (laser_efficiency*d*1e21)
            qvals.append(q)
        figq, axq = plt.subplots()
        axq.plot(durations*1e9, qvals)
        axq.set_xlabel("레이저 펄스 시간 (ns)")
        axq.set_ylabel("Q값")
        axq.set_title("레이저 펄스 시간에 따른 Q값 변화")
        st.pyplot(figq)

    # PDF 결과 요약 리포트
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Fusion Simulation Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Final Temperature: {final_T:.2f} keV", ln=True)
    pdf.cell(200, 10, txt=f"Q Value: {Q:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"n: {n:.2e} cm^-3", ln=True)
    pdf.cell(200, 10, txt=f"Pulse Duration: {pulse_duration*1e9:.1f} ns", ln=True)
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    b64_pdf = base64.b64encode(pdf_output.getvalue()).decode()
    href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="fusion_report.pdf">📄 PDF 리포트 다운로드</a>'
    st.markdown(href_pdf, unsafe_allow_html=True)

    # 결과 CSV 다운로드
    csv = "시간(ns),온도(keV),밀도(cm^-3)\n" + "\n".join([
        f"{t[i]*1e9:.3f},{T_curve[i]:.3f},{n_curve[i]:.3e}" for i in range(len(t))
    ])
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="fusion_simulation_result.csv">📄 결과 CSV 다운로드</a>'
    st.markdown(href, unsafe_allow_html=True)
