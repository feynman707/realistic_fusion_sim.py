import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
e = 1.6e-19  # elementary charge [C]
k_B = 1.38e-23  # Boltzmann constant [J/K]
keV_to_J = 1.602e-16

# --- Reaction parameters ---
E_alpha = 3.5 * keV_to_J  # alpha energy [J]
E_total = 17.6 * keV_to_J

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

# Temperature evolution model
def simulate(T0, n, dt, steps):
    Ts = [T0]
    for i in range(steps):
        T = Ts[-1]
        P_fusion = fusion_power(n, T)
        P_alpha = alpha_heating(n, T)
        P_loss = bremsstrahlung_loss(n, T)

        # Energy balance equation: dT/dt ~ (P_gain - P_loss) / (3/2 n k_B)
        dT = (P_alpha - P_loss) * dt / (1.5 * n * keV_to_J)
        T_new = max(T + dT, 0.01)
        Ts.append(T_new)
    return np.arange(steps + 1) * dt, Ts

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🔥 실험 기반 레이저 핵융합 시뮬레이션")

col1, col2 = st.columns(2)

with col1:
    T0 = st.slider("초기 온도 T₀ (keV)", 1.0, 30.0, 5.0, step=0.5)
    n = st.slider("입자 밀도 n (cm⁻³)", 1e20, 1e22, 1e21, step=1e20)

with col2:
    dt = st.number_input("시간 간격 dt (ns)", value=0.5, step=0.1) * 1e-9
    steps = st.number_input("총 시간 스텝 수", value=200, step=10, format="%d")

# Run simulation
if st.button("▶️ 시뮬레이션 시작"):
    t, T_curve = simulate(T0, n, dt, int(steps))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t * 1e9, T_curve)
    ax.set_xlabel("시간 (ns)")
    ax.set_ylabel("온도 (keV)")
    ax.set_title("핵융합 플라즈마 온도 변화")
    st.pyplot(fig)

    final_T = T_curve[-1]
    st.success(f"✅ 최종 온도: {final_T:.2f} keV")

    # Lawson 조건 판정
    tau = 1e-9 * steps  # 단순한 시간 스케일
    if n * tau > 1e14:
        st.markdown("### 🌟 핵융합 조건 만족! (n·τ > 10¹⁴)")
    else:
        st.markdown("### ❌ 핵융합 조건 불충분 (n·τ < 10¹⁴)")
