"""
Occurrence Rate Estimator for Giant Planets and Brown Dwarfs

Updated: Bug-fixed & optimized version
"""

import numpy as np
import streamlit as st
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ln10 = np.log(10)

# GLOBAL CSS
st.markdown("""
<style>
.title-container {
    background: linear-gradient(to right, #1E88E5, #5E35B1);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 25px;
}
.main-title {
    color: white;
    font-size: 36px;
    font-weight: 800;
    text-align: center;
}
.subtitle {
    color: rgba(255, 255, 255, 0.9);
    font-size: 18px;
    text-align: center;
    font-style: italic;
}
.section-header {
    color: #1E88E5;
    font-size: 28px;
    font-weight: bold;
    margin: 30px 0 20px 0;
    border-bottom: 2px solid #1E88E5;
}
.subsection-header {
    color: #5E35B1;
    font-size: 22px;
    font-weight: bold;
    margin-top: 15px;
}
.tool-header {
    color: white;
    background: linear-gradient(to right, #1E88E5, #5E35B1);
    padding: 12px;
    font-size: 28px;
    border-radius: 5px;
    text-align: center;
}
</style>

<div class='title-container'>
    <div class='main-title'>Occurrence Rate Estimator</div>
    <div class='subtitle'>for Planets and Brown Dwarfs</div>
    <div class='subtitle'>Meyer & Li et al. (2025)</div>
</div>
""", unsafe_allow_html=True)


# ------------------------ INTRO -------------------------------
st.write("""Welcome to the tool ... (same intro text)""")

# ------------------------ MATH MODEL --------------------------
st.markdown("<div class='section-header'>Mathematical Model</div>", unsafe_allow_html=True)
st.latex(r"N_{TOTAL} = \int{\phi_{pl}(x) \psi_{pl}(q) dq dx} + \int{\phi_{bd}(x) \psi_{bd}(q) dq dx}")
st.latex(r"\psi_{pl}(q)=q^{-\alpha}")
st.latex(r"\psi_{bd}(q)=q^{-\beta}")
st.latex(r"\phi_{pl}(x)=\frac{A_{pl} e^{-(x-\mu_{pl})^2/2\sigma_{pl}^2}}{x\sqrt{2\pi}\sigma_{pl}\ln10}")
st.latex(r"\phi_{bd}(x)=\frac{A_{bd} e^{-(x-\mu_{bd})^2/2\sigma_{bd}^2}}{x\sqrt{2\pi}\sigma_{bd}\ln10}")

# ------------------------ TABLE -------------------------------
st.write("**Table 1: Companion Frequency & Log-normal Parameters**")
st.table({
    'Spectral Type': ['M', 'FGK', 'A'],
    'CF': ['0.236', '0.61', '0.219'],
    'μ (base-10)': ['1.43', '1.70', '2.72'],
    'σ (base-10)': ['1.21', '1.68', '0.79']
})

st.markdown("<div class='tool-header'>Frequency Calculation Tool</div>", unsafe_allow_html=True)

# ===============================================================
# SECTION 2 — HOST STAR PARAMETERS
# ===============================================================
st.markdown("<div class='section-header'>Host Star Parameters</div>", unsafe_allow_html=True)

ln_A_bd_default = -1.407
ln_A_pl_default = -4.720
alpha_bd_default = -0.292
alpha_gp_default = 1.296
mu_pl_default = 1.299
sigma_pl_default = np.exp(0.215)

# RADIO BUTTON + dynamic defaults
st_type = st.radio("Choose Stellar Type", ["M Dwarfs", "FGK", "A Stars"], index=1)

if st_type == "M Dwarfs":
    mu_bd_default = 1.43
    s_bd_default = 1.21
    key_suffix = "M"
elif st_type == "FGK":
    mu_bd_default = 1.70
    s_bd_default = 1.68
    key_suffix = "FGK"
else:
    mu_bd_default = 2.72
    s_bd_default = 0.79
    key_suffix = "A"

host_mass = st.number_input("Host Mass (Msun)", 0.1, 10.0, 1.0, step=0.01)

# ===============================================================
# SECTION 3 — MODEL PARAMETERS
# ===============================================================
st.markdown("<div class='section-header'>Model Parameters</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    alpha_bd = st.slider("β (BD slope)", -3.0, 3.0, alpha_bd_default, 0.01)
    A_bd = st.slider("A_bd", 0.0001, 1.0, np.exp(ln_A_bd_default)/ln10, 0.0001)
    mean_bd = st.slider("log10(mu_bd)", 0.0, 3.0, mu_bd_default, 0.01, key=f"mu_bd_{key_suffix}")
    sigma_bd = st.slider("log10(sigma_bd)", 0.0, 3.0, s_bd_default, 0.01, key=f"sigma_bd_{key_suffix}")

with col2:
    alpha_gp = st.slider("α (planet slope)", -3.0, 3.0, alpha_gp_default, 0.01)
    A_pl = st.slider("A_pl", 0.0001, 0.1, np.exp(ln_A_pl_default)/ln10, 0.0001)
    mu_pl = st.slider("log10(mu_pl)", 0.0, 3.0, mu_pl_default/ln10, 0.01)
    sigma_pl = st.slider("log10(sigma_pl)", 0.0, 3.0, sigma_pl_default/ln10, 0.01)


# ===============================================================
# EXACT SURFACE DENSITY FUNCTIONS
# ===============================================================
def surface_den_bd_exact(a):
    return (A_bd *
        np.exp(-(np.log10(a) - mean_bd)**2 / (2*sigma_bd**2)) /
        (a * np.sqrt(2*np.pi) * sigma_bd))

def surface_den_pl_exact(a):
    return (A_pl *
        np.exp(-(np.log10(a) - mu_pl)**2 / (2*sigma_pl**2)) /
        (a * np.sqrt(2*np.pi) * sigma_pl))

# ===============================================================
# SECTION 4 — MASS RANGE
# ===============================================================
st.markdown("<div class='section-header'>Companion Mass Range</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    Jup_min = st.number_input("Min Mass (MJup)", 0.03, 4000.0, 1.0)

with c2:
    Jup_max = st.number_input("Max Mass (MJup)", 0.03, 4000.0, 85.0)

if Jup_min >= Jup_max:
    st.error("Minimum mass must be less than maximum mass.")
    st.stop()

q_Jupiter = 0.001 / host_mass

# ===============================================================
# SECTION 5 — ORBITAL SEPARATION
# ===============================================================
st.markdown("<div class='section-header'>Orbital Separation</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    amin_calc = st.number_input("Min Separation (AU)", 0.1, 3000.0, 1.0)
with c2:
    amax_calc = st.number_input("Max Separation (AU)", 0.1, 3000.0, 100.0)

if amin_calc >= amax_calc:
    st.error("Minimum separation must be less than maximum.")
    st.stop()

# Precompute orbital integrals ONCE ✔
orb_int_bd = integrate.quad(surface_den_bd_exact, amin_calc, amax_calc)[0]
orb_int_pl = integrate.quad(surface_den_pl_exact, amin_calc, amax_calc)[0]

def mass_fctn_bd(q): return q**(-alpha_bd)
def mass_fctn_pl(q): return q**(-alpha_gp)

def dN_bd(q): return mass_fctn_bd(q) * orb_int_bd
def dN_pl(q): return mass_fctn_pl(q) * orb_int_pl

# ===============================================================
# SECTION 6 — COMPANION DISTRIBUTION PLOT
# ===============================================================
fig, ax = plt.subplots(figsize=(10, 8))

q_pl_min = Jup_min * q_Jupiter
q_pl_max = Jup_max * q_Jupiter

q_bd_min = 3 * q_Jupiter
q_bd_max = 0.67

mass_pl = np.logspace(np.log10(q_pl_min), np.log10(q_pl_max), 800)
mass_bd = np.logspace(np.log10(q_bd_min), np.log10(q_bd_max), 800)

pl_vals = dN_pl(mass_pl) * mass_pl * ln10
bd_vals = dN_bd(mass_bd) * mass_bd * ln10

mass_total = np.logspace(np.log10(q_pl_min), np.log10(q_bd_max), 800)
total_vals = (
    (dN_pl(mass_total)*(mass_total*ln10)) * ((mass_total>=q_pl_min)&(mass_total<=q_pl_max)) +
    (dN_bd(mass_total)*(mass_total*ln10)) * ((mass_total>=q_bd_min)&(mass_total<=q_bd_max))
)

ax.plot(np.log10(mass_pl), pl_vals, 'r', lw=3, label="Giant Planets")
ax.plot(np.log10(mass_bd), bd_vals, 'b', lw=3, label="Brown Dwarfs")
ax.plot(np.log10(mass_total), total_vals, 'orange', lw=2, label="Total")

ax.set_xlabel("log(q)", fontsize=20)
ax.set_ylabel("dN/dlog(q)", fontsize=20)
ax.legend(fontsize=14)
ax.set_title("Companion Frequency Distribution", fontsize=22)
st.pyplot(fig)

# ===============================================================
# SECTION 7 — FREQUENCY CALCULATION
# ===============================================================
def f_pl(m1, m2, amin, amax, Mhost):
    qmin = (m1*0.001)/Mhost
    qmax = (m2*0.001)/Mhost
    return integrate.quad(mass_fctn_pl, qmin, qmax)[0] * orb_int_pl

def f_bd(m1, m2, amin, amax, Mhost):
    qmin = (m1*0.001)/Mhost
    qmax = (m2*0.001)/Mhost
    return integrate.quad(mass_fctn_bd, qmin, qmax)[0] * orb_int_bd

mean_pl = f_pl(Jup_min, Jup_max, amin_calc, amax_calc, host_mass)
mean_bd = f_bd(Jup_min, Jup_max, amin_calc, amax_calc, host_mass)

st.write(f"**Mean Number of Planets per Star:** `{mean_pl:.10f}`")
st.write(f"**Mean Number of Brown Dwarfs per Star:** `{mean_bd:.10f}`")
