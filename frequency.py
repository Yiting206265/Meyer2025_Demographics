"""
Occurrence Rate Estimator for Giant Planets and Brown Dwarfs

This Streamlit application calculates and visualizes the frequency of giant planets and 
brown dwarfs around different stellar types based on a companion population model.
The model uses log-normal distributions for orbital separations and power-law distributions
for mass ratios to estimate companion frequencies.

Author: Yiting Li
Date: August 2025
Paper Reference: Meyer Li et al. 2025, Demographics of Planetary and Brown Dwarf Companions
"""

import numpy as np
import streamlit as st
from scipy import integrate
from scipy import interpolate
import operator
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ln10 = np.log(10)  # Natural log of 10 for proper normalization of log10 distributions

# --------------------------------------------------------------
# GLOBAL CSS + TITLE BLOCK (unchanged)
# --------------------------------------------------------------
st.markdown("""
<style>
.title-container {
    background: linear-gradient(to right, #1E88E5, #5E35B1);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 25px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.main-title {
    color: white;
    font-size: 36px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}
.subtitle {
    color: rgba(255,255,255,0.9);
    text-align: center;
    font-size: 18px;
    font-style: italic;
}
.section-header {
    color: #1E88E5;
    font-size: 28px;
    font-weight: bold;
    margin: 30px 0 20px 0;
    padding: 10px 0;
    border-bottom: 2px solid #1E88E5;
}
.tool-header {
    color: white;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(to right, #1E88E5, #5E35B1);
    padding: 12px;
    border-radius: 5px;
    margin-top: 25px;
}
</style>

<div class='title-container'>
    <div class='main-title'>Occurrence Rate Estimator</div>
    <div class='subtitle'>for Planets and Brown Dwarfs</div>
    <div class='subtitle'>Companion population model from Meyer Li et al. (2025)</div>
</div>
""", unsafe_allow_html=True)



# --------------------------------------------------------------
# INTRO (unchanged)
# --------------------------------------------------------------

st.write("""Welcome to the on-line tool based on Meyer Li et al. (arxiv: https://arxiv.org/abs/2508.05122) meant to provide estimates of the expectation values of the mean number of gas giant planets per star and the mean number of brown dwarfs per star generated from our model. The model assumes that the companion mass ratio of gas giants and brown dwarf companions does not vary with orbital separation. However, it explicitly treats brown dwarf companions as an extension of stellar mass companions drawn from the same orbital separations as a function of host star mass.

In the paper we fit the orbital distribution of gas giants and find that a log-normal function provides a good fit, with a peak near 3.8 AU (two parameters). We also fit for power-law exponents for the companion mass ratio distributions for the brown dwarf companions and gas giant populations separately (two parameters). Finally, we fit for the normalization of both populations (two parameters).

Note: The data are fitted in the natural ln but we present the results in log-10 here for consistency with stellar binary orbital distributions.""")



# --------------------------------------------------------------
# MATHEMATICAL MODEL (unchanged)
# --------------------------------------------------------------

st.markdown("<div class='section-header'>Mathematical Model</div>", unsafe_allow_html=True)

st.write(r"""
Our model combines two components: gas giant planets and brown dwarf companions.  
The total companion frequency is:
""")

st.latex(r"""
N_{\text{TOTAL}} = 
\int \phi_{pl}(x)\,\psi_{pl}(q)\,dq\,dx 
+ 
\int \phi_{bd}(x)\,\psi_{bd}(q)\,dq\,dx
""")

st.write("Mass-ratio distributions (power-law):")

st.latex(r"\psi_{pl}(q) = q^{-\alpha}")
st.latex(r"\psi_{bd}(q) = q^{-\beta}")

st.write("Orbital separation distributions (log-normal):")

st.latex(r"\phi_{pl}(x) = \frac{A_{pl}\,e^{-(x-\mu_{pl})^2/(2\sigma_{pl}^2)}}{x\sqrt{2\pi}\sigma_{pl}\ln 10}")

st.latex(r"\phi_{bd}(x) = \frac{A_{bd}\,e^{-(x-\mu_{bd})^2/(2\sigma_{bd}^2)}}{x\sqrt{2\pi}\sigma_{bd}\ln 10}")

st.write("""
where  \(x = \log_{10}(a)\).  
The brown dwarf orbital distributions depend on stellar type (M / FGK / A).  
""")



# --------------------------------------------------------------
# TABLE (unchanged)
# --------------------------------------------------------------

st.write("**Table 1: Companion Frequency (CF) & Log-Normal Separation Distribution vs. Host Type**")

st.table({
    'Spectral Type': ['M', 'FGK', 'A'],
    'CF': ['0.236', '0.61', '0.219'],
    'μ (log10 AU)': ['1.43', '1.70', '2.72'],
    'σ (log10 AU)': ['1.21', '1.68', '0.79']
})

st.write("""
The model integrates these distributions over mass ratio and orbital separation to obtain expectation values of companion frequencies.  
""")


st.markdown("---")
st.markdown("<div class='tool-header'>Frequency Calculation Tool</div>", unsafe_allow_html=True)



# --------------------------------------------------------------
# SECTION 2 — HOST STAR PARAMETERS (updated)
# --------------------------------------------------------------

st.markdown("<div class='section-header'>Host Star Parameters</div>", unsafe_allow_html=True)

ln_A_bd_default = -1.407
ln_A_pl_default = -4.720
alpha_bd_default = -0.292
alpha_gp_default = 1.296
mu_pl_default = 1.299
sigma_pl_default = np.exp(0.215)

# Stellar Type
st_type = st.radio(
    "Select stellar spectral type:",
    ("M Dwarfs", "FGK", "A Stars"),
    index=1
)

# Dynamic defaults + unique keys
if st_type == "M Dwarfs":
    mu_bd_default = 1.43
    s_bd_default = 1.21
    suffix = "M"
elif st_type == "FGK":
    mu_bd_default = 1.70
    s_bd_default = 1.68
    suffix = "FGK"
else:
    mu_bd_default = 2.72
    s_bd_default = 0.79
    suffix = "A"

# Host mass
host_mass = st.number_input(
    "Host Mass (M☉)",
    min_value=0.0001, max_value=10.0, value=1.0, step=0.01
)



# --------------------------------------------------------------
# SECTION 3 — MODEL PARAMETERS (updated)
# --------------------------------------------------------------

st.markdown("<div class='section-header'>Model Parameters</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    alpha_bd = st.slider("β (BD slope)", -3.0, 3.0, alpha_bd_default, 0.01)
    A_bd = st.slider("A_bd", 0.0001, 1.0, np.exp(ln_A_bd_default)/ln10, 0.0001)
    mean_bd = st.slider("log10(μ_bd)", 0.0, 3.0, mu_bd_default, 0.01, key=f"mu_bd_{suffix}")
    sigma_bd = st.slider("log10(σ_bd)", 0.0, 3.0, s_bd_default, 0.01, key=f"sigma_bd_{suffix}")

with col2:
    alpha_gp = st.slider("α (Planet slope)", -3.0, 3.0, alpha_gp_default, 0.01)
    A_pl = st.slider("A_pl", 0.0001, 0.1,
                     np.exp(ln_A_pl_default)/ln10,
                     0.0001,
                     format="%.6f")
    mu_pl = st.slider("log10(μ_pl)", 0.0, 3.0, mu_pl_default/ln10, 0.01)
    sigma_pl = st.slider("log10(σ_pl)", 0.0, 3.0, sigma_pl_default/ln10, 0.01)



# --------------------------------------------------------------
# EXACT SURFACE DENSITY FUNCTIONS
# --------------------------------------------------------------

def surface_den_bd_exact(a):
    return (
        A_bd * np.exp(-(np.log10(a)-mean_bd)**2/(2*sigma_bd**2))
        / (a*np.sqrt(2*np.pi)*sigma_bd)
    )

def surface_den_pl_exact(a):
    return (
        A_pl * np.exp(-(np.log10(a)-mu_pl)**2/(2*sigma_pl**2))
        / (a*np.sqrt(2*np.pi)*sigma_pl)
    )



# --------------------------------------------------------------
# SECTION 4 — MASS RANGE (unchanged)
# --------------------------------------------------------------

st.markdown("<div class='section-header'>Companion Parameters</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    Jup_min = st.number_input("Minimum Mass (MJup)", 0.03, 4000.0, 1.0)

with c2:
    Jup_max = st.number_input("Maximum Mass (MJup)", 0.03, 4000.0, 85.0)

if Jup_min >= Jup_max:
    st.error("Minimum mass must be less than maximum mass.")
    st.stop()

q_Jupiter = 0.001 / host_mass



# --------------------------------------------------------------
# SECTION 5 — ORBITAL SEPARATION (fix: st.stop)
# --------------------------------------------------------------

st.markdown("<div class='section-header'>Orbital Separation Range</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    amin_calc = st.number_input("Minimum Separation (AU)", 0.1, 3000.0, 1.0)
with c2:
    amax_calc = st.number_input("Maximum Separation (AU)", 0.1, 3000.0, 100.0)

if amin_calc >= amax_calc:
    st.error("Minimum separation must be less than maximum separation.")
    st.stop()

# Precompute integrals once
orb_bd = integrate.quad(surface_den_bd_exact, amin_calc, amax_calc)[0]
orb_pl = integrate.quad(surface_den_pl_exact, amin_calc, amax_calc)[0]

def mass_fctn_bd(q): return q**(-alpha_bd)
def mass_fctn_pl(q): return q**(-alpha_gp)
def dN_bd(q): return mass_fctn_bd(q)*orb_bd
def dN_pl(q): return mass_fctn_pl(q)*orb_pl



##############################################################################
# Section 6 - Orbital Separation Range for Frequency Calculations
##############################################################################

# Create columns for orbital separation inputs only
col1, col2 = st.columns(2)

with col1:
    amin_calc = st.number_input(
        'Minimum Separation (AU)',
        min_value=0.1,
        max_value=3000.0,
        value=1.0,
        step=0.1,
        help="Minimum orbital separation in AU"
    )

with col2:
    amax_calc = st.number_input(
        'Maximum Separation (AU)',
        min_value=0.1,
        max_value=3000.0,
        value=100.0,
        step=0.1,
        help="Maximum orbital separation in AU"
    )

# Validate orbital separation range
if amin_calc >= amax_calc:
    st.error("⚠️ Minimum separation must be less than maximum separation!")
    st.stop()


# ======================================================================
# Frequency Calculation Functions (Planets + BDs)
# ======================================================================

def f_pl(mass_min_mj, mass_max_mj, sep_min_au, sep_max_au, host_mass_msun):
    """
    Calculate the mean number of giant planets per star.
    """
    M_J = 1.0 / 1000.0  # Jupiter mass ratio
    qmin = (mass_min_mj * M_J) / host_mass_msun
    qmax = (mass_max_mj * M_J) / host_mass_msun

    mass_integral = integrate.quad(mass_fctn_pl, qmin, qmax)[0]
    surf_integral = integrate.quad(surface_den_pl, sep_min_au, sep_max_au)[0]
    return mass_integral * surf_integral


def f_bd(mass_min_mj, mass_max_mj, sep_min_au, sep_max_au, host_mass_msun):
    """
    Calculate the mean number of brown dwarfs per star.
    """
    M_J = 1.0 / 1000.0  # Jupiter mass ratio
    qmin = (mass_min_mj * M_J) / host_mass_msun
    qmax = (mass_max_mj * M_J) / host_mass_msun

    mass_integral = integrate.quad(mass_fctn_bd, qmin, qmax)[0]
    surf_integral = integrate.quad(surface_den_bd, sep_min_au, sep_max_au)[0]
    return mass_integral * surf_integral


##############################################################################
# Section 5 - Plotting Section (moved here after orbital separation inputs)
##############################################################################

# Define specific mass ratio ranges for each population
q_pl_min = 0.03 * q_Jupiter     # Planets: lower mass, lower limit
q_pl_max = 0.1
q_bd_min = 3 * q_Jupiter
q_bd_max = 0.67

bd_freq = []
pl_freq = []
total_freq = []

# Define the range for mass ratios using correct log10 ranges
mass_ratio_values_pl = np.logspace(
    np.log10(q_pl_min),
    np.log10(q_pl_max),
    1000
)

mass_ratio_values_bd = np.logspace(
    np.log10(q_bd_min),
    np.log10(q_bd_max),
    1000
)

mass_ratio_values_total = np.logspace(
    np.log10(q_pl_min),
    np.log10(q_bd_max),
    1000
)

# Compute frequency curves
for q in mass_ratio_values_pl:
    pl_val = dN_pl(q) * q * np.log(10) if q_pl_min <= q <= q_pl_max else 0
    pl_freq.append(pl_val)

for q in mass_ratio_values_bd:
    bd_val = dN_bd(q) * q * np.log(10) if q_bd_min <= q <= q_bd_max else 0
    bd_freq.append(bd_val)

for q in mass_ratio_values_total:
    bd_val = dN_bd(q) * q * np.log(10) if q_bd_min <= q <= q_bd_max else 0
    pl_val = dN_pl(q) * q * np.log(10) if q_pl_min <= q <= q_pl_max else 0
    total_freq.append(bd_val + pl_val)

# Convert lists to arrays
bd_freq_array = np.array(bd_freq)
pl_freq_array = np.array(pl_freq)
total_freq_array = np.array(total_freq)

# Plot the frequency distribution
ax.plot(np.log10(mass_ratio_values_pl), pl_freq_array, color='r', linewidth=3, label='Giant Planet Model')
ax.plot(np.log10(mass_ratio_values_bd), bd_freq_array, color='blue', linewidth=3, label='Brown Dwarf Model')
ax.plot(np.log10(mass_ratio_values_total), total_freq_array, color='orange', linewidth=2, linestyle='-', label='Total Frequency')

# Axis labels
ax.set_xlabel('log(q)', fontsize=20)
ax.set_ylabel('dN / dlog(q)', fontsize=20)

# X-axis autoscaling with padding
try:
    valid_mask = ~np.isnan(total_freq_array)
    if np.any(valid_mask):
        valid_x_pl = np.log10(mass_ratio_values_pl)[valid_mask]
        valid_x_bd = np.log10(mass_ratio_values_bd)[valid_mask]
        x_min = np.min(valid_x_pl)
        x_max = np.max(valid_x_bd)
        pad = (x_max - x_min) * 0.11
        ax.set_xlim(x_min - pad, x_max + pad)
    else:
        ax.set_xlim(-4, -0.5)
except:
    ax.set_xlim(-4, -0.5)

# Y-axis autoscaling
try:
    max_y = np.nanmax(total_freq_array)
    ax.set_ylim(0, max_y * 1.2)
except:
    ax.set_ylim(0, 0.3)

ax.tick_params(axis='both', which='major', labelsize=15)
ax.legend(loc='upper right', fontsize=14)
ax.set_title("Companion Frequency Distribution", pad=10, fontsize=22)

# Display the plot
st.pyplot(fig)

##############################################################################
# Section 7 - Frequency Calculations
##############################################################################

mean_num_pl = f_pl(Jup_min, Jup_max, amin_calc, amax_calc, host_mass)
mean_num_bd = f_bd(Jup_min, Jup_max, amin_calc, amax_calc, host_mass)

# Display results
st.write(f"Mean Number of Planets Per Star: `{mean_num_pl:.10f}`")
st.write(f"Mean Number of Brown Dwarfs Per Star: `{mean_num_bd:.10f}`")

st.write("")
st.write("*Note: These values represent the expected number of companions per star within the specified mass ratio and orbital separation ranges.*")


st.write("")
st.write("*Note: These values represent the expected number of companions per star within the specified mass ratio and orbital separation ranges.*")
