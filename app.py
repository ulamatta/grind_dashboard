import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Page configuration
st.set_page_config(page_title="Grind Comparison: Ditting vs. Colombini", layout="wide")

# Title
st.title("Grind Comparison: Ditting vs. Colombini")

# Introduction
st.markdown("""
**Eximius Coffee** is evaluating whether to invest in an $80,000 Colombini grinder or continue using our $2,000 Ditting grinder for Nespresso production. This dashboard compares grind quality to determine if the Colombini justifies its cost. Spoiler: It doesn’t.
""")

# Metrics Table
metrics_data = {
    "Metric": ["D10 (µm)", "D50 (µm)", "D90 (µm)", "Span", "% <100µm", "% >1000µm"],
    "Ditting": [44.8, 456, 827, 1.715, 16.88, 1.87],
    "Colombini Test 1": [70.1, 474, 859, 1.664, 12.49, 2.46],
    "Colombini Test 2": [52.0, 411, 754, 1.708, 15.70, 0.26],
    "Plastic Pod Sample": [39.5, 515, 930, 1.729, 16.29, 6.96]
}
metrics_df = pd.DataFrame(metrics_data)

st.subheader("Key Grind Metrics")
st.dataframe(metrics_df.style.format({
    "Ditting": "{:.2f}",
    "Colombini Test 1": "{:.2f}",
    "Colombini Test 2": "{:.2f}",
    "Plastic Pod Sample": "{:.2f}",
}))

st.markdown("""
**What These Metrics Mean:**
- **D10, D50, D90**: Particle sizes where 10%, 50%, and 90% of the volume is smaller. D50 is the median.
- **Span**: (D90 - D10) / D50. Lower = more uniform grind.
- **% <100µm**: Fines, critical for espresso crema.
- **% >1000µm**: Oversized particles, which can clog machines or alter flavor.
""")

# Cumulative Distribution Data (unchanged from original)
samples = {
    "Ditting": {
        "sizes": [10,20,30,40,60,70,80,90,100,120,140,160,180,200,250,300,350,400,500,550,600,650,700,750,800,850,900,950,1000,1100,1200,1300,1400,1500],
        "undersize": [0.19,2.65,6.04,8.89,12.73,14.08,15.19,16.12,16.88,18.00,18.74,19.27,19.77,20.40,23.05,27.64,33.84,41.23,56.84,64.00,70.49,76.19,80.97,85.21,88.41,91.31,93.36,95.14,96.51,98.36,99.23,99.75,99.94,100.00]
    },
    "Colombini Test 1": {
        "sizes": [10,20,30,40,60,70,80,90,100,120,140,160,180,200,250,300,350,400,500,550,600,650,700,750,800,850,900,950,1000,1100,1200,1300,1400],
        "undersize": [0.10,1.75,4.01,5.97,8.87,9.99,10.96,11.79,12.49,13.58,14.36,14.99,15.62,16.41,19.48,24.39,30.80,38.29,53.99,61.21,67.78,73.01,78.56,82.97,86.38,89.49,91.76,93.75,95.32,97.54,99.43,99.77,99.77]
    },
    "Colombini Test 2": {
        "sizes": [10,20,30,40,60,70,80,90,100,120,140,160,180,200,250,300,350,400,500,550,600,650,700,750,800,850,900,950,1000,1100,1200,1300],
        "undersize": [0.18,2.31,5.13,7.61,11.33,12.73,13.91,14.89,15.70,16.90,17.78,18.59,19.53,20.74,25.22,31.72,39.53,48.01,64.21,71.09,77.10,82.16,86.23,89.75,92.25,94.48,95.96,97.21,98.13,99.26,99.70,99.93,99.98]
    },
    "Plastic Pod Sample": {
        "sizes": [10,20,30,40,60,70,80,90,100,120,140,160,180,200,250,300,350,400,500,550,600,650,700,750,800,850,900,950,1000,1100,1200,1500,2900],
        "undersize": [0.25,3.39,7.22,10.13,13.54,14.62,15.52,16.29,16.96,18.02,18.77,19.28,19.65,20.00,21.38,24.15,28.47,34.22,47.98,54.92,61.49,67.60,73.01,77.93,81.94,85.62,88.44,90.97,93.01,95.04,97.75,99.52,100.00,100.00]
    }
}

# Build DataFrames for charts
cum_list = []
dens_list = []
for name, vals in samples.items():
    sizes = np.array(vals["sizes"])
    undersize = np.array(vals["undersize"])
    min_len = min(len(sizes), len(undersize))
    sizes = sizes[:min_len]
    undersize = undersize[:min_len]
    for s, u in zip(sizes, undersize):
        cum_list.append({"Machine": name, "Size": s, "Undersize": u})
    d_u = np.diff(undersize)
    d_s = np.diff(sizes)
    mids = (sizes[:-1] + sizes[1:]) / 2
    density = d_u / d_s
    for m, den in zip(mids, density):
        dens_list.append({"Machine": name, "Size": m, "Density": den})

cum_df = pd.DataFrame(cum_list)
dens_df = pd.DataFrame(dens_list)

# Charts
st.subheader("Grind Distribution Visuals")
col1, col2 = st.columns(2)
with col1:
    cum_chart = alt.Chart(cum_df).mark_line().encode(
        x=alt.X("Size:Q", scale=alt.Scale(type="log"), title="Particle Size (µm)"),
        y=alt.Y("Undersize:Q", title="% Volume Under"),
        color="Machine:N"
    ).properties(title="Cumulative Distribution")
    st.altair_chart(cum_chart, use_container_width=True)
with col2:
    dens_chart = alt.Chart(dens_df).mark_line().encode(
        x=alt.X("Size:Q", scale=alt.Scale(type="log"), title="Particle Size (µm)"),
        y=alt.Y("Density:Q", title="Approx. Volume Density"),
        color="Machine:N"
    ).properties(title="Particle Density")
    st.altair_chart(dens_chart, use_container_width=True)

# Findings and Recommendation
st.subheader("Why We Should Stick with Ditting")
st.markdown("""
### Key Findings
- **Cost**: Ditting costs **$2,000**, Colombini **$80,000**—a 40x difference.
- **Median Size (D50)**: Ditting’s 456 µm is between Colombini’s 474 µm and 411 µm, all near the Plastic Pod Sample’s 515 µm. For Nespresso, this range works fine.
- **Fines (<100µm)**: Ditting’s 16.88% matches the reference (16.29%) closely, while Colombini varies (12.49% and 15.70%). More fines = better crema, giving Ditting an edge.
- **Consistency (Span)**: All around 1.7 (Ditting 1.715, Colombini 1.664–1.708). The 0.05 difference is tiny—too small to taste.
- **Variability**: Colombini’s two tests differ (e.g., D50 shifts 63 µm), showing inconsistency. Ditting’s single test aligns with our needs.

### Why the Differences Are Small
The grind profiles are nearly identical. Look at the charts: curves overlap significantly. The Colombini’s slight span advantage (1.664 vs. 1.715) is just 3% tighter—within normal test variation, not a game-changer for coffee quality.

### Recommendation
The Ditting grinder delivers a grind perfect for Nespresso—fines-rich and consistent—for a fraction of the cost. The Colombini’s minor tweaks don’t justify $78,000 more. **Stick with Ditting.**
""")