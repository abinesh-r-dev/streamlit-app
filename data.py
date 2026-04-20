"""
data.py
Cached loaders for the basin station dataset and IREL mineralogical analysis.
"""

import streamlit as st
import pandas as pd
from config import K40_DEFAULT


@st.cache_data
def load_basin() -> pd.DataFrame:
    """Ten sampling stations along the Thamirabarani River transect."""
    return pd.DataFrame({
        "id":      ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10"],
        "station": ["Pechiparai","Kaliyal","Thirparappu","Moovattumugham",
                    "Thickurichy","Vayakalloor","Parakanni","Thengapattanam",
                    "Ponmanai","Surulacode"],
        "type":    ["Main"]*8 + ["Branch"]*2,
        "lat":     [8.4442,8.4100,8.3920,8.3600,8.3118,
                    8.2900,8.2650,8.2384,8.3591,8.3500],
        "lon":     [77.3147,77.2900,77.2560,77.2500,77.2278,
                    77.1720,77.1710,77.1698,77.3305,77.3600],
        "geology": ["Granite","Alluvial","Feldspar","Quartz","Sedimentary",
                    "Monazite Sand","Ilmenite","Thorium Sand","Clay","Laterite"],
        "monazite": [0.02,0.18,0.01,0.07,0.01,0.24,0.62,0.22,0.04,0.10],
        "s3_imp":  [False,False,True,False,False,False,False,False,False,False],
        "k40":     [K40_DEFAULT]*10,
    })


@st.cache_data
def load_irel() -> pd.DataFrame:
    """IREL QC Lab mineralogical analysis — NABL accredited, report dated 18.02.2026."""
    return pd.DataFrame({
        "id":         ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10"],
        "station":    load_basin()["station"].tolist(),
        "Ilmenite":   [3.69,1.08,1.54,6.80,0.80,1.77,2.62,9.54,6.10,7.09],
        "Rutile":     [0.17,0.10,0.04,0.12,0.06,0.01,0.06,0.52,0.11,0.50],
        "Zircon":     [0.04,0.02,0.07,0.13,0.26,0.08,0.16,0.71,0.16,0.30],
        "Monazite":   [0.08,0.04,0.10,0.07,0.18,0.02,0.62,0.22,0.24,0.22],
        "Garnet":     [4.68,1.12,1.44,10.84,0.92,5.00,3.87,3.01,0.54,3.42],
        "Sillimanite":[3.92,2.90,2.29,2.78,1.51,4.19,0.53,1.82,0.60,2.61],
        "Kyanite":    [0.01,0.03,0.01,0.07,0.02,0.01,0.02,0.02,0.01,0.07],
        "Leucoxene":  [0.11,0.02,0.02,0.03,0.05,0.02,0.40,0.30,0.27,0.44],
        "Others":     [0.09,0.11,0.18,0.50,0.07,1.14,0.04,0.01,0.01,0.03],
        "Heavies":    [12.79,5.42,5.59,21.34,3.87,12.24,7.80,16.55,8.04,14.68],
    })
