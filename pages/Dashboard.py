import streamlit as st
import pandas as pd
import numpy as np

import math

st.title("Analytics Dashboard")
st.write("**Incident Impact Map**",font_size=34)
df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [12.840302, 80.153787],
    columns=["lat", "lon"],
)
st.map(df)




st.write("**Network Load**",font_size=34)
import pydeck as pdk

chart_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [12.840302, 80.153787],
    columns=["lat", "lon"],
)

st.pydeck_chart(
    pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=12.840302,
            longitude=80.153787,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=chart_data,
                get_position="[lon, lat]",
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=chart_data,
                get_position="[lon, lat]",
                get_color="[200, 30, 0, 160]",
                get_radius=200,
            ),
        ],
    )
)
st.write("**Call Quality Metrics**",font_size=34)
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

st.line_chart(chart_data)