import streamlit as st
import pandas as pd
import geopandas as gpd
import altair as alt
import folium
from streamlit_folium import st_folium
import numpy as np
import json
from vega_datasets import data as vds
from branca.colormap import linear

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv('50140NED_TypedDataSet_22092025_192544.csv', sep=';')
    df_geo = pd.read_csv("georef-netherlands-gemeente.csv", sep=";", low_memory=False, on_bad_lines="skip")
    df_shp = gpd.read_file('georef-netherlands-gemeente-millesime.shp')
    return df, df_geo, df_shp

df, df_geo, df_shp = load_data()

gemeente_to_prov = df_geo.set_index("Gemeente code (with prefix)")["Provincie name"].to_dict()
df_geo["Provincie"] = df_geo["Gemeente code (with prefix)"].apply(lambda x: gemeente_to_prov.get(x, "Onbekend"))

df = df.merge(
    df_geo[["Gemeente code (with prefix)", "Provincie"]],
    left_on="RegioS", right_on="Gemeente code (with prefix)", how="left"
)

df['RegioS'] = df['RegioS'].astype(str).str.strip()
df_shp['gem_code'] = df_shp['gem_code'].astype(str).str.strip()
gdf_merged = df_shp.merge(df, left_on='gem_code', right_on='RegioS', how='left')

st.set_page_config('Gezondheidsmonitor 2024 Dashboard', layout='wide')

st.title('Gezondheidsmonitor 2024 Dashboard')

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(label='Aantal Waarnemingen', value=len(df))
with c2:
    st.metric(label='Aantal Waarnemingen', value=len(df))
with c3:
    st.metric(label='Aantal Waarnemingen', value=len(df))
with c4:
    st.metric(label='Aantal Waarnemingen', value=len(df))

st.divider()

sidebar = st.sidebar.header('Gezondheidsmonitor 2024 Dashboard')
with sidebar:
    page_sb = st.selectbox('Selecteer Pagina', ['Intro & Data', 'Visualisaties'])

st.subheader('Visualisaties')

# --- Scatterplot ---
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        ms_scatter = st.multiselect('Selecteer Variabelen voor de Scatterplot', df.columns.tolist(), default=['MoeiteMetRondkomen_1', 'HeeftSchulden_3'])
        sb_scatter = st.selectbox('Kleur op', ['Geen', 'Provincie', 'Gemeente'], index=0)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            x_axis = st.selectbox('X-as', ms_scatter, index=ms_scatter.index('MoeiteMetRondkomen_1'))
        with col_s2:
            y_axis = st.selectbox('Y-as', ms_scatter, index=ms_scatter.index('HeeftSchulden_3'))
        add_regression = st.checkbox('Regressielijn', value=True)

        scatter = alt.Chart(df).mark_circle().encode(
            x=alt.X(f'{x_axis}:Q', title=x_axis, scale=alt.Scale(zero=False)),
            y=alt.Y(f'{y_axis}:Q', title=y_axis, scale=alt.Scale(zero=False)),
            color=alt.Color(f'{sb_scatter}:N') if sb_scatter != 'Geen' else alt.value('steelblue'),
            tooltip=[x_axis, y_axis]
        ).properties(width=800, height=800).interactive()

        if add_regression:
            scatter += scatter.transform_regression(x_axis, y_axis).mark_line(color='red')

        st.altair_chart(scatter, use_container_width=True)

# --- Bar chart ---
with col2:
    with st.container(border=True):
        bar_options = [col for col in df.columns if col not in ['ID', 'RegioS', 'Persoonskenmerken', 'Marges', 'Provincie']]
        selected_bar = st.selectbox('Selecteer Variabele', bar_options, index=0)

        if pd.api.types.is_numeric_dtype(df[selected_bar]):
            bar_data = df.groupby("Provincie", as_index=False)[selected_bar].mean()
        else:
            bar_data = df.groupby("Provincie", as_index=False)[selected_bar].count()

        bar_chart = alt.Chart(bar_data).mark_bar().encode(
            x=alt.X("Provincie:N", sort='-y', title="Provincie"),
            y=alt.Y(f"{selected_bar}:Q", title=selected_bar),
            tooltip=["Provincie:N", f"{selected_bar}:Q"]
        ).properties(width=700, height=400, title=f"{selected_bar} per provincie")

        st.altair_chart(bar_chart, use_container_width=True)

    # --- Boxplot ---
    with st.container(border=True):
        provinces = df['Provincie'].dropna().unique().tolist()
        box_ms = st.multiselect('Selecteer Provincies', provinces, default=provinces[:3])
        box_sb = st.selectbox('X-as Boxplot', bar_options, index=0)
        df_filtered = df[df['Provincie'].isin(box_ms)]
        n_provs = max(len(box_ms), 1)
        box_size = max(10, 200 // n_provs)
        box = alt.Chart(df_filtered).mark_boxplot(size=box_size).encode(
            x=alt.X(f'{box_sb}:Q', title=box_sb, scale=alt.Scale(zero=False)),
            y=alt.Y('Provincie:N', title='Provincie'),
            color=alt.Color('Provincie:N', legend=None)
        ).properties(height=392)
        st.altair_chart(box, use_container_width=True)

# --- Histogram ---
with st.container(border=True):
    hist_ms = st.multiselect('Selecteer Provincies voor Histogram', provinces, default=provinces[:3], key='hist_ms')
    selected_hist = st.selectbox('X-as Histogram', bar_options, index=0, key='hist_select')
    kleur = st.checkbox('Kleur op Provincie', value=False)
    df_hist_filtered = df[df['Provincie'].isin(hist_ms)]
    hist = alt.Chart(df_hist_filtered).mark_bar().encode(
        x=alt.X(f"{selected_hist}:Q", bin=alt.Bin(maxbins=30), title=f'{selected_hist}'),
        y=alt.Y('count()', title='Aantal'),
        color=alt.Color('Provincie:N') if kleur else alt.value('steelblue')
    )
    st.altair_chart(hist, use_container_width=True)

col10, col11 = st.columns(2)

with col10:
    with st.container(border=True):
        st.subheader("Stacked Bar per Provincie")

        stack_vars = st.multiselect(
            'Selecteer Variabelen voor Stacked Bar',
            [col for col in bar_options if pd.api.types.is_numeric_dtype(df[col])],
            default=[bar_options[0]]
        )

        if stack_vars:
            stack_data = df.groupby("Provincie")[stack_vars].mean().reset_index()

            stacked_bar = (
                alt.Chart(stack_data, title="Stacked bar")
                .transform_fold(stack_vars, as_=['Variable', 'Value'])
                .mark_bar()
                .encode(
                    x=alt.X('Provincie:N', title="Provincie"),
                    y=alt.Y('Value:Q', title="Gemiddelde waarde"),
                    color=alt.Color('Variable:N', legend=alt.Legend(orient='bottom')),
                    tooltip=['Provincie:N', 'Variable:N', 'Value:Q']
                )
                .properties(width=700, height=400)
            )

            st.altair_chart(stacked_bar, use_container_width=True)
        else:
            st.info("Selecteer minimaal één numerieke variabele om de stacked bar te tonen.")

st.expander('Gebruikte Data', expanded=False).write(df)

df_map = df[['RegioS', 'MoeiteMetRondkomen_1']].copy()
df_map = df_map.rename(columns={'MoeiteMetRondkomen_1':'val'})

gdf = gpd.read_file("gemeente_gegeneraliseerd.geojson")[['statcode','statnaam','geometry']]
gdf = gdf.merge(df_map, left_on='statcode', right_on='RegioS', how='left')

def fill_with_neighbors(row, gdf):
    if pd.notna(row['val']):
        return row['val']
    neighbors = gdf[gdf.geometry.touches(row['geometry'])]
    if len(neighbors) > 0:
        return neighbors['val'].mean()
    return np.nan

gdf['val'] = gdf.apply(lambda row: fill_with_neighbors(row, gdf), axis=1)
gdf['val'] = gdf['val'].fillna(gdf['val'].mean())

@st.cache_resource
def create_folium_map(_gdf):
    map_sb = st.selectbox('Kies variabele voor de kaart', ['MoeiteMetRondkomen_1'], index=0, key='map_sb')

    m = folium.Map(location=[52.1, 5.3], zoom_start=7)
    
    colormap = linear.Blues_09.scale(_gdf['val'].min(), _gdf['val'].max())
    colormap.caption = 'Moeite Met Rondkomen (%)'
    colormap.add_to(m)

    folium.GeoJson(
        _gdf,
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['val']),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.8
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['statnaam', 'statcode', 'val'],
            aliases=['Gemeente:', 'Code:', 'Moeite met rondkomen:'],
            localize=True
        )
    ).add_to(m)
    return m

with st.container(border=True):
    m = create_folium_map(gdf)
    st_folium(m, width=700, height=800)
