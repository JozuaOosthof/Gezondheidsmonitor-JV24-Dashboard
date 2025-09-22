import altair as alt
import pandas as pd
import streamlit as st

def load_data():
    df = pd.read_csv('50140NED_UntypedDataSet_16092025_134851.csv')
    return df

df = load_data()

st.set_page_config('Gezondheidsmonitor 2024 Dashboard', layout = 'wide')
st.title('Gezondheidsmonitor 2024 Dashboard')

st.expander('Gebruikte Data', expanded = False).write(df)