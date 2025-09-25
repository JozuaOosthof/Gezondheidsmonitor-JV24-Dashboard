import streamlit as st
import pandas as pd
import geopandas as gpd
import altair as alt
import folium
from streamlit_folium import st_folium
import numpy as np
from branca.colormap import linear
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import load_data
warnings.filterwarnings('ignore')

df, df_geo, df_shp, gdf_merged, gdf, gdf_map = load_data.load_data()

df = df.drop(columns=df.columns[df.isna().sum() > 75])

st.set_page_config('Statistische Analyse', layout='wide', page_icon='🔍')

# --- Nieuwe variabelen creëren ---
df['FinancieelRisicoScore'] = df[['MoeiteMetRondkomen_1', 'WeinigControleOverGeldzaken_2', 'HeeftSchulden_3']].mean(axis=1)
df['MentaleGezondheidsScore'] = df[['GoedErvarenMentaleGezondheid_12', 'AngstDepressiegevoelensAfg4Weken_13', 'BeperktDoorPsychischeKlachten_14']].mean(axis=1)

bins = [0, 10, 30, 100]
labels = ['Laag', 'Gemiddeld', 'Hoog']
df['MoeiteMetRondkomenCat'] = pd.cut(df['MoeiteMetRondkomen_1'], bins=bins, labels=labels, right=False)


st.subheader('Statistische Analyse: Correlatie en Regressie')
with st.container(border=True):
    st.write("#### Correlatie tussen Financiën en Gezondheid")
    corr_vars = [
        'FinancieelRisicoScore', 
        'MentaleGezondheidsScore', 
        'GoedErvarenMentaleGezondheid_12',
        'MoeiteMetRondkomen_1', 
        'WeinigControleOverGeldzaken_2', 
        'HeeftSchulden_3', 
        'ZorgenOverStudieschuld_5'
    ]
    
    corr_matrix = df[corr_vars].corr()
    st.dataframe(corr_matrix)
    
    st.write("#### Meervoudige Lineaire Regressie")
    st.write("Dit model voorspelt 'GoedErvarenMentaleGezondheid' op basis van diverse financiële variabelen.")

    financial_vars = [
        'MoeiteMetRondkomen_1', 
        'WeinigControleOverGeldzaken_2', 
        'HeeftSchulden_3', 
        'ZorgenOverStudieschuld_5'
    ]
    
    X = df[financial_vars].dropna()
    y = df.loc[X.index]['GoedErvarenMentaleGezondheid_12']

    if not X.empty and len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write(f"R-kwadraat (R²) score: {r2_score(y_test, y_pred):.2f}")
        st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")

        st.write("#### Coëfficiënten van de variabelen")
        coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coëfficiënt'])
        st.dataframe(coefficients)
        st.write(f"Intercept: {model.intercept_:.2f}")

    else:
        st.warning("De geselecteerde variabelen bevatten te veel missende waarden voor regressie-analyse.")