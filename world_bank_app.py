import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import xml.etree.ElementTree as ET
import numpy as np

# page configuration l
st.set_page_config(
    page_title="World Development Map",
    page_icon="🌍",
    layout="wide"
)

# fetch and merge data from World Bank API
@st.cache_data
def load_and_merge_data():
    gdp_id = 'NY.GDP.PCAP.CD'
    life_exp_id = 'SP.DYN.LE00.IN'
    pop_id = 'SP.POP.TOTL'
    def fetch_world_bank_data(indicator_id):
        url = f"http://api.worldbank.org/v2/country/all/indicator/{indicator_id}?date=1960:2025&format=xml&per_page=20000"
        response = requests.get(url)
        if response.status_code != 200: return None
        root = ET.fromstring(response.content)
        namespace = {'wb': 'http://www.worldbank.org'}
        data = []
        for record in root.findall('wb:data', namespace):
            country_name = record.find('wb:country', namespace).text
            country_code = record.find('wb:countryiso3code', namespace).text
            year = record.find('wb:date', namespace).text
            value = record.find('wb:value', namespace).text
            if value is not None and float(value) > 0:
                data.append({'Country Name': country_name, 'Country Code': country_code, 'Year': int(year), indicator_id: float(value)})
        return pd.DataFrame(data)

    gdp_df = fetch_world_bank_data(gdp_id)
    life_exp_df = fetch_world_bank_data(life_exp_id)
    pop_df = fetch_world_bank_data(pop_id)
    if gdp_df is None or life_exp_df is None or pop_df is None: return None
    gdp_df.rename(columns={gdp_id: 'GDP per capita'}, inplace=True)
    life_exp_df.rename(columns={life_exp_id: 'Life Expectancy'}, inplace=True)
    pop_df.rename(columns={pop_id: 'Population'}, inplace=True)
    merged_df = pd.merge(gdp_df, life_exp_df, on=['Country Name', 'Country Code', 'Year'])
    final_df = pd.merge(merged_df, pop_df, on=['Country Name', 'Country Code', 'Year'])
    return final_df.dropna()

df = load_and_merge_data()

# layout of the app
st.title('World Development Indicators Map')
st.markdown('Select an indicator and year to visualize how the world has changed over time.')

if df is None:
    st.error("Failed to load data from the World Bank API. Please try again later.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        indicator_options = ['GDP per capita', 'Life Expectancy', 'Population']
        selected_indicator = st.selectbox('Select Indicator', indicator_options)
    with col2:
        country_list = ["Worldwide Average"] + sorted(df['Country Name'].unique())
        selected_country = st.selectbox('Select Country', country_list)
    with col3:
        years = sorted(df['Year'].unique())
        selected_year = st.select_slider('Select Year', options=years, value=years[-1])
    
    st.divider()
    
    # Metric display for like the country or worldwide average
    df_current_year = df[df['Year'] == selected_year]
    
    if selected_country == "Worldwide Average":
        df_previous_year = df[df['Year'] == selected_year - 1]
        if not df_current_year.empty:
            avg_current = df_current_year[selected_indicator].mean()
            avg_previous = df_previous_year[selected_indicator].mean() if not df_previous_year.empty else 0
            delta = avg_current - avg_previous
            st.metric(
                label=f"Worldwide Average {selected_indicator}",
                value=f"{avg_current:,.2f}",
                delta=f"{delta:,.2f} from previous year"
            )
    else:
        current_data = df_current_year[df_current_year['Country Name'] == selected_country]
        df_previous_year = df[df['Year'] == selected_year - 1]
        previous_data = df_previous_year[df_previous_year['Country Name'] == selected_country]
        if not current_data.empty:
            current_value = current_data[selected_indicator].iloc[0]
            previous_value = previous_data[selected_indicator].iloc[0] if not previous_data.empty else 0
            delta = current_value - previous_value
            st.metric(
                label=f"{selected_indicator} in {selected_country}",
                value=f"{current_value:,.2f}",
                delta=f"{delta:,.2f} from previous year"
            )

    # map creation
    color_axis_config = {'colorscale': px.colors.sequential.Plasma}
    color_variable = selected_indicator

    if selected_indicator in ['GDP per capita', 'Population']:
        df_current_year = df_current_year.copy()
        df_current_year['log_indicator'] = np.log10(df_current_year[selected_indicator])
        color_variable = 'log_indicator'
        color_axis_config['colorbar'] = dict(
            title=f"{selected_indicator} (Log Scale)",
            tickvals=[3, 4, 5, 6, 7, 8, 9, 10], 
            ticktext=["1k", "10k", "100k", "1M", "10M", "100M", "1B", "10B"]
        )

    fig = px.choropleth(
        df_current_year,
        locations='Country Code',
        color=color_variable,
        custom_data=[selected_indicator],
        hover_name='Country Name',
        template='plotly_dark'
    )
    
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>' + f'{selected_indicator}: %{{customdata[0]:,.2f}}'
    )

    fig.update_layout(
        title=dict(text=f"<b>{selected_indicator} in {selected_year}</b>", x=0.5, y=0.95, xanchor='center'),
        margin=dict(l=0, r=0, t=40, b=0),
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            showframe=False,
            projection_type='natural earth'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis=color_axis_config
    )
    
    st.plotly_chart(fig, use_container_width=True)