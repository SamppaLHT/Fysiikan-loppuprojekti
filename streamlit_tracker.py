import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

st.set_page_config(page_title="Kävelydatan Analyysi", layout="wide")
st.title("Kävelydatan Analyysi")

# Lataa data
@st.cache_data
def load_data():
    # GitHub raw URLs for data files
    base_url = 'https://raw.githubusercontent.com/SamppaLHT/Fysiikan-loppuprojekti/refs/heads/main/'
    acc_data = pd.read_csv(base_url + 'Data/Accelerometer.csv')
    loc_data = pd.read_csv(base_url + 'Data/Location.csv')
    
    cutoff_time = 5.0
    acc_data_filtered = acc_data[acc_data['Time (s)'] >= cutoff_time].copy()
    acc_data_filtered['Time (s)'] = acc_data_filtered['Time (s)'] - cutoff_time
    
    loc_data_filtered = loc_data[loc_data['Time (s)'] >= cutoff_time].copy()
    loc_data_filtered['Time (s)'] = loc_data_filtered['Time (s)'] - cutoff_time
    
    return acc_data_filtered, loc_data_filtered

acc_data, loc_data = load_data()

# Z-komponentti (eteen-taakse suuntainen kiihtyvyys)
z_acceleration = acc_data['Acceleration z (m/s^2)'].values
time_data = acc_data['Time (s)'].values

# Butterworth-alipäästösuodatin
def apply_lowpass_filter(data, cutoff_freq, sampling_rate, order=4):
    nyquist = sampling_rate / 2
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

sampling_rate = len(time_data) / time_data[-1]
cutoff_frequency = 2.0 
z_filtered = apply_lowpass_filter(z_acceleration, cutoff_frequency, sampling_rate)


height_threshold = 7.0  # Kynnysarvo askelten tunnistamiseen
distance_samples = int(0.12 * sampling_rate)  # Vähimmäisetäisyys piikkien välillä (120 ms)
peaks, properties = signal.find_peaks(z_filtered, height=height_threshold, distance=distance_samples)

steps_filtered = len(peaks)

# Suorita FFT
N = len(z_acceleration)
yf = fft(z_acceleration)
xf = fftfreq(N, 1/sampling_rate)

# Ota vain positiiviset taajuudet
positive_freq_mask = xf > 0
xf_positive = xf[positive_freq_mask]
yf_positive = np.abs(yf[positive_freq_mask])

# Rajoita kävelyyn sopivaan taajuusalueeseen (0.5 - 4 Hz)
walking_freq_mask = (xf_positive >= 0.5) & (xf_positive <= 4.0)
xf_walking = xf_positive[walking_freq_mask]
yf_walking = yf_positive[walking_freq_mask]

# Etsi dominoiva taajuus
dominant_freq_idx = np.argmax(yf_walking)
dominant_frequency = xf_walking[dominant_freq_idx]

# Laske askelten määrä Fourier-analyysistä
steps_fourier = int(dominant_frequency * time_data[-1])

# GPS-datan analyysi
def calculate_distance(lat1, lon1, lat2, lon2):
    """Laske etäisyys kahden GPS-pisteen välillä (Haversine-kaava)"""
    R = 6371000  # Maapallon säde metreinä
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    distance = R * c
    return distance

# Laske kuljettu matka GPS-datasta
latitudes = loc_data['Latitude (°)'].values
longitudes = loc_data['Longitude (°)'].values
gps_time = loc_data['Time (s)'].values

total_distance = 0
for i in range(len(latitudes) - 1):
    dist = calculate_distance(latitudes[i], longitudes[i], 
                              latitudes[i+1], longitudes[i+1])
    total_distance += dist

# Laske keskinopeus
total_time = gps_time[-1] - gps_time[0]  # sekunteina
average_speed = total_distance / total_time  # m/s
average_speed_kmh = average_speed * 3.6  # km/h

# Laske askelpituus molemmilla menetelmillä
step_length_filtered = total_distance / steps_filtered  # metriä
step_length_fourier = total_distance / steps_fourier  # metriä

# NUMEERISET TULOKSET YLHÄÄLLÄ
st.header("Numeeriset Tulokset")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Askelmäärä (Suodatettu)", steps_filtered)
            
    
with col2:
    st.metric("Askelmäärä (Fourier)", steps_fourier,)
    
with col3:
    st.metric("Keskinopeus", f"{average_speed:.2f} m/s")
    st.caption(f"= {average_speed_kmh:.2f} km/h")
    
with col4:
    st.metric("Kuljettu matka", f"{total_distance:.1f} m")
    st.caption(f"= {total_distance/1000:.3f} km")
    
with col5:
    st.metric("Askelpituus", f"{step_length_filtered:.2f} m")
    st.caption(f"= {step_length_filtered*100:.1f} cm")

st.divider()

st.header("GPS-datan epätarkkuus")

# Laske realistinen matka askelten perusteella
typical_step_length = 0.70  
estimated_distance = steps_filtered * typical_step_length

col_text, col_image = st.columns([3, 2])

with col_text:
    st.markdown(f"""
    ### Askelpituuden analyysi


    GPS-pohjainen askelpituus ({step_length_filtered:.2f} m) vaikuttaa epärealistisen pitkältä verrattuna tyypilliseen 
    askelpituuteen ({typical_step_length:.2f} m). Tämä viittaa siihen, että **GPS-data on todennäköisesti epätarkka** ja 
    on lisännyt matkaa, jota ei ole todellisuudessa kuljettu.

    Jos käytämme realistista askelpituutta n. {typical_step_length:.2f} m, kuljettu matka olisi:
    - **Arvioitu matka:** {estimated_distance:.1f} m (0.70 m askelpituudella)
    - **GPS-matka:** {total_distance:.1f} m (GPS-datan perusteella)
    - **Arvioitu matka:** 320m (Kolmannen osapuolen karttapalvelu) **(Kuva oikealla)**

    Tämä vahvistaa, että GPS-datassa on epätarkkuuksia, jotka johtavat yliarvioituun kuljettuun matkaan. 
    
    **Laskettu askelpituus kun etäisyys on 320m:**
    - Askelpituus = 320 m ÷ {steps_filtered} askelta = **{320/steps_filtered:.2f} m ({320/steps_filtered*100:.1f} cm)**
    
    {320/steps_filtered:.2f} m askelpituus on realistisempi arvo kuin 0.84 m kyseisessä tilanteessa kun otetaan huomioon kävelijän pituus (n. 1.7 m).
    """)

with col_image:
    st.image('https://raw.githubusercontent.com/SamppaLHT/Fysiikan-loppuprojekti/refs/heads/main/Images/distanceComparison.png', caption='Matkojen vertailu')

st.divider()

# Suodatettu data + piikinetsintä
st.header("Suodatettu Signaali")

fig1, ax1 = plt.subplots(figsize=(16, 6))

ax1.plot(time_data, z_filtered, label='Suodatettu', linewidth=0.8, color='blue')

ax1.axhline(y=height_threshold, color='red', linestyle='--', alpha=0.5, label='Kynnysarvo')
ax1.set_xlabel('Aika (s)', fontsize=12)
ax1.set_ylabel('Z-kiihtyvyys (m/s²)', fontsize=12)
ax1.set_title(f'Suodatettu Signaali (koko mittaus)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

st.pyplot(fig1)
    

# Fourier-analyysi
st.header("Fourier-analyysi")

N = len(z_acceleration)
yf = fft(z_acceleration)
xf = fftfreq(N, 1/sampling_rate)

positive_freq_mask = xf > 0
xf_positive = xf[positive_freq_mask]
yf_positive = np.abs(yf[positive_freq_mask])

power_spectrum = (2.0/N) * yf_positive**2

walking_freq_mask = (xf_positive >= 0.5) & (xf_positive <= 4.0)
xf_walking = xf_positive[walking_freq_mask]
power_walking = power_spectrum[walking_freq_mask]
yf_walking = yf_positive[walking_freq_mask]

dominant_freq_idx = np.argmax(yf_walking)
dominant_frequency = xf_walking[dominant_freq_idx]

steps_fourier = int(dominant_frequency * time_data[-1])

# GPS-datan analyysi
def calculate_distance(lat1, lon1, lat2, lon2):
    """Laske etäisyys kahden GPS-pisteen välillä (Haversine-kaava)"""
    R = 6371000 
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    distance = R * c
    return distance

# Laske kuljettu matka GPS-datasta
latitudes = loc_data['Latitude (°)'].values
longitudes = loc_data['Longitude (°)'].values
gps_time = loc_data['Time (s)'].values

total_distance = 0
for i in range(len(latitudes) - 1):
    dist = calculate_distance(latitudes[i], longitudes[i], 
                              latitudes[i+1], longitudes[i+1])
    total_distance += dist

# Laske keskinopeus
total_time = gps_time[-1] - gps_time[0]  # sekunteina
average_speed = total_distance / total_time  # m/s
average_speed_kmh = average_speed * 3.6  # km/h

# Laske askelpituus molemmilla menetelmillä
step_length_filtered = total_distance / steps_filtered  # metriä
step_length_fourier = total_distance / steps_fourier  # metriä

fig2, ax2 = plt.subplots(figsize=(16, 6))

ax2.plot(xf_walking, power_walking, linewidth=2, color='purple')
ax2.axvline(x=dominant_frequency, color='red', linestyle='--', 
            label=f'Askeltaajuus: {dominant_frequency:.2f} Hz')
ax2.set_xlabel('Taajuus (Hz)', fontsize=12)
ax2.set_ylabel('Teho (m²/s⁴)', fontsize=12)
ax2.set_title('Tehospektri', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.5, 4.0)

st.pyplot(fig2)

st.divider()

# Karttakuva kuljetusta reitistä
st.header("Kuljettu Reitti Kartalla")

latitudes = loc_data['Latitude (°)'].values
longitudes = loc_data['Longitude (°)'].values

center_lat = np.mean(latitudes)
center_lon = np.mean(longitudes)

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=17,
    tiles='OpenStreetMap',
    zoom_control=False,
    scrollWheelZoom=False,
    dragging=False,
    doubleClickZoom=False
)

route_coordinates = [[lat, lon] for lat, lon in zip(latitudes, longitudes)]
folium.PolyLine(
    route_coordinates,
    color='blue',
    weight=4,
    opacity=0.8,
    tooltip='Kuljettu reitti'
).add_to(m)

folium.Marker(
    [latitudes[0], longitudes[0]],
    popup='Aloituspiste',
    icon=folium.Icon(color='green', icon='play')
).add_to(m)

folium.Marker(
    [latitudes[-1], longitudes[-1]],
    popup='Lopetuspiste',
    icon=folium.Icon(color='red', icon='stop')
).add_to(m)

st_folium(m, width=1200, height=600)

st.divider()

# Lisätiedot
with st.expander("Menetelmien ja laskujen selitykset"):
    st.markdown("""
    ### Suodatettu signaali + piikinetsintä
    - **Z-komponentti:** Käytetään eteen-taakse suuntaista kiihtyvyyttä 
    - **Alipäästösuodatin:** Butterworth-suodatin (2.0 Hz) poistaa korkeataajuisen kohinan
    - **Piikinetsintä:** Etsii paikalliset maksimiarvot, jotka ylittävät 7.0 m/s² kynnysarvon
    - **Etuna:** Tarkka yksittäisten askelten tunnistus
    - **Haittana:** Vaatii parametrien säätöä (kynnysarvo, etäisyys)
    
    ### Fourier-analyysi (FFT)
    - **FFT:** Muuntaa aikasarjan taajuusalueelle
    - **Tehospektri:** Näyttää voimakkuuden eri taajuuksilla (m²/s⁴)
    - **Dominoiva taajuus:** Kävelyalueella (0.5-4.0 Hz) vastaa askeltaajuutta
    - **Etuna:** Ei vaadi parametrien säätöä, toimii kohinaisella datalla
    - **Haittana:** Olettaa tasaisen askeltaajuuden
    
    ### GPS-datan epätarkkuus
    - **GPS-matka:** {:.1f} m (laskettu Haversine-kaavalla GPS-pisteiden välillä)
    - **Arvioitu todellinen matka:** 320 m (kolmannen osapuolen karttapalvelusta)
    - **GPS-virhe:** GPS yliarvioi matkaa noin {:.1f}%
    - **Askelpituus GPS:llä:** {:.2f} m (epärealistinen)
    - **Askelpituus 320m:llä:** 0.72 m (realistinen 1.7m kävelijälle)
    
    ### Käytetyt kaavat
    - GPS-matka: Haversine-kaava (lat/lon → metrit)
    - Nopeus: v = s / t
    - Askelpituus: L = s / n (missä s = matka, n = askelmäärä)
    - Tehospektri: P = (2/N) × |FFT|²
    """.format(total_distance, (total_distance - 320) / 320 * 100, step_length_filtered))
