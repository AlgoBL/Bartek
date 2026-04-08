import numpy as np
import plotly.graph_objects as go
from scipy import signal

def compute_fourier_spectrum(prices, fs=252):
    """
    Zwraca periodygram (Welch) by wykryć dominujące cykle koniunkturalne w czasie.
    """
    # Log-zwroty
    returns = np.diff(np.log(prices))
    returns = returns - np.mean(returns) # detrend
    
    # Stosujemy Welch's method dla gładszego spektrum
    f, Pxx = signal.welch(returns, fs, nperseg=min(len(returns), int(252*4)))
    
    # Przelicz częstotliwość (cykli na rok) na okres (lata)
    # f=0 wyeliminuje się automatycznie, ale dla pewności ignorujemy
    valid = f > 0
    periods = np.zeros_like(f)
    periods[valid] = 1.0 / f[valid]
    
    return periods[valid], Pxx[valid]

def plot_fourier_spectrum(periods, Pxx):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=Pxx, fill='tozeroy', name='Power Spectrum', line=dict(color='#00e676')))
    
    # Oznaczenia teoretyczne cykli
    # Kitchin: ~3-5 years
    # Juglar: ~7-11 years
    fig.add_vrect(x0=3, x1=5, fillcolor="rgba(255, 234, 0, 0.15)", layer="below", line_width=0, annotation_text="Cykl Kitchina (Zapasów) 3-5 lat", annotation_position="top left", annotation_font_size=10)
    fig.add_vrect(x0=7, x1=11, fillcolor="rgba(255, 23, 68, 0.15)", layer="below", line_width=0, annotation_text="Cykl Juglara (Inwestycji) 7-11 lat", annotation_position="top right", annotation_font_size=10)
    
    fig.update_layout(
        template="plotly_dark",
        title="Dekompozycja Cykli Rynkowych (Fourier Power Spectrum)",
        xaxis_title="Okres Cyklu (Lata)",
        yaxis_title="Moc Sygnału (Power Density)",
        xaxis=dict(range=[0, 15]),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def _custom_cwt(data, widths, w=5.0):
    """
    Własna implementacja CWT z falką Morleta (zastępuje usunięte scipy.signal.cwt).
    """
    output = np.zeros((len(widths), len(data)), dtype=complex)
    for ind, width in enumerate(widths):
        N = int(np.min([10 * width, len(data)]))
        x = np.arange(0, N) - (N - 1.0) / 2
        x = x / width
        wavelet_data = np.exp(1j * w * x) * np.exp(-0.5 * x**2) * np.pi**(-0.25) * np.sqrt(1/width)
        wavelet_data = np.conj(wavelet_data[::-1])
        output[ind, :] = np.convolve(data, wavelet_data, mode='same')
    return output

def compute_wavelet_transform(prices):
    """
    Oblicza falową dekompozycję (Continuous Wavelet Transform) przy użyciu falki Morleta.
    Ujawnia JAK cykle zmieniają się w czasie (czasowo-częstotliwościowo).
    """
    returns = np.diff(np.log(prices))
    returns = returns - np.mean(returns)
    returns = returns / (np.std(returns) + 1e-8)
    
    # Szerokości od 1 miesiąca do ~15 lat
    widths = np.geomspace(21, 252*15, num=60)
    
    # Używamy Morlet2. Okres = szerokość dla standardowej falki
    cwtmatr = _custom_cwt(returns, widths)
    power = np.abs(cwtmatr)**2
    
    return widths, power

def plot_wavelet_transform(widths, power, dates=None):
    # Okres w latach ~ widths / 2520. Dokładny współczynnik zależy od Morlet w, 
    # dla w=5 okres fourierowski to ~ widths / 252 lat. Daje akceptowalne przybliżenie analityczne.
    periods_years = widths / 252.0
    
    if dates is None or len(dates) != power.shape[1]:
        dates = np.arange(power.shape[1])
    else:
        # returns array length is len(prices) - 1
        dates = dates[1:]
        
    fig = go.Figure(data=go.Contour(
        z=power,
        x=dates,
        y=periods_years,
        colorscale='Magma',
        contours=dict(showlines=False)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title="Spektrogram Falowy Skorygowany Czasowo (Continuous Wavelet Transform)",
        yaxis_title="Długość Cyklu (Lata)",
        xaxis_title="Czas (Historia)",
        yaxis_type="log", # Skala logarytmiczna dla czytelności obu cykli
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
