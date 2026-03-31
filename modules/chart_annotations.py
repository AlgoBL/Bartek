"""
chart_annotations.py — Interaktywne Adnotacje Naukowe (V.10)
===========================================================
Pozwala nanosić na wykresy Plotly istotne zdarzenia historyczne i rynkowe,
wspomagając w ten sposób \"Narrative Visualization\" (Segel & Heer).
Zmniejsza to problem gubienia kontekstu historycznego przy analizie 
time-series w Day Trading / Macro Analysis.
"""

import pandas as pd
import plotly.graph_objects as go

# Lista kluczowych eventów rynkowych do kontekstualizacji
MARKET_EVENTS = {
    # Data: (Label, Kolor linii/tekstu, Description)
    '2020-02-19': ('COVID Crash Start', '#ff1744', 'S&P500 peaks pre-crash (-34% w 1 msc)'),
    '2020-03-23': ('COVID Dno', '#00e676', 'Fed zapowiada nielimitowane QE'),
    '2022-01-03': ('Fed Pivot', '#f39c12', 'Start cyklu podwyżek stóp: 0 → 5.5%'),
    '2022-10-12': ('Bear Market Low', '#00ccff', 'Dno cyklu zacieśniania'),
    '2023-03-10': ('SVB Collapse', '#a855f7', 'Regional Bank Crisis, zasilenie płynności (BTFP)'),
    '2024-04-19': ('Bitcoin Halving', '#ffea00', 'Zmniejszenie podaży BTC'),
    '2024-08-05': ('Yen Carry Unwind', '#ff1744', 'Panika rynkowa i skok VIX'),
}

def add_market_annotations(fig: go.Figure, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> go.Figure:
    """
    Dodaje linie pionowe z etykietami dla kluczowych momentów na osi X, 
    które mieszczą się w zakresie [start_date, end_date].
    
    Parameters
    ----------
    fig : go.Figure
        Wykres Plotly (np. z symulacji Monte Carlo lub backtestu)
    start_date, end_date : str lub datetime
        Zakres widoczny na osi X, pozwalający pominąć eventy spoza widoku.
    
    Returns
    -------
    go.Figure
        Nienaruszony, ale uzupełniony o referencje obiekt Figure
    """
    try:
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
    except Exception:
        # W przypadku błędu parsowania (np. gdy podano dziwny format osi kategorycznej)
        # rezygnujemy z dodawania adnotacji, by nie wyłożyć całej aplikacji.
        return fig
    
    for date_str, (label, color, desc) in MARKET_EVENTS.items():
        event_ts = pd.to_datetime(date_str)
        
        # Dodaj tylko te wydarzenia, które mieszczą się w analizowanym przedziale
        if start_ts <= event_ts <= end_ts:
            
            # 1) Pionowa linia referencyjna (kropkowana, transparentna)
            fig.add_vline(
                x=event_ts,
                line_dash='dot',
                line_color=color,
                line_width=1.5,
                opacity=0.6,
                layer='below'
            )
            
            # 2) Etykieta (tekst na górze)
            # Jeśli mamy multiple traces z dużą wariancją wartości na osi Y,
            # najbezpieczniej użyć yref='paper' (oznacza % wysokości obrazu)
            fig.add_annotation(
                x=event_ts,
                y=1.0,  # Zawsze na samej górze wykresu
                yref='paper',
                text=label,
                textangle=-90,
                align='left',
                valign='top',
                showarrow=False,
                font=dict(color=color, size=10, family='Inter'),
                opacity=0.9,
                hovertext=desc,       # Tooltip ze szczegółami wydarzenia
                hoverlabel=dict(bgcolor='#0f111a', bordercolor=color)
            )

    return fig
