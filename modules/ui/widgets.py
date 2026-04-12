import streamlit as st
from modules.isin_resolver import ISINResolver

def _on_ticker_change(key: str):
    """Callback process that hooks into session state to replace ISINs with tickers."""
    if key in st.session_state:
        val = st.session_state[key]
        if val and isinstance(val, str):
            new_val = ISINResolver.replace_isins_in_text(val)
            st.session_state[key] = new_val

def _wrap_on_change(kwargs: dict, key: str):
    """Pomocnicza funkcja by połączyć istniejący callback z naszym callbackiem."""
    if "on_change" in kwargs:
        orig_cb = kwargs["on_change"]
        orig_args = kwargs.get("args", tuple())
        orig_kwargs = kwargs.get("kwargs", {})
        def new_cb():
            _on_ticker_change(key)
            orig_cb(*orig_args, **orig_kwargs)
        kwargs["on_change"] = new_cb
        kwargs.pop("args", None)
        kwargs.pop("kwargs", None)
    else:
        kwargs["on_change"] = _on_ticker_change
        kwargs["args"] = (key,)

def ticker_input(label: str, value: str = "", key: str = None, parent=st, **kwargs):
    """
    Wraper wokół st.text_input, który automatycznie wyłuskuje i podmienia 
    numery ISIN na ich Ticker w momencie zatwierdzenia przez użytkownika.
    Można podać np. parent=st.sidebar
    """
    if key is None:
        key = "ticker_input_" + label.lower().replace(" ", "_")
        
    _wrap_on_change(kwargs, key)
    return parent.text_input(label, value=value, key=key, **kwargs)

def tickers_area(label: str, value: str = "", key: str = None, parent=st, **kwargs):
    """
    Wraper wokół st.text_area...
    Można podać np. parent=st.sidebar
    """
    if key is None:
        key = "tickers_area_" + label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        
    _wrap_on_change(kwargs, key)
    return parent.text_area(label, value=value, key=key, **kwargs)
