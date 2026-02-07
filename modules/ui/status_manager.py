import streamlit as st
import time

class StatusManager:
    """
    Manages the visual state of operations (Math vs AI) using Streamlit's status container.
    """
    def __init__(self, label="Inicjalizacja...", expanded=True):
        self.container = st.status(label, expanded=expanded)
        self.last_update = time.time()

    def update(self, label, state="running", expanded=None):
        """Updates the main status label and state."""
        kwargs = {"label": label, "state": state}
        if expanded is not None:
            kwargs["expanded"] = expanded
        self.container.update(**kwargs)

    def write(self, message):
        """Writes a raw message to the container."""
        self.container.write(message)

    def info_math(self, message):
        """Displays a message related to mathematical calculations."""
        self.container.markdown(f"🧮 **Obliczenia:** {message}")

    def info_ai(self, message):
        """Displays a message related to AI operations."""
        self.container.markdown(f"🤖 **Gemini AI:** {message}")

    def info_data(self, message):
        """Displays a message related to data fetching."""
        self.container.markdown(f"📡 **Dane:** {message}")
        
    def success(self, message):
        """Displays a success message and closes the container."""
        self.container.update(label=message, state="complete", expanded=False)
    
    def error(self, message):
        """Displays an error message."""
        self.container.update(label="Błąd", state="error", expanded=True)
        self.container.error(message)

    def get_progress_callback(self):
        """Returns a callback function compatible with simulation modules."""
        def callback(pct, msg):
            # Throttle updates slightly to avoid UI flickering if needed, 
            # but Streamlit handles this reasonably well.
            # We assume msg contains context.
            if "AI" in msg or "Gemini" in msg or "Reżim" in msg or "Agent" in msg:
                self.info_ai(f"{msg} ({pct:.0%})")
            else:
                self.info_math(f"{msg} ({pct:.0%})")
        return callback
