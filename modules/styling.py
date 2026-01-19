def apply_styling():
    return """
    <style>
        /* Modern Dark Theme Pallette */
        :root {
            --background-color: #0e0e0e;
            --surface-color: #1a1a1a;
            --primary-color: #00ff88; /* Cyberpunk Green */
            --secondary-color: #00ccff; /* Cyberpunk Blue */
            --text-color: #e0e0e0;
            --accent-color: #ff0055;
        }

        /* General App Styling */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
            font-family: 'Inter', sans-serif;
        }

        /* Headers */
        h1, h2, h3 {
            color: var(--text-color) !important;
            font-weight: 600;
        }
        h1 {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            padding-bottom: 1rem;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: var(--surface-color);
            border-right: 1px solid #333;
        }
        [data-testid="stSidebar"] .css-1d391kg {
            padding-top: 2rem;
        }

        /* Card-like containers using st.metric or custom divs */
        div[data-testid="stMetricValue"] {
            color: var(--primary-color);
            font-weight: 700;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            color: #000;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 255, 136, 0.4);
        }

        /* Input Fields */
        .stSlider {
            color: var(--primary-color);
        }
    </style>
    """
