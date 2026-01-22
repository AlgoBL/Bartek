import sys
import os

# Add the current directory to sys.path
sys.path.append(os.getcwd())

try:
    import modules.ai.trader
    print("Successfully imported modules.ai.trader")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
