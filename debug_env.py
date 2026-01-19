import sys
import site
import os

print(f"Executable: {sys.executable}")
print(f"User Site: {site.getusersitepackages()}")
print("Sys Path:")
for p in sys.path:
    print(p)

try:
    import pandas
    print(f"Pandas found at: {pandas.__file__}")
except ImportError as e:
    print(f"Pandas not found: {e}")
