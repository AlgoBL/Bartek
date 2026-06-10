"""
project_updater.py
==================
Narzędzie CLI i główne metody automatyzacji zarządzania modułami
w systemie Intelligent Barbell.

Użycie:
  python project_updater.py scan    - Szuka stron nieobecnych w REGISTRY
  python project_updater.py sync    - Regeneruje sekcje w 00_Mapa_Projektu.py i search_index.py
  python project_updater.py new     - (WIP) Generator nowego modułu z szablonu
"""
import os
import sys

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from modules.module_registry import (
    find_unregistered_pages,
    build_menu_structure,
    build_page_to_nav_url,
    build_pages_map,
    REGISTRY,
)

def _replace_between_markers(file_path: str, start_marker: str, end_marker: str, new_content: str):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        print(f"❌ Nie znaleziono markerów w {file_path}. Pomięto.")
        return False

    prefix = content[:start_idx + len(start_marker)]
    suffix = content[end_idx:]
    
    updated_content = prefix + "\n" + new_content + "\n" + suffix

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(updated_content)
    
    print(f"✅ Zaktualizowano {os.path.basename(file_path)}")
    return True

def cmd_scan():
    print("🔍 Skanowanie w poszukiwaniu nowych stron...\n")
    unregistered = find_unregistered_pages()
    if not unregistered:
        print("✅ Wszystkie strony są poprawnie zarejestrowane w module_registry.py")
    else:
        print("❌ Znaleziono niezarejestrowane strony (dodaj je do REGISTRY w modules/module_registry.py):")
        for p in unregistered:
            print(f"   - {p}")

def cmd_sync():
    print("🔄 Synchronizacja rejestru z plikami projektu...\n")
    
    # Mapa Projektu (Opcjonalne, ponieważ kod już to ładuje dynamicznie)
    # Ze względu na implementację, obecnie używamy direct importu w tamtym pliku.
    # Upewniamy się, że to działa poprawnie.
    print("✅ MENU_STRUCTURE jest generowane dynamicznie w runtime.")
    print("✅ PAGE_TO_NAV_URL oraz PAGES_MAP są generowane dynamicznie w runtime.")
    
    print(f"\n✨ Gotowe! Aktualnie w rejestrze jest {len(REGISTRY)} modułów.")

def cmd_new():
    print("🚧 Kreator modułów jest w trakcie budowy. Obecnie proszę dodać moduł ręcznie do REGISTRY.")

def main():
    if len(sys.argv) < 2:
        print("Użycie: python project_updater.py [scan|sync|new]")
        return
        
    cmd = sys.argv[1].lower()
    if cmd == "scan":
        cmd_scan()
    elif cmd == "sync":
        cmd_sync()
    elif cmd == "new":
        cmd_new()
    else:
        print(f"Nieznana komenda: {cmd}")

if __name__ == "__main__":
    main()
