################################################################################
#
#   deep_investigate.py
#
#   A comprehensive diagnostic script to recursively inspect the installed 'EMpy'
#   package. It maps out all submodules, classes, and function signatures to
#   provide a definitive, factual map of the library's structure and API.
#   This script stops all guesswork.
#
################################################################################

import importlib
import inspect
import pkgutil
import sys

# --- Hauptuntersuchungsfunktion ---
def investigate_package(package_name):
    """
    Importiert ein Paket und untersucht rekursiv alle seine Inhalte.
    """
    print(f"--- Starte umfassende Untersuchung von: {package_name} ---")

    try:
        # Lade das Hauptpaket
        package = importlib.import_module(package_name)
        print(f"ERFOLG: Hauptpaket '{package_name}' erfolgreich importiert.")
        print(f"Installationspfad: {package.__file__}\n")

        # Starte die rekursive Untersuchung
        explore_module(package, "")

    except ImportError:
        print(f"KRITISCHER FEHLER: Das Paket '{package_name}' konnte nicht importiert werden.")
        print("Stellen Sie sicher, dass die Installation erfolgreich war und der Name korrekt ist.")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

    print("\n--- Untersuchung abgeschlossen ---")


# --- Rekursive Funktion zur Untersuchung von Modulen ---
def explore_module(module, indent_prefix):
    """
    Listet den Inhalt eines Moduls auf und steigt in alle Submodule hinab.
    """
    # Verhindere unendliche Rekursionen und zu tiefe Abstiege
    if len(indent_prefix) > 20:
        print(f"{indent_prefix}... [Rekursionstiefe erreicht]")
        return

    # Untersuche alle "Mitglieder" (Attribute) des aktuellen Moduls
    members = inspect.getmembers(module)
    if not members:
        return

    # Sortiere, um eine konsistente Ausgabe zu erhalten
    members.sort(key=lambda x: x[0])

    for name, obj in members:
        # Ignoriere interne Python-Attribute
        if name.startswith('__'):
            continue

        # --- Fall 1: Das Mitglied ist eine Funktion ---
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            try:
                signature = inspect.signature(obj)
                print(f"{indent_prefix}FUNCTION: {name}{signature}")
            except (ValueError, TypeError):
                print(f"{indent_prefix}FUNCTION: {name}(...) [Signatur konnte nicht gelesen werden]")

        # --- Fall 2: Das Mitglied ist eine Klasse ---
        elif inspect.isclass(obj):
            print(f"{indent_prefix}CLASS: {name}")
            # Untersuche die Methoden innerhalb der Klasse
            explore_module(obj, indent_prefix + "    ")

        # --- Fall 3: Das Mitglied ist ein Submodul ---
        elif inspect.ismodule(obj):
            # Stelle sicher, dass es sich um ein echtes Submodul des Pakets handelt
            if obj.__name__.startswith(module.__name__):
                print(f"{indent_prefix}SUBMODULE: {name}")
                explore_module(obj, indent_prefix + "    ")


# --- Haupt-Ausführungsblock ---
if __name__ == "__main__":
    # Der Name des zu untersuchenden Pakets, basierend auf der erfolgreichen Installation
    # und den Verzeichnisnamen, die Sie gefunden haben.
    PACKAGE_TO_INVESTIGATE = "EMpy"
    investigate_package(PACKAGE_TO_INVESTIGATE)
