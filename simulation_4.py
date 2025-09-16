################################################################################
#
#   simulation_v4.py (Final Robust Method)
#
#   This version abandons the flawed attempt to simulate a Y-Z plane and returns
#   to the validated X-Y cross-section method. It tests the acousto-optic
#   hypothesis (H4) by simulating two distinct waveguide cross-sections:
#   one for the "trough" (n) and one for the "crest" (n + Δn) of the SAW wave.
#
#   The key metric is the difference in their effective indices (Δn_eff),
#   which directly quantifies the modulation strength and allows calculation
#   of the coupling length. This method is robust and uses only validated parts
#   of the EMpy library.
#
################################################################################

import numpy as np
from EMpy.modesolvers.FD import VFDModeSolver
import matplotlib.pyplot as plt

# --- 1. Physikalische Parameter ---
# Materialien
N_SIC = 2.65
N_SIO2 = 1.44
N_AIR = 1.00

# SAW-induzierte Brechungsindex-Modulation (aus H4)
DELTA_N_INDUCED_BY_SAW = 1.5e-4

# --- 2. Simulations-Setup (X-Y Querschnitt) ---
WAVELENGTH = 1.55e-6
GRID_RESOLUTION_PER_MICRON = 40

# Geometrie
SIM_WIDTH = 4.0e-6
SIM_HEIGHT = 3.0e-6
WG_WIDTH = 1.5e-6  # Etwas breiter für besseres Confinement in SiC
SIC_THICKNESS = 0.5e-6
SIO2_THICKNESS = 1.0e-6

# --- 3. Simulationsfunktion für einen gegebenen Brechungsindex ---
def simulate_sic_waveguide(core_refractive_index, material_name):
    """
    Simuliert einen SiC-Wellenleiter-Querschnitt für einen gegebenen Kern-Brechungsindex.
    """
    print(f"\n--- Simulating SiC Waveguide Cross-Section for: {material_name} (n={core_refractive_index:.5f}) ---")

    def eps_func(x, y):
        n = np.full((len(x), len(y)), N_AIR, dtype=float)
        # Gitter für Masken erstellen
        xx, yy = np.meshgrid(x, y, indexing='ij')
        # Substrat
        n[yy < SIO2_THICKNESS] = N_SIO2
        # Wellenleiter-Kern
        wg_mask = (yy >= SIO2_THICKNESS) & \
                  (yy < SIO2_THICKNESS + SIC_THICKNESS) & \
                  (np.abs(xx) < WG_WIDTH / 2)
        n[wg_mask] = core_refractive_index
        return n**2

    x = np.linspace(-SIM_WIDTH / 2, SIM_WIDTH / 2, int(SIM_WIDTH * 1e6 * GRID_RESOLUTION_PER_MICRON))
    y = np.linspace(0, SIM_HEIGHT, int(SIM_HEIGHT * 1e6 * GRID_RESOLUTION_PER_MICRON))

    # DEFINITIVE KORREKTUR: Verwende die korrekte 4-Zeichen-Boundary für den X-Y Solver
    boundary = '0000' # Reflektierende Ränder

    solver = VFDModeSolver(WAVELENGTH, x, y, eps_func, boundary)
    solver.solve(neigs=2)

    if not solver.modes:
        print("ERROR: No modes found!")
        return None

    # Speichere den Plot der fundamentalen Mode
    intensity = np.abs(solver.modes[0].intensity())
    plot_extent = [x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6]
    plt.figure(figsize=(6,5))
    plt.imshow(intensity.T, extent=plot_extent, origin='lower', aspect='auto', cmap='inferno')
    plt.title(f'Fundamental Mode Intensity ({material_name})')
    plt.xlabel('x (µm)')
    plt.ylabel('y (µm)')
    plt.colorbar(label='Intensity (a.u.)')
    plt.tight_layout()
    plt.savefig(f"simulation_v4_mode_{material_name}.pdf")
    plt.close()
    print(f"  - Mode plot saved as 'simulation_v4_mode_{material_name}.pdf'")

    return solver.modes[0].neff.real

# --- 4. Ausführung der Simulationen und Analyse ---

# 4a. Simuliere das "Tal" der SAW-Welle (unbeeinflusstes SiC)
neff_unstrained = simulate_sic_waveguide(N_SIC, "SiC_unstrained")

# 4b. Simuliere den "Berg" der SAW-Welle (maximal beeinflusstes SiC)
neff_strained = simulate_sic_waveguide(N_SIC + DELTA_N_INDUCED_BY_SAW, "SiC_strained")

# --- 5. Ergebnisse ausgeben ---
if neff_unstrained is not None and neff_strained is not None:
    delta_neff = abs(neff_strained - neff_unstrained)
    # Kopplungslänge Lc = pi / (2 * kappa) = lambda / (2 * delta_neff)
    coupling_length = WAVELENGTH / (2 * delta_neff)

    print("\n\n--- Acousto-Optic (SAW) Coupling Analysis ---")
    print("="*70)
    print(f"  SAW-induced Δn: {DELTA_N_INDUCED_BY_SAW:.1e}")
    print(f"  n_eff (unstrained SiC): {neff_unstrained:.5f}")
    print(f"  n_eff (strained SiC):   {neff_strained:.5f}")
    print("-"*70)
    print(f"  Effective Index Modulation (Δn_eff): {delta_neff:.2e}")
    print(f"  Projected Coupling Length (Lc): {coupling_length*1e6:.1f} µm  ({coupling_length*1e3:.1f} mm)")
    print("="*70)

    print("\n--- INTERPRETATION ---")
    print("  Δn_eff is the direct measure of how strongly the SAW wave modulates the")
    print("  phase of the guided light.")
    print("  Lc is the distance over which the modulator would achieve a 100% effect.")
    print("  A short coupling length (in the mm range or less) for a realistic Δn")
    print("  is a very strong indicator that the acousto-optic coupling is highly")
    print("  efficient and a viable alternative to electro-optics, supporting Hypothesis H4.")
    print("--------------------------------------------------------------------------")
