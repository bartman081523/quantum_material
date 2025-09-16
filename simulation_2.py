################################################################################
#
#   simulation_v2.py
#
#   Reflecting the new synthesis, this script abandons the comparison and
#   focuses exclusively on optimizing the promising BTO platform.
#   It performs a parameter sweep over the active layer thickness to understand
#   the trade-offs and find an optimal design point for our quantum application.
#
################################################################################

from EMpy.modesolvers.FD import VFDModeSolver
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Physikalische & chemische Parameter (SI-Einheiten) ---
materials = {
    'SiO2': 1.44, 'Si3N4': 2.00, 'BTO': 2.36, 'Air': 1.00
}
pockels_coeffs = {
    'BTO': 923e-12,  # m/V
}
ACTIVE_MATERIAL = 'BTO'

# --- 2. Geometrie- und Simulationsparameter (SI-Einheiten) ---
WAVELENGTH = 1.55e-6
GRID_RESOLUTION_PER_MICRON = 40
SIM_WIDTH = 4.0e-6
SIM_HEIGHT = 3.0e-6
WG_WIDTH = 1.2e-6
SI3N4_THICKNESS = 0.3e-6
SIO2_THICKNESS = 2.0e-6
ELECTRODE_GAP = 1.5e-6

# --- 3. Kernlogik: Simulationsfunktion mit variabler Dicke ---
def simulate_bto_waveguide(active_thickness, n_modes_to_find=4):
    """
    Simuliert die Si3N4+BTO-Struktur für eine gegebene Dicke der aktiven Schicht.
    Gibt die berechneten Leistungsmetriken zurück.
    """
    print(f"\n--- Simulating BTO with thickness = {active_thickness*1e9:.0f} nm ---")

    def n_profile_func(x_1d, y_1d):
        xx, yy = np.meshgrid(x_1d, y_1d, indexing='ij')
        n = np.ones_like(xx, dtype=float) * materials['Air']
        si3n4_start_y = SIO2_THICKNESS
        active_start_y = si3n4_start_y + SI3N4_THICKNESS
        active_end_y = active_start_y + active_thickness # <-- Parameter wird hier verwendet
        sio2_mask = yy < si3n4_start_y
        wg_mask = (yy >= si3n4_start_y) & (yy < active_start_y) & (np.abs(xx) < WG_WIDTH / 2)
        active_mask = (yy >= active_start_y) & (yy < active_end_y) & (np.abs(xx) < WG_WIDTH / 2)
        n[sio2_mask] = materials['SiO2']
        n[wg_mask] = materials['Si3N4']
        n[active_mask] = materials[ACTIVE_MATERIAL]
        return n

    x = np.linspace(-SIM_WIDTH / 2, SIM_WIDTH / 2, int(SIM_WIDTH * 1e6 * GRID_RESOLUTION_PER_MICRON))
    y = np.linspace(0, SIM_HEIGHT, int(SIM_HEIGHT * 1e6 * GRID_RESOLUTION_PER_MICRON))
    solver = VFDModeSolver(WAVELENGTH, x, y, lambda x, y: n_profile_func(x, y)**2, '0000')
    solver.solve(neigs=n_modes_to_find)

    if not solver.modes:
        return None, None, None

    fundamental_mode = solver.modes[0]
    n_eff = fundamental_mode.neff.real
    intensity = np.abs(fundamental_mode.intensity())
    total_power = np.sum(intensity)

    Nx_field, Ny_field = intensity.shape
    x_field, y_field = np.meshgrid(
        np.linspace(-SIM_WIDTH / 2, SIM_WIDTH / 2, Nx_field),
        np.linspace(0, SIM_HEIGHT, Ny_field),
        indexing='ij'
    )

    active_start_y = SIO2_THICKNESS + SI3N4_THICKNESS
    active_end_y = active_start_y + active_thickness
    active_mask = (y_field >= active_start_y) & (y_field < active_end_y) & (np.abs(x_field) < WG_WIDTH / 2)

    gamma = np.real(np.sum(intensity[active_mask]) / total_power)
    n_active = materials[ACTIVE_MATERIAL]
    r_eff = pockels_coeffs[ACTIVE_MATERIAL]
    v_pi_l_meters = (WAVELENGTH * ELECTRODE_GAP) / (n_active**3 * r_eff * gamma)
    v_pi_l_cm = v_pi_l_meters * 100

    return n_eff, gamma, v_pi_l_cm

# --- 4. Haupt-Ausführungsblock: Parameter-Sweep ---
if __name__ == "__main__":
    # Definiere den Bereich der zu testenden BTO-Dicken (in Metern)
    thickness_sweep_nm = np.linspace(50, 200, 7)
    thickness_sweep_m = thickness_sweep_nm * 1e-9

    results = []
    for thickness in thickness_sweep_m:
        n_eff, gamma, v_pi_l = simulate_bto_waveguide(thickness)
        if n_eff is not None:
            results.append((thickness * 1e9, n_eff, gamma * 100, v_pi_l))

    # --- 5. Ergebnisse ausgeben und visualisieren ---
    print("\n\n--- PARAMETER SWEEP RESULTS ---")
    print("=" * 45)
    print(f"{'BTO Thickness (nm)':<20} {'Gamma (%)':<12} {'VπL (V·cm)':<12}")
    print("-" * 45)
    for res in results:
        print(f"{res[0]:<20.0f} {res[2]:<12.2f} {res[3]:<12.4f}")
    print("=" * 45)

    # Visualisierung der Ergebnisse
    thicknesses_nm = [r[0] for r in results]
    gammas = [r[2] for r in results]
    v_pi_ls = [r[3] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color1 = 'tab:red'
    ax1.set_xlabel('BTO Layer Thickness (nm)')
    ax1.set_ylabel('Projected VπL (V·cm)', color=color1)
    ax1.plot(thicknesses_nm, v_pi_ls, 'o-', color=color1, label='VπL')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx() # Zweite Y-Achse
    color2 = 'tab:blue'
    ax2.set_ylabel('Confinement Factor (%)', color=color2)
    ax2.plot(thicknesses_nm, gammas, 's--', color=color2, label='Confinement (Gamma)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    plt.title('BTO Hybrid Platform: VπL and Confinement vs. Layer Thickness')
    plt.grid(True)
    fig.savefig('simulation_v2_optimization_sweep.pdf')
    print("\nOptimization plot saved as 'simulation_v2_optimization_sweep.pdf'")
    plt.show()
