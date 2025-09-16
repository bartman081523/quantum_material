################################################################################
#
#   simulation_final.py
#
#   Final version, now including the crucial TFLN hybrid baseline simulation
#   for a direct and fair comparison against the proposed alternatives.
#
################################################################################

from EMpy.modesolvers.FD import VFDModeSolver
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Physikalische & chemische Parameter (in SI-Einheiten) ---
# Brechungsindizes
materials = {
    'SiO2': 1.44,
    'Si3N4': 2.00,
    'BTO': 2.36,
    'AlN': 2.10,
    'TFLN': 2.21,  # Thin-Film Lithium Niobate (extraordinary index)
    'Air': 1.00
}
# Pockels-Koeffizienten (Literaturwerte)
pockels_coeffs = {
    'BTO': 923e-12,  # m/V
    'AlN': 1.5e-12,   # m/V
    'TFLN': 30.8e-12 # m/V (r33)
}

# Geometrie- und Simulationsparameter (in Metern)
WAVELENGTH = 1.55e-6
GRID_RESOLUTION_PER_MICRON = 40
SIM_WIDTH = 4.0e-6
SIM_HEIGHT = 3.0e-6
WG_WIDTH = 1.2e-6
SI3N4_THICKNESS = 0.3e-6
SIO2_THICKNESS = 2.0e-6
ACTIVE_LAYER_THICKNESS = 0.1e-6
ELECTRODE_GAP = 1.5e-6 # Angenommener Elektrodenabstand für VpiL-Berechnung

# --- 2. Kernlogik: Wiederverwendbare Simulationsfunktion ---
def simulate_waveguide(active_material_name, n_modes_to_find=4):
    """
    Definiert die Wellenleiterstruktur, führt den EMpy Modensolver aus,
    berechnet alle Metriken und speichert die Plots.
    """
    print(f"\n--- Starting simulation for Si3N4 + {active_material_name} ---")

    def n_profile_func(x_1d, y_1d):
        xx, yy = np.meshgrid(x_1d, y_1d, indexing='ij')
        n = np.ones_like(xx, dtype=float) * materials['Air']
        si3n4_start_y = SIO2_THICKNESS
        active_start_y = si3n4_start_y + SI3N4_THICKNESS
        active_end_y = active_start_y + ACTIVE_LAYER_THICKNESS
        sio2_mask = yy < si3n4_start_y
        wg_mask = (yy >= si3n4_start_y) & (yy < active_start_y) & (np.abs(xx) < WG_WIDTH / 2)
        active_mask = (yy >= active_start_y) & (yy < active_end_y) & (np.abs(xx) < WG_WIDTH / 2)
        n[sio2_mask] = materials['SiO2']
        n[wg_mask] = materials['Si3N4']
        n[active_mask] = materials[active_material_name]
        return n

    def eps_func(x, y):
        return n_profile_func(x, y)**2

    # Definiere Simulationsgitter
    x = np.linspace(-SIM_WIDTH / 2, SIM_WIDTH / 2, int(SIM_WIDTH * 1e6 * GRID_RESOLUTION_PER_MICRON))
    y = np.linspace(0, SIM_HEIGHT, int(SIM_HEIGHT * 1e6 * GRID_RESOLUTION_PER_MICRON))
    boundary = '0000'

    solver = VFDModeSolver(WAVELENGTH, x, y, eps_func, boundary)
    print("Solving eigenmodes with EMpy...")
    solver.solve(neigs=n_modes_to_find)
    print(f"Simulation finished. Found {len(solver.modes)} mode(s).")

    if not solver.modes:
        print("WARNING: No guided modes found!")
        return

    # --- 3. Analyse und Auswertung ---
    fundamental_mode = solver.modes[0]
    n_eff = fundamental_mode.neff.real
    intensity = np.abs(fundamental_mode.intensity())
    total_power = np.sum(intensity)

    Nx_field, Ny_field = intensity.shape
    x_field = np.linspace(-SIM_WIDTH / 2, SIM_WIDTH / 2, Nx_field)
    y_field = np.linspace(0, SIM_HEIGHT, Ny_field)
    xx_field, yy_field = np.meshgrid(x_field, y_field, indexing='ij')

    active_start_y = SIO2_THICKNESS + SI3N4_THICKNESS
    active_end_y = active_start_y + ACTIVE_LAYER_THICKNESS
    active_mask = (yy_field >= active_start_y) & (yy_field < active_end_y) & (np.abs(xx_field) < WG_WIDTH / 2)

    active_layer_power = np.sum(intensity[active_mask])
    gamma = np.real(active_layer_power / total_power) # Konfinement-Faktor als Bruchteil

    # VπL Berechnung (vereinfachte Annahme Γ_eo ≈ Γ_opt)
    n_active = materials[active_material_name]
    r_eff = pockels_coeffs[active_material_name]
    v_pi_l_meters = (WAVELENGTH * ELECTRODE_GAP) / (n_active**3 * r_eff * gamma)
    v_pi_l_cm = v_pi_l_meters * 100 # Umrechnung in V·cm

    print("\n--- CONSOLE OUTPUT FOR PAPER ---")
    print(f"  RESULTS FOR: {active_material_name}")
    print(f"  Effective Refractive Index (n_eff): {n_eff:.4f}")
    print(f"  Optical Confinement Factor (Gamma): {gamma*100:.2f}%")
    print(f"  PROJECTED VπL: {v_pi_l_cm:.4f} V·cm")
    print("---------------------------------")

    # --- 4. Visualisierung ---
    print("Creating and saving plots...")
    plot_extent = [x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6] # Achsen in µm für die Plots

    # Plot 1: Brechungsindex-Profil
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    n_plot = n_profile_func(x, y)
    im1 = ax1.imshow(n_plot.T, extent=plot_extent, origin='lower', aspect='auto', cmap='viridis')
    ax1.set_title(f"Refractive Index Profile ({active_material_name})")
    ax1.set_xlabel(r'x ($\mu$m)')
    ax1.set_ylabel(r'y ($\mu$m)')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Refractive Index n")
    fig1.tight_layout()
    filename_n = f"simulation_n_profile_{active_material_name}.pdf"
    plt.savefig(filename_n)
    print(f"  - Structure plot saved as '{filename_n}'")
    plt.close(fig1)

    # Plot 2: Moden-Intensität
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    im2 = ax2.imshow(intensity.T, extent=plot_extent, origin='lower', aspect='auto', cmap='inferno')
    ax2.set_title(f"Fundamental Mode Intensity (n_eff={n_eff:.4f})")
    ax2.set_xlabel(r'x ($\mu$m)')
    ax2.set_ylabel(r'y ($\mu$m)')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Intensity (a.u.)")
    fig2.tight_layout()
    filename_intensity = f"simulation_intensity_{active_material_name}.pdf"
    plt.savefig(filename_intensity)
    print(f"  - Intensity plot saved as '{filename_intensity}'")
    plt.close(fig2)

# --- 5. Haupt-Ausführungsblock ---
if __name__ == "__main__":
    simulate_waveguide(active_material_name='BTO')
    simulate_waveguide(active_material_name='AlN')
    simulate_waveguide(active_material_name='TFLN')
    print("\nAll simulations finished.")
