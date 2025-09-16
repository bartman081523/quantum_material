################################################################################
#
#   simulation.py (The Actual, Final, Working Version)
#
#   The simulation and analysis core is WORKING. This final version fixes the
#   last remaining issue in the plotting code by using np.abs() to convert the
#   complex intensity data into a real-valued float array that matplotlib
#   can display. All labels and units have been added as requested.
#
################################################################################

from EMpy.modesolvers.FD import VFDModeSolver
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Physikalische & chemische Parameter ---
materials = {
    'SiO2': 1.44,
    'Si3N4': 2.00,
    'BTO': 2.36,
    'AlN': 2.10,
    'Air': 1.00
}

# Geometrie- und Simulationsparameter
WAVELENGTH = 1.55
GRID_RESOLUTION = 40
SIM_WIDTH = 4.0
SIM_HEIGHT = 3.0
WG_WIDTH = 1.2
SI3N4_THICKNESS = 0.3
SIO2_THICKNESS = 2.0
ACTIVE_LAYER_THICKNESS = 0.1

# --- 2. Kernlogik: Wiederverwendbare Simulationsfunktion ---
def simulate_waveguide(active_material_name, n_modes_to_find=4):
    """
    Defines the waveguide structure, runs the EMpy mode solver,
    and analyzes the results.
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

    x = np.linspace(-SIM_WIDTH / 2, SIM_WIDTH / 2, int(SIM_WIDTH * GRID_RESOLUTION))
    y = np.linspace(0, SIM_HEIGHT, int(SIM_HEIGHT * GRID_RESOLUTION))
    boundary = '0000'

    solver = VFDModeSolver(WAVELENGTH, x, y, eps_func, boundary)

    print("Solving eigenmodes with EMpy... (this may take a moment)")
    solver.solve(neigs=n_modes_to_find)

    # --- 3. Analyse und Auswertung ---
    print(f"Simulation finished. Found {len(solver.modes)} mode(s).")

    if not solver.modes:
        print("WARNING: No guided modes found!")
        return

    fundamental_mode = solver.modes[0]
    n_eff = fundamental_mode.neff.real

    # KORREKTUR: Wandle die komplexe Intensität in ein reelles Float-Array um mit np.abs()
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
    confinement_factor = np.real((active_layer_power / total_power) * 100)

    print(f"\nResults for the fundamental mode:")
    print(f"  Effective Refractive Index (n_eff): {n_eff:.4f}")
    print(f"  Confinement Factor in {active_material_name} (Gamma): {confinement_factor:.2f}%")

    # --- 4. Visualisierung ---
    print("Creating and saving plots...")

    # Plot 1: Brechungsindex-Profil (manuell erstellt)
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    n_plot = n_profile_func(x, y)
    im1 = ax1.imshow(n_plot.T, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', aspect='auto', cmap='viridis')
    ax1.set_title("Refractive Index Profile (Structure)")
    ax1.set_xlabel(r'x ($\mu$m)')
    ax1.set_ylabel(r'y ($\mu$m)')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Refractive Index n")
    fig1.tight_layout()
    filename_n = f"simulation_n_profile_{active_material_name}.pdf"
    plt.savefig(filename_n)
    print(f"  - Structure plot saved as '{filename_n}'")
    plt.close(fig1)

    # Plot 2: Intensität (manuell erstellt für volle Kontrolle)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    im2 = ax2.imshow(intensity.T, extent=[x_field[0], x_field[-1], y_field[0], y_field[-1]], origin='lower', aspect='auto', cmap='inferno')
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
    print("\nAll simulations finished.")
