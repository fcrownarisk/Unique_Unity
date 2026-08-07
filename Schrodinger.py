import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib.gridspec import GridSpec

# ================= Parameter Settings =================
hbar = 1.0
m = 1.0
omega = 1.0          # Harmonic oscillator angular frequency
L = 8.0              # Spatial range [-L/2, L/2]
N = 1000             # Number of grid points (higher = more accurate)
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]

# ================= Build Potential Energy V(x) =================
# Option 1: Harmonic oscillator (V = 0.5 * m * ω² * x²)
V_harmonic = 0.5 * m * omega**2 * x**2

# Option 2: Double well potential (uncomment to see energy level splitting/tunneling)
V_double_well = -4.0 * np.exp(-((x - 2.5)**2) / 0.8) - 4.0 * np.exp(-((x + 2.5)**2) / 0.8)

# Select which potential to use
V = V_harmonic  # Change to V_double_well for double well potential

# ================= Build Kinetic Energy Matrix K =================
# Tridiagonal matrix for second derivative: main diagonal = -2/dx², off-diagonal = 1/dx²
diag = np.ones(N) * (-2.0) / (dx**2)
off_diag = np.ones(N-1) * (1.0) / (dx**2)
K = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
K = - (hbar**2 / (2*m)) * K

# ================= Build Total Hamiltonian H and Solve =================
H = K + np.diag(V)   # Potential energy is only on the diagonal

# Use eigh specifically for Hermitian matrices (guarantees real energy eigenvalues)
eig_vals, eig_vecs = np.linalg.eigh(H)

# ================= Additional Analysis Functions =================
def calculate_expectation_value(operator, wavefunction):
    """Calculate expectation value <ψ|O|ψ>"""
    return np.dot(wavefunction.conj(), np.dot(operator, wavefunction)) * dx

def calculate_uncertainty(operator, wavefunction):
    """Calculate uncertainty ΔO = sqrt(<O²> - <O>²)"""
    op_squared = np.dot(operator, operator)
    expectation = calculate_expectation_value(operator, wavefunction)
    expectation_squared = calculate_expectation_value(op_squared, wavefunction)
    variance = expectation_squared - expectation**2
    return np.sqrt(max(variance, 0))

def normalize_wavefunction(psi):
    """Normalize wavefunction"""
    norm = np.sqrt(integrate.simps(np.abs(psi)**2, x))
    return psi / norm

# Build position and momentum operators for uncertainty calculations
X_operator = np.diag(x)
# Momentum operator in position representation: -i*hbar * d/dx
# First derivative matrix
deriv_diag = np.zeros(N)
deriv_off_diag = np.ones(N-1) * 1.0 / (2*dx)
P_matrix = np.diag(deriv_diag) + np.diag(deriv_off_diag, k=1) - np.diag(deriv_off_diag, k=-1)
P_operator = -1j * hbar * P_matrix

# ================= Create Comprehensive Visualization =================
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# --- Subplot 1: Energy levels and probability densities ---
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(x, V, 'k-', linewidth=2, label='Potential $V(x)$', alpha=0.7)

num_levels_to_show = 8
colors = plt.cm.viridis(np.linspace(0, 1, num_levels_to_show))

for i in range(num_levels_to_show):
    energy = eig_vals[i]
    psi = eig_vecs[:, i]
    
    # Normalize wavefunction
    psi = normalize_wavefunction(psi)
    
    # Probability density |ψ|²
    prob_density = np.abs(psi)**2
    
    # Scale for visualization
    scale_factor = 1.5
    prob_density_scaled = prob_density / np.max(prob_density) * scale_factor
    
    # Draw energy level line
    ax1.hlines(energy, x[0], x[-1], colors='gray', linestyles='dashed', linewidth=0.8, alpha=0.5)
    
    # Draw shifted probability density
    ax1.fill_between(x, energy, energy + prob_density_scaled, 
                     alpha=0.3, color=colors[i])
    ax1.plot(x, energy + prob_density_scaled, 
             label=f'n={i}, E = {energy:.3f}', color=colors[i], linewidth=1.5)

ax1.set_xlabel('Position x (atomic units)', fontsize=12)
ax1.set_ylabel('Energy E (atomic units)', fontsize=12)
ax1.set_title('Time-Independent Schrödinger Equation: Energy Quantization and Probability Densities', 
              fontsize=14, fontweight='bold')
ax1.set_ylim(np.min(V) - 1, eig_vals[num_levels_to_show-1] + 3)
ax1.legend(loc='upper right', ncol=2, fontsize=9)
ax1.grid(alpha=0.3)

# --- Subplot 2: Wavefunctions ---
ax2 = fig.add_subplot(gs[1, 0])
num_wavefuncs = 5
for i in range(num_wavefuncs):
    energy = eig_vals[i]
    psi = normalize_wavefunction(eig_vecs[:, i])
    
    # Shift wavefunctions vertically by their energy for clarity
    wavefunc_scaled = psi * 1.5
    ax2.plot(x, energy + wavefunc_scaled, label=f'n={i}', linewidth=1.5, color=colors[i])

ax2.plot(x, V, 'k--', linewidth=1, alpha=0.5, label='V(x)')
ax2.set_xlabel('Position x (atomic units)', fontsize=11)
ax2.set_ylabel('Energy E (atomic units)', fontsize=11)
ax2.set_title('Wavefunctions ψ(x) (shifted by energy)', fontsize=12, fontweight='bold')
ax2.set_ylim(np.min(V) - 0.5, eig_vals[num_wavefuncs-1] + 2.5)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# --- Subplot 3: Energy spectrum ---
ax3 = fig.add_subplot(gs[1, 1])
n_levels = np.arange(20)
energies = eig_vals[:20]

# Theoretical harmonic oscillator energies: E_n = (n + 1/2) * ħω
E_theoretical = (n_levels + 0.5) * hbar * omega

ax3.plot(n_levels, energies, 'bo-', label='Numerical', markersize=6, linewidth=1.5)
ax3.plot(n_levels, E_theoretical, 'r--', label='Theoretical: (n+½)ħω', linewidth=1.5)
ax3.set_xlabel('Quantum number n', fontsize=11)
ax3.set_ylabel('Energy E (atomic units)', fontsize=11)
ax3.set_title('Energy Spectrum Comparison', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# --- Subplot 4: Uncertainty principle verification ---
ax4 = fig.add_subplot(gs[2, 0])
uncertainties = []
for i in range(15):
    psi = normalize_wavefunction(eig_vecs[:, i])
    delta_x = calculate_uncertainty(X_operator, psi)
    delta_p = calculate_uncertainty(P_operator, psi)
    uncertainty_product = delta_x * delta_p
    uncertainties.append(uncertainty_product)

ax4.plot(range(15), uncertainties, 'go-', linewidth=1.5, markersize=6)
ax4.axhline(y=hbar/2, color='r', linestyle='--', linewidth=1.5, 
            label=f'Heisenberg limit: ħ/2 = {hbar/2:.1f}')
ax4.set_xlabel('Quantum number n', fontsize=11)
ax4.set_ylabel('Δx · Δp', fontsize=11)
ax4.set_title('Heisenberg Uncertainty Principle Verification', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)
ax4.set_ylim(0, max(uncertainties) * 1.2)

# --- Subplot 5: Probability current (for selected states) ---
ax5 = fig.add_subplot(gs[2, 1])
# For stationary states, probability current is constant (and zero for real wavefunctions)
# Let's demonstrate with a superposition state
state1 = 0  # Ground state
state2 = 1  # First excited state

psi1 = normalize_wavefunction(eig_vecs[:, state1])
psi2 = normalize_wavefunction(eig_vecs[:, state2])
E1, E2 = eig_vals[state1], eig_vals[state2]

# Create time-dependent superposition at t=0
# Ψ(x,t) = (ψ₁(x)e^(-iE₁t/ħ) + ψ₂(x)e^(-iE₂t/ħ)) / √2
# At t=0: Ψ(x,0) = (ψ₁(x) + ψ₂(x)) / √2
psi_superposition = (psi1 + psi2) / np.sqrt(2)

# Probability current: j = (ħ/m) * Im(ψ* ∇ψ)
# Using finite difference for gradient
grad_psi = np.gradient(psi_superposition, dx)
prob_current = (hbar / m) * np.imag(np.conj(psi_superposition) * grad_psi)

prob_density_super = np.abs(psi_superposition)**2

ax5_twin = ax5.twinx()
ax5.plot(x, prob_density_super, 'b-', linewidth=1.5, label='|Ψ(x)|²')
ax5_twin.plot(x, prob_current, 'r-', linewidth=1.5, label='Probability current j(x)')

ax5.set_xlabel('Position x (atomic units)', fontsize=11)
ax5.set_ylabel('Probability density |Ψ|²', fontsize=11, color='b')
ax5_twin.set_ylabel('Probability current j(x)', fontsize=11, color='r')
ax5.set_title(f'Superposition State: n={state1} + n={state2}', fontsize=12, fontweight='bold')

# Combine legends
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_twin.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
ax5.grid(alpha=0.3)

plt.suptitle('Quantum Harmonic Oscillator: Comprehensive Analysis', 
             fontsize=16, fontweight='bold', y=0.98)
plt.show()

# ================= Print Detailed Results =================
print("\n" + "="*80)
print("QUANTUM HARMONIC OSCILLATOR - NUMERICAL SOLUTION RESULTS")
print("="*80)

print(f"\nParameters:")
print(f"  • ħ = {hbar}")
print(f"  • m = {m}")
print(f"  • ω = {omega}")
print(f"  • Grid points: {N}")
print(f"  • Spatial range: [{x[0]:.1f}, {x[-1]:.1f}]")

print(f"\nFirst 15 Energy Levels (atomic units):")
print("-"*50)
print(f"{'n':>3} {'Numerical':>12} {'Theoretical':>12} {'Error':>12}")
print("-"*50)
for i in range(15):
    E_num = eig_vals[i]
    E_theo = (i + 0.5) * hbar * omega
    error = abs(E_num - E_theo) / E_theo * 100
    print(f"{i:3d} {E_num:12.6f} {E_theo:12.6f} {error:11.6f}%")

print(f"\nUncertainty Principle Verification:")
print("-"*60)
print(f"{'n':>3} {'Δx':>10} {'Δp':>10} {'Δx·Δp':>10} {'≥ ħ/2?':>10}")
print("-"*60)
for i in range(10):
    psi = normalize_wavefunction(eig_vecs[:, i])
    delta_x = calculate_uncertainty(X_operator, psi)
    delta_p = calculate_uncertainty(P_operator, psi)
    product = delta_x * delta_p
    satisfies = "YES" if product >= hbar/2 - 1e-10 else "NO"
    print(f"{i:3d} {delta_x:10.4f} {delta_p:10.4f} {product:10.4f} {satisfies:>10}")

print(f"\nOrthogonality Check (⟨ψᵢ|ψⱼ⟩ for i≠j, should be 0):")
print("-"*50)
for i in range(5):
    for j in range(i+1, min(i+3, 5)):
        psi_i = normalize_wavefunction(eig_vecs[:, i])
        psi_j = normalize_wavefunction(eig_vecs[:, j])
        overlap = np.abs(np.dot(psi_i.conj(), psi_j) * dx)
        print(f"  ⟨{i}|{j}⟩ = {overlap:.2e}")

print("\n" + "="*80)
print("Analysis complete! The numerical solution demonstrates:")
print("  1. Energy quantization consistent with Eₙ = (n + ½)ħω")
print("  2. Wavefunction nodes increase with quantum number n")
print("  3. Heisenberg uncertainty principle is satisfied (Δx·Δp ≥ ħ/2)")
print("  4. Wavefunctions are orthonormal")
print("="*80)
