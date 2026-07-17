#### note: this documentation has been generated; claims in it are unverified at best or unwarrented for at the least. emptor caveat


# Macroscopic Quantum Simulated Annealing via Stochastic Mean-Field Boundary Coupling and In-Place Random Circuit Sampling

**Abstract**
The simulation of large-scale adiabatic quantum processes is fundamentally constrained by the exponential memory requirements of exact statevector representations. In this paper, we present the architecture and theoretical mechanics of a high-throughput, volumetric quantum engine designed to simulate a 729-qubit Transverse-Field Ising Model (TFIM) subject to a longitudinal field. To bypass intractable memory limits, the system is spatially decomposed into a macroscopic $3 \times 3 \times 3$ grid of 27 distinct 27-qubit patches. Each patch maintains full internal coherence and is evolved using a second-order Strang-split Trotterization scheme. Entanglement between patches is approximated utilizing a stochastic mean-field boundary coupling approach, injecting finite-sampling variance to model local quantum fluctuations. To rigorously validate the entanglement geometry and Trotter state fidelity, we introduce a novel "Zero-VRAM" In-Place Random Circuit Sampling (RCS) method. This technique maps the state to a Porter-Thomas distribution for linear Cross-Entropy Benchmarking (XEB) and mathematically restores the exact pre-measurement Trotter state via exact inverse adjoint operations, circumventing the need for state vector cloning. The simulation is efficiently distributed using the OpenCL PyQrack backend across a symmetric 6-GPU topography.

---

## I. Introduction
Adiabatic quantum computing and quantum simulated annealing hold significant promise for optimization and solving complex Hamiltonians. However, exact simulation of these processes on classical hardware hits a strict "memory wall." For a system of $N$ qubits, storing the full statevector requires $\mathcal{O}(2^N)$ complex amplitudes. Simulating systems approaching the hundred-qubit regime requires architectural compromises.

In this work, we detail a framework for simulating a 729-qubit lattice through spatial decomposition. By dividing the system into 27 discrete patches of 27 qubits, we reduce the exact memory requirements to tractable levels. To reintroduce inter-patch correlation, we employ a cluster mean-field approximation at the boundaries, enhanced by stochastic noise injection to emulate finite quantum sampling effects. Furthermore, assessing the fidelity and entanglement complexity of the evolving states is paramount. We adapt Random Circuit Sampling (RCS) and linear Cross-Entropy Benchmarking (XEB) protocols [1]—traditionally used to demonstrate quantum computational advantage—into an in-place validation layer that requires zero additional Video RAM (VRAM) overhead.

## II. Hamiltonian Formulation & Trotterized Evolution
The system evolves under a Transverse-Field Ising Model with a longitudinal field, defined by the global Hamiltonian:

$$H = H_x + H_z + H_{zz}$$

where the individual components are given by:
$$H_x = -h_x(t) \sum_{i} X_i$$
$$H_z = -h_z(t) \sum_{i} Z_i$$
$$H_{zz} = -J(t) \sum_{\langle i, j \rangle} Z_i Z_j$$

Here, $X_i$ and $Z_i$ are the Pauli matrices acting on qubit $i$, $h_x(t)$ represents the transverse driving field, $h_z(t)$ the longitudinal field, and $J(t)$ the nearest-neighbor Ising coupling. The simulation interpolates these parameters to perform the simulated anneal.

To simulate the time evolution operator $U(\Delta t) = e^{-i H \Delta t}$, we employ a second-order Strang-split Trotterization scheme [3]:

$$U(\Delta t) \approx e^{-i H_x \frac{\Delta t}{2}} e^{-i (H_z + H_{zz}) \Delta t} e^{-i H_x \frac{\Delta t}{2}}$$

In the specific implementation, this sequence expands to a symmetrical application of single-qubit rotations and entangling gates:
$$R_x(\theta_x / 2) \to R_z(\theta_z / 2) \to ZZ(\theta_{zz}) \to R_z(\theta_z / 2) \to R_x(\theta_x / 2)$$
where $\theta_x = -2 h_x \Delta t$, $\theta_z = -2 h_z \Delta t$, and $\theta_{zz} = -J \Delta t$.

## III. Volumetric Decomposition & Stochastic Mean-Field Boundaries
Executing a continuous 729-qubit evolution is classically intractable. The global lattice is spatially decomposed into a $3 \times 3 \times 3$ grid, yielding 27 patches. Each patch contains 27 qubits and is simulated exactly.

To couple the distinct patches and approximate global entanglement, we utilize a stochastic variant of cluster mean-field theory [2]. At discrete measurement intervals, local 1-qubit Pauli expectation values are extracted for the qubits residing on the boundary faces of each patch:

$$\langle X_i \rangle, \langle Y_i \rangle, \langle Z_i \rangle$$

Rather than passing deterministic expectation values, the engine injects statistical variance to simulate finite-sampling noise, scaling as the inverse square root of the number of effective shots $N_{shots}$:

$$\langle P_i \rangle_{stoch} = \langle P_i \rangle + \eta \sqrt{\frac{\Delta t}{N_{shots}}}$$

where $\eta \sim \mathcal{N}(0, 1 - \langle P_i \rangle^2)$ is Gaussian noise bounded by the local variance. Adjacent patches are then entangled via unitary kicks (rotations) applied to their boundary qubits. These rotations are proportional to the expectation values of the neighboring patch, weighted by an inter-patch coupling coefficient $g_{face}$.

## IV. In-Place Random Circuit Sampling (RCS) for Phase Validation
To validate the entanglement spread and verify the fidelity of the Trotterized state without disrupting the annealing process or doubling memory usage, we introduce the "Zero-VRAM" In-Place RCS method.

1. **Random Unitary Application**: A pseudo-random quantum circuit of depth $d=20$ is applied directly to the exact statevector. This circuit consists of alternating layers of uniform single-qubit rotations $U(\theta, \phi, \lambda)$ and non-overlapping entangling $iSWAP$ gates.
2. **Sampling**: Bitstrings are non-destructively sampled from the post-RCS state.
3. **Probability Scoring**: The exact ideal probability $p_{ideal}(x)$ of each sampled bitstring $x$ is queried directly from the statevector.
4. **Exact Restoration**: The engine dynamically reconstructs the gate list in reverse. Mathematical adjoints are applied ($U(-\theta, -\lambda, -\phi)$ and $adjiswap$) to exactly uncompute the random circuit and precisely restore the original Trotter state vector.

The linear Cross-Entropy Benchmarking (XEB) score [1] is then computed from the sampled probabilities:

$$\text{XEB} = 2^N \langle p_{ideal}(x_i) \rangle - 1$$

As the output distribution of the random circuit approaches a Porter-Thomas distribution, an XEB score near 1 confirms high fidelity and complex entanglement geometries intrinsic to the underlying Trotter state.

## V. Hardware Topography
The engine is architected for high-throughput, hardware-accelerated execution. The 27 exact statevectors are distributed symmetrically across 6 physical AMD Radeon Pro V340 GPUs. Utilizing the OpenCL PyQrack backend, each GPU handles multiple 27-qubit subvolumes. The inter-process communication orchestrates the stochastic boundary exchanges synchronously at predetermined measurement steps.

## VI. References

[1] Sergio Boixo et al. "Characterizing Quantum Supremacy in Near-Term Devices". arXiv:1608.00263 (2016).

[2] F. M. Zimmer et al. "Quantum correlated cluster mean-field theory applied to the transverse Ising model". arXiv:1604.03486 (2016).

[3] Ish Dhand and Barry C. Sanders. "Stability of the Trotter-Suzuki decomposition". arXiv:1403.3469 (2014).
