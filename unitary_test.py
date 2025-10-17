import numpy as np
import math
from qiskit import QuantumCircuit

def depol_U_8x8(p: float) -> np.ndarray:
    a = np.sqrt(1 - p)
    if p == 0:
        b, beta = 0.0, 0.0
    else:
        b = np.sqrt(p/3.0)
        beta = (b*b)/(1 - a)

    I = np.array([[1,0],[0,1]], dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)

    V = np.array([
        [a,    b,      b,      b     ],
        [b, 1-beta,  -beta,  -beta    ],
        [b,  -beta, 1-beta,  -beta    ],
        [b,  -beta,  -beta, 1-beta    ],
    ], dtype=complex)

    # U = diag(I,X,Y,Z) @ (I ⊗ V)  → block formula U_{ij} = V_{ij} * P_i
    P = [I, X, Y, Z]
    U_blocks = [[V[i,j]*P[i] for j in range(4)] for i in range(4)]
    U = np.block(U_blocks)
    # sanity check
    assert np.allclose(U.conj().T @ U, np.eye(8), atol=1e-10)
    return U


def p_from_target_fidelity(F: float) -> float:
    """
    Invert F(p) = 1 - 2p + (4/3)p^2  for two-sided, identical single-qubit depolarizing noise:
        E(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)

    Returns:
        p in [0, 3/4] for F in [1/4, 1].

    Raises:
        ValueError if F is outside the physically reachable range [1/4, 1].
    """
    if not (0.25 <= F <= 1.0):
        raise ValueError("F must be in [1/4, 1] for this parameterization (p ∈ [0, 3/4]).")
    # Closed-form inverse: p(F) = 3/4 - (sqrt(3)/4) * sqrt(4F - 1)
    return 0.75 - 0.25 * math.sqrt(3.0 * (4.0*F - 1.0))



def depol_stinespring(p: float) -> QuantumCircuit:
    # qubit order: [sys, eX, eY, eZ]
    qc = QuantumCircuit(4, name="Depol")
    theta = 0.5 * np.arccos(1 - 2*p)
    qc.ry(theta, 1)
    qc.ry(theta, 2)
    qc.ry(theta, 3)
    qc.cx(1, 0)
    qc.cy(2, 0)         # fallback if cy missing: qc.sdg(0); qc.cx(2,0); qc.s(0)
    qc.cz(3, 0)
    return qc
