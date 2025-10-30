from qiskit_braket_provider import BraketAwsBackend, to_braket
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

class VerbatimBraketBackend(BraketAwsBackend):
    running = False
    def run(
        self,
        run_input: QuantumCircuit | list[QuantumCircuit],
        **options,
    ):
        self.running = True
        if isinstance(run_input, QuantumCircuit):
            circuits = [run_input]
        elif isinstance(run_input, list):
            circuits = run_input
        braket_circuits = [
                to_braket(circ, verbatim=True, qubit_labels=[1,2,3,4,5,6,7,8])
                for circ in circuits ]
        shots = options.pop("shots", None)
        return (
            self._run_program_set(braket_circuits, shots, **options)
            if self._supports_program_sets and shots != 0 and len(braket_circuits) > 1
            else self._run_batch(braket_circuits, shots, **options)
        )

def ring_observables(n: int = 6, num_qubits: int | None = None):
    """
    Build dict of observables for an n-qubit ring:
    - S{i}:      X_i Z_{i-1} Z_{i+1}
    S{i}{j}:   (for neighbors j=i+1 mod n) = Y_i Y_j Z_{i-1} Z_{j+1}
    """
    if num_qubits is None:
        num_qubits = n

    obs = {}

    # Single-site stabilizers S_i
    for i in range(n):
        left  = (i - 1) % n
        right = (i + 1) % n
        # label order corresponds to indices order
        obs[f"S{i}"] = SparsePauliOp.from_sparse_list(
            [("ZZX", [left, right, i], 1.0)],
            num_qubits=num_qubits,
        )

    # Neighbor-pair products S_i * S_{i+1}
    for i in range(n):
        j     = (i + 1) % n
        left  = (i - 1) % n         # neighbor of i other than j
        right = (j + 1) % n         # neighbor of j other than i (i+2)
        obs[f"S{i}{j}"] = SparsePauliOp.from_sparse_list(
            [("YZZY", [i, left, right,j], 1.0)],
            num_qubits=num_qubits,
        )

    return obs

# Example for a 6-qubit ring mapped into a larger device layout (e.g., 54 qubits):
