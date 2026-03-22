from qiskit import QuantumCircuit, ClassicalRegister
import numpy as np

def build_pauli_measurement_circuit(
    circuit: QuantumCircuit,
    pauli_string: str,
):
    """
    Build a circuit that measures ONLY the support of a Pauli string,
    allocating one classical bit per non-identity Pauli.

    Returns
    -------
    meas_circuit : QuantumCircuit
    qubit_indices : list[int]
        Mapping from classical bit index -> qubit index
    """
    pauli_string = pauli_string[::-1]  # Qiskit endianness

    # qubits to measure (Pauli support)
    qubit_indices = [i for i, p in enumerate(pauli_string) if p != "I"]

    # special case: all-identity Pauli (Estimator behavior)
    if not qubit_indices:
        qubit_indices = [0]

    qc = circuit.copy()
    qc.remove_final_measurements(inplace=True)

    creg = ClassicalRegister(len(qubit_indices),"meas")
    qc.add_register(creg)

    for clbit, qubit in enumerate(qubit_indices):
        p = pauli_string[qubit]

        if p == "X":
            qc.h(qubit)
        elif p == "Y":
            qc.sdg(qubit)
            qc.h(qubit)
        # Z → no rotation

        qc.measure(qubit, clbit)
    measured_pauli_support = "Z" * len(qubit_indices)


    return qc, qubit_indices,measured_pauli_support


def expectation_from_support_counts(counts: dict) -> float:
    """
    Compute <P> from counts where each bit corresponds to
    one non-identity Pauli qubit.
    """
    shots = sum(counts.values())
    exp = 0.0

    for bitstring, count in counts.items():
        # parity = (-1)^(number of 1s)
        parity = -1 if bitstring.count("1") % 2 else 1
        exp += parity * count / shots

    return exp


def pauli_std_from_counts(counts: dict, ev: float) -> float:
    shots = sum(counts.values())
    return np.sqrt((1.0 - ev**2) / shots)


def sampler_expectation_and_std_option_a(
    sampler,
    circuit: QuantumCircuit,
    pauli_string: str,
    coeff: float = 1.0,
):
    # 1) build Estimator-style measurement circuit
    meas_circuit, qubit_indices,new_pauli = build_pauli_measurement_circuit(
        circuit, pauli_string
    )

    # 2) run sampler
    job = sampler.run([meas_circuit], shots= 1024)
    result = job.result()

    # SamplerV2 shape handling
    #if hasattr(raw_result, "results"):
        #res = raw_result.results[0]
    #else:
        #res = raw_result
    print(pauli_string,new_pauli)
    # 3) get support-only bitstrings
    ev = result[0].data.meas.expectation_values(new_pauli)

    # 4) expectation + std
    std = 0

    return coeff * ev, std