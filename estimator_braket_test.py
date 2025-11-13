from qiskit import QuantumCircuit
from math import pi
from qiskit_braket_provider.providers.braket_instructions import CCPRx, MeasureFF
from qiskit_braket_provider import BraketAwsBackend, to_braket
from qiskit.primitives import BackendEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from braket.experimental_capabilities import EnableExperimentalCapability

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
                to_braket(circ, verbatim=True, qubit_labels=[1])
                for circ in circuits ]
        shots = options.pop("shots", None)
        return (
            self._run_program_set(braket_circuits, shots, **options)
            if self._supports_program_sets and shots != 0 and len(braket_circuits) > 1
            else self._run_batch(braket_circuits, shots, **options)
        )
            

qc_dyn = QuantumCircuit(1, 1)
qc_dyn.r(pi, 0, 0)
qc_dyn.append(MeasureFF(feedback_key=0), qargs=[0])
qc_dyn.append(CCPRx(pi, 0, feedback_key=0), qargs=[0])

qd = VerbatimBraketBackend("arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald")

obs = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=1)

est = BackendEstimatorV2(backend=qd)

print("\nEstimator on dynamic circuit (expected to fail):")
with EnableExperimentalCapability():
    job = est.run([(qc_dyn, obs)]) 
    print(qd.running)
    res = job.result()
    print("Unexpectedly ran estimator.")
    print(res, res[0].data.evs)