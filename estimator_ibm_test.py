from qiskit import QuantumCircuit
from math import pi
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp


qc_dyn = QuantumCircuit(1, 1)
qc_dyn.r(pi, 0, 0)
qc_dyn.measure(0,0)
with qc_dyn.if_test((qc_dyn.clbits[0],1)):
    qc_dyn.z(0)

service = QiskitRuntimeService()
backend1 = service.backend("ibm_torino")

obs = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=1)

est = Estimator(mode=backend1)

job = est.run([(qc_dyn, obs)]) 
res = job.result()
print("Unexpectedly ran estimator.")
print(res, res[0].data.evs)