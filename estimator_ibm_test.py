from qiskit import QuantumCircuit,transpile
from math import pi
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
#from qiskit.primitives import BaseEstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from  qiskit_ibm_runtime import SamplerV2 as Sampler
from Sampler_to_est_utils import sampler_expectation_and_std_option_a


qc_dyn = QuantumCircuit(1, 1)
qc_dyn.r(pi, 0, 0)
qc_dyn.measure(0,0)
with qc_dyn.if_test((qc_dyn.clbits[0],1)):
    qc_dyn.z(0)

service = QiskitRuntimeService()
backend1 = service.backend("ibm_torino")
qc_dyn = transpile(qc_dyn, backend1)
sampler = Sampler(mode=backend1)
obs = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=1)
obs = obs.apply_layout(qc_dyn.layout)
ev, std = sampler_expectation_and_std_option_a(
    sampler,
    qc_dyn,
    obs,
)

print("EV :", ev)
print("STD:", std)
print("Unexpectedly ran estimator.")
