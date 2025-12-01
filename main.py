import logging

# Enable detailed debug output for Qiskit + Braket internals
# logging.getLogger("qiskit").setLevel(logging.DEBUG)
# logging.getLogger("qiskit_braket_provider").setLevel(logging.DEBUG)
# logging.getLogger("braket").setLevel(logging.DEBUG)

# # Optional: send logs to console explicitly
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# %%
#run options
target_fidelity = .996
#Degredation method
depol_U_bool = True
depol_Pauli_bool = False
depol_QPD = False
#Run Method
braket_hw_bool = False
noise_model_bool = True

#Set delay value options
delay_ns = 10

# %%
from qiskit.circuit.library import UnitaryGate
import numpy as np
from importlib import reload
import unitary_test as unitary_test
reload(unitary_test)
from unitary_test import depol_U_8x8, p_from_target_fidelity, depol_stinespring

if depol_U_bool:
    p = p_from_target_fidelity(target_fidelity)
    U = UnitaryGate(depol_U_8x8(p), label="Depol")
        
elif depol_Pauli_bool:
    p = (4/3)*p_from_target_fidelity(target_fidelity)
    dep_gate = depol_stinespring(p)
        


# %%
#This builds a linear graph state 0-1-2-3
from qiskit import QuantumCircuit, transpile

def make_base_graph():
    qc = QuantumCircuit(6)
    for q in [0,1,2,3,4,5]:
        qc.h(q)  
    #qc.cz(3,0)
    for a, b in [(0,1),(1,2),(2,3),(3,4),(4,5)]:
        qc.cz(a,b)
    # leave out cz(0,3) here – we’ll insert it via the three methods
    return qc

qc_real = make_base_graph()
emerald_native = ["r", "cz", "measure"]
if braket_hw_bool:
    qc_real= transpile(
        qc_real,
        basis_gates=emerald_native,
        optimization_level=3
    )
qc_real.draw('mpl')

# %%
#This builds the cut bell pair factory for one cut bell pair
from qiskit import qpy
from qiskit import transpile
with open("data/one_bell_pair.qpy", "rb") as fd:
    cut_bell_pair_factory1 = qpy.load(fd)[0]
if depol_U_bool:
    cut_bell_pair_factory = QuantumCircuit(6)

    cut_bell_pair_factory.compose(cut_bell_pair_factory1,[2,3], inplace = True)
    cut_bell_pair_factory.compose(U,[2,1,0], inplace = True)
    cut_bell_pair_factory.compose(U,[3,4,5], inplace = True)
elif depol_Pauli_bool:
    cut_bell_pair_factory = QuantumCircuit(8)
    cut_bell_pair_factory.compose(cut_bell_pair_factory1,[3,4], inplace = True)
    cut_bell_pair_factory.compose(dep_gate,[3,0,1,2], inplace=True)
    cut_bell_pair_factory.compose(dep_gate,[4,5,6,7], inplace=True)
else: 
    cut_bell_pair_factory = QuantumCircuit(2)

    cut_bell_pair_factory.compose(cut_bell_pair_factory1,[0,1], inplace = True)

#cut_bell_pair_factory.draw("mpl", fold=False)

if braket_hw_bool:
    cut_bell_pair_factory= transpile(
        cut_bell_pair_factory,
        basis_gates=emerald_native,
        optimization_level=3
    )
cut_bell_pair_factory.draw('mpl')

# %%
#This puts the cut bell pair factory in a teleportation circuit
from math import pi
from qiskit_braket_provider.providers.braket_instructions import CCPRx, MeasureFF

if depol_U_bool:
    teleportation_circuit = QuantumCircuit(8,2)
elif depol_Pauli_bool:
    teleportation_circuit = QuantumCircuit(10,2)
else:
    teleportation_circuit = QuantumCircuit(4,2)
#teleportation_circuit.h(2)
#teleportation_circuit.h(1)
#teleportation_circuit.h(0)
#teleportation_circuit.h(3)

if depol_U_bool:
    teleportation_circuit.compose(cut_bell_pair_factory,[4,5,1,2,6,7],inplace=True)
elif depol_Pauli_bool:
    teleportation_circuit.compose(cut_bell_pair_factory,[4,5,6,1,2,7,8,9],inplace=True)
else: 
    teleportation_circuit.compose(cut_bell_pair_factory,[1,2],inplace=True)
#teleportation_circuit.h(3)

if noise_model_bool:
    teleportation_circuit.h(3)
    teleportation_circuit.cx(0,1)
    teleportation_circuit.cx(2,3)
    teleportation_circuit.h(2)
    teleportation_circuit.measure(1,0)
    teleportation_circuit.measure(2,1)
    if delay_ns > 0:
        teleportation_circuit.delay(delay_ns, [0,3], "ns") 

    with teleportation_circuit.if_test((teleportation_circuit.clbits[1],1)):
        teleportation_circuit.z(0)
    with teleportation_circuit.if_test((teleportation_circuit.clbits[0],1)):
        teleportation_circuit.x(3)
    teleportation_circuit.h(3)
else: 
    

    # H(3)  (your initial Ry(pi/2); Rx(pi) on q3)
    teleportation_circuit.r(np.pi/2, np.pi/2, 3)
    teleportation_circuit.r(np.pi, 0, 3)

    # --- CX(0,1) = H(1) · CZ(0,1) · H(1) ---
    teleportation_circuit.r(np.pi/2, np.pi/2, 1)
    teleportation_circuit.r(np.pi, 0, 1)
    teleportation_circuit.cz(0, 1)
    teleportation_circuit.r(np.pi/2, np.pi/2, 1)
    teleportation_circuit.r(np.pi, 0, 1)

    # --- CX(2,3) = H(3) · CZ(2,3) · H(3) ---
    teleportation_circuit.r(np.pi/2, np.pi/2, 3)
    teleportation_circuit.r(np.pi, 0, 3)
    teleportation_circuit.cz(2, 3)
    teleportation_circuit.r(np.pi/2, np.pi/2, 3)
    teleportation_circuit.r(np.pi, 0, 3)

    # H(2)
    teleportation_circuit.r(np.pi/2, np.pi/2, 2)
    teleportation_circuit.r(np.pi, 0, 2)

    # feed-forward measurements (kept as-is)
    teleportation_circuit.append(MeasureFF(feedback_key=0), qargs=[1])
    teleportation_circuit.append(MeasureFF(feedback_key=1), qargs=[2])

    # Step 2: conditional Z on qubit 0 (your pattern: H · ccprx(pi,0) · H)
    teleportation_circuit.r(np.pi/2, np.pi/2, 0)
    teleportation_circuit.r(np.pi, 0, 0)
    teleportation_circuit.append(CCPRx(np.pi, 0, feedback_key=1), qargs=[0])
    teleportation_circuit.r(np.pi/2, np.pi/2, 0)
    teleportation_circuit.r(np.pi, 0, 0)

    # Step 3: conditional X on qubit 3 (ccprx directly), then H(3)
    teleportation_circuit.append(CCPRx(np.pi, 0, feedback_key=0), qargs=[3])
    teleportation_circuit.r(np.pi/2, np.pi/2, 3)
    teleportation_circuit.r(np.pi, 0, 3)

    




teleportation_circuit.draw('mpl')

# %%
# now put the cut bell pair into the graph state 
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
if depol_U_bool:
    cut_graph_state = QuantumCircuit(12,2)
elif depol_Pauli_bool:
    cut_graph_state = QuantumCircuit(14,2)
else: 
    cut_graph_state = QuantumCircuit(8,2)
cut_graph_state.compose(qc_real, [0,1,2,3,4,5], inplace=True)
if noise_model_bool and delay_ns>0:
    cut_graph_state.delay(delay_ns, [1,2,3,4], "ns")
if depol_U_bool:
    cut_graph_state.compose(teleportation_circuit, [0,6,7,5,8,9,10,11], [0,1], inplace=True)
elif depol_Pauli_bool:
    cut_graph_state.compose(teleportation_circuit, [0,6,7,5,8,9,10,11,12,13], [0,1], inplace=True)
else: 
    cut_graph_state.compose(teleportation_circuit, [0,6,7,5], [0,1], inplace=True)
cut_graph_state.draw('mpl')
#service = QiskitRuntimeService() 
#backend1 = service.backend("ibm_torino")
#tc = transpile(cut_graph_state, backend=backend1, optimization_level=3)
#tc.draw('mpl')











# %%
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit_braket_provider import BraketProvider, BraketLocalBackend
from qiskit_braket_provider import BraketAwsBackend, to_braket
from qiskit.quantum_info import SparsePauliOp
import importlib, estimator_helper_functions
importlib.reload(estimator_helper_functions)
from estimator_helper_functions import VerbatimBraketBackend, ring_observables
from braket.experimental_capabilities import EnableExperimentalCapability
import traceback
import statistics
if noise_model_bool:
    from qiskit_aer.primitives import EstimatorV2 as Estimator
else:
    from qiskit.primitives import BackendEstimatorV2 as Estimator




if depol_QPD:
    num_qubits = 8
elif depol_Pauli_bool:
    num_qubits = 14
elif depol_U_bool:
    num_qubits = 12


# ---- load QPD parameter tuples ----
thetas = np.loadtxt("data/one_qpd_bell_pair_param_values.txt")
if depol_QPD:
    I_LOCC = len(thetas)
else:
    I_LOCC = len(thetas) -2

#observables = {
#    "S0":  SparsePauliOp.from_sparse_list([("ZZX", [3,1,0], 1)], num_qubits=num_qubits),
#    "S3":  SparsePauliOp.from_sparse_list([("ZZX", [0,2,3], 1)], num_qubits=num_qubits),
#    "S03": SparsePauliOp.from_sparse_list([("YZZY", [0,1,2,3], 1)], num_qubits=num_qubits)
#}
observables = ring_observables(n=6, num_qubits=num_qubits)
print(observables)
#-------Simulator setup start here-----------
# Setup simulator and transpile circuit
#run on real noise model
if noise_model_bool:
    service = QiskitRuntimeService()
    hw = service.backend("ibm_torino")
    props = hw.properties()
    print(props.t1(0), props.t2(0))
# 2) Aer simulator with Torino noise (keeps dynamics by NOT copying coupling_map)
    noise = NoiseModel.from_backend(hw)
#This applies 
    tau_ns = 1
    tau_s  = tau_ns * 1e-9
    for q in range(num_qubits):
        T1 = props.t1(q)
        T2 = props.t2(q)
        if T1 is None or T2 is None or T1 <= 0:
            continue
        T2_eff = min(T2, 2.0*T1)  # enforce T2 <= 2*T1
        err = thermal_relaxation_error(T1, T2_eff, tau_s)
        noise.add_quantum_error(err, "delay", [q])
    backend = AerSimulator(
        noise_model=noise,
        basis_gates=noise.basis_gates,
        coupling_map=None,          # keep dynamic-circuit-friendly
    )
    qct = transpile(cut_graph_state,backend)
else:
#-------Simulator setup end here-----------
#-------Real computer setup start here-----------
    backend = VerbatimBraketBackend("arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald")
    qct = cut_graph_state
#-------Real computer setup end here-----------


if noise_model_bool:
    exact_estimator = Estimator.from_backend(backend, options={ "run_options": {"shots": 1024}}) #Use for simulator
else:
    if depol_QPD:
        backend.qubit_labels = [27,19,11,5,1,2,28,6]
    elif depol_Pauli_bool:
        backend.qubit_labels = [27,26,18,10,4,5,28,6,29,20,36,12,7,2]
    #elif depol_U_bool:n can't do this rn
        
    exact_estimator = Estimator(backend=backend) #Use for real computer
    #quantum computer down can't actually test this
    


isa_circuit = qct


# --- collect means & variances from EstimatorV2 ---

results_mean = {}
results_var  = {}  # store variance = std**2

for i in range(I_LOCC):
    theta = thetas[i]
    print(i)
    results_mean[i] = {}
    results_var[i]  = {}

    params = list(isa_circuit.parameters)
    bind_map = dict(zip(params, theta))
    bound = isa_circuit.assign_parameters(bind_map, inplace=False)

    for label, observable in observables.items():
        pub = (isa_circuit, observable, theta)
        if noise_model_bool:
            sim_results =[]
            for r in range(100):
                job = exact_estimator.run([pub])
                result = job.result()
                sim_results.append(float(result[0].data.evs))
            ev = float(sum(sim_results)/len(sim_results))
            std = statistics.stdev(sim_results)
        else:
            with EnableExperimentalCapability():
                try:
                    job = exact_estimator.run([pub])
                    result = job.result()
                except Exception as e:
                    import pdb, traceback
                    traceback.print_exc()
                    pdb.post_mortem()

            ev  = float(result[0].data.evs)
            std = float(result[0].data.stds)
        results_mean[i][label] = ev
        results_var[i][label]  = std * std  # store variance
        print("label", label, ev, std)

# --- combine QPD terms: keep both mean and variance ---

mu = (4*target_fidelity - 1) / 3

# initialize accumulators
labels_any = list(next(iter(results_mean.values())).keys())
expect_bell_mean = {label: 0.0 for label in labels_any}
expect_bell_var  = {label: 0.0 for label in labels_any}
expect_id_mean   = {label: 0.0 for label in labels_any}
expect_id_var    = {label: 0.0 for label in labels_any}

# accumulate with coefficients; for variance, square the coeff
for i in range(I_LOCC):
    for label in labels_any:
        x_mean = results_mean[i][label]
        x_var  = results_var[i][label]

        if i < 3:
            c = 2/9
            expect_bell_mean[label] += c * x_mean
            expect_bell_var[label]  += (c*c) * x_var
        elif i < 5:
            c = -1/6
            expect_bell_mean[label] += c * x_mean
            expect_bell_var[label]  += (c*c) * x_var

        if depol_QPD and i in (3, 4, 5, 6):
            c = 0.25
            expect_id_mean[label] += c * x_mean
            expect_id_var[label]  += (c*c) * x_var

# final mixture
expect_mean = {}
expect_var  = {}

for label in labels_any:
    if depol_QPD:
        # y = mu*(3*b) + (1-mu)*i  -> note the factor 3 on bell part
        y_mean = mu * (3.0 * expect_bell_mean[label]) + (1.0 - mu) * expect_id_mean[label]
        y_var  = (mu*3.0)**2 * expect_bell_var[label] + (1.0 - mu)**2 * expect_id_var[label]
    else:
        # y = 3*b
        y_mean = 3.0 * expect_bell_mean[label]
        y_var  = (3.0**2) * expect_bell_var[label]

    expect_mean[label] = y_mean
    expect_var[label]  = y_var

# --- print stabilizers with 1σ uncertainty ---
for i in range(6):
    key = f"S{i}"
    m = expect_mean[key]
    s = (expect_var[key] ** 0.5)
    print(f"⟨{key}⟩ = {m:.5f} ± {s:.5f}")

# neighboring pairs on the 6-qubit ring
for i in range(6):
    j = (i + 1) % 6
    key = f"S{i}{j}"
    m = expect_mean[key]
    s = (expect_var[key] ** 0.5)
    print(f"⟨{key}⟩ = {m:.5f} ± {s:.5f}")

# --- pairwise entanglement witnesses with propagated σ ---
for i in range(6):
    j = (i + 1) % 6
    k_ij = f"S{i}{j}"
    # W = (1 - S_i - S_j - S_ij)/4
    w_mean = (1.0 - expect_mean[f"S{i}"] - expect_mean[f"S{j}"] - expect_mean[k_ij]) / 4.0
    # Var(W) = (1/4)^2 * (Var(S_i) + Var(S_j) + Var(S_ij)), assuming independence
    w_var  = (1.0/4.0)**2 * (expect_var[f"S{i}"] + expect_var[f"S{j}"] + expect_var[k_ij])
    w_std  = (w_var ** 0.5)
    print(f"Entanglement witness W{i}{j} = {w_mean:.5f} ± {w_std:.5f}")


#Batching runs -- good idea don't have time to implement
# #if noise_model_bool:
# # Prepare results data structure
# results_dict = {}

# for i in range(I_LOCC):
#     theta = thetas[i]
#     print(i)
#     results_dict[i] = {}

#     params = list(isa_circuit.parameters)
#     bind_map = dict(zip(params, theta))
#     bound = isa_circuit.assign_parameters(bind_map, inplace=False)

#     for label, observable in observables.items():
#         pub = (isa_circuit, observable, theta)
#         if noise_model_bool:
#             job = exact_estimator.run([pub])
#             result = job.result()
#         else:
#             with EnableExperimentalCapability():
#                 try:
#                     job = exact_estimator.run([pub])
#                     result = job.result()
#                 except Exception as e:
#                     import pdb, traceback
#                     traceback.print_exc()
#                     pdb.post_mortem()
#         exact_value = float(result[0].data.evs)
#         variance = (result[0].data.stds)
#         results_dict[i][label] = exact_value
#         print("label", label, exact_value, variance)
# # else:
# #    # Prepare results data structure
# #     results_dict = {}
# #     pubs =[]

# #     for i in range(I_LOCC):
# #         theta = thetas[i]
# #         print(i)
# #         results_dict[i] = {}

# #         params = list(isa_circuit.parameters)
# #         bind_map = dict(zip(params, theta))
# #         bound = isa_circuit.assign_parameters(bind_map, inplace=False)
    
# #         for label, observable in observables.items():
# #             pub = (bound, [observable], [])
# #             pubs.append(pub)
# #             print("example theta len:", len(np.atleast_1d(thetas[0]).ravel()))
            
            
    
# #     with EnableExperimentalCapability():
# #         try:
# #             job = exact_estimator.run([pub])
# #             res = job.result()
# #             print("Job submitted successfully.")
# #         except Exception as e:
# #             print(f"Error running estimator job: {e}")
# #             traceback.print_exc()





# mu = (4*target_fidelity -1) / 3 

# expect_bell = {label: 0.0 for label in next(iter(results_dict.values())).keys()}
# expect_id   = {label: 0.0 for label in next(iter(results_dict.values())).keys()}
# expect      = {label: 0.0 for label in next(iter(results_dict.values())).keys()}

# # both masks are {0,1,2} since we measured 3 classical bits [b2,b1,b0]


# for i, result in results_dict.items():
#     for label in result:
#         if i < 3:
#             expect_bell[label] += (2/9) * result[label]
#         elif i < 5:
#             expect_bell[label] += (-1/6) * result[label]
#         if depol_QPD and i in (3, 4, 5, 6):
#             expect_id[label] += 0.25 * result[label]

# for label in expect:
#     if depol_QPD:
#         expect[label] = mu * (3.0 * expect_bell[label]) + (1.0 - mu) * expect_id[label]
#     else:
#         expect[label] = 3 * expect_bell[label]

# # 6) Final entanglement witness = ⟨S0⟩ + ⟨S3⟩

# # Print <Si> for all single-qubit stabilizers
# for i in range(6):
#     print(f"⟨S{i}⟩ = {expect[f'S{i}']:.5f}")

# # Print <SiSj> for all neighboring pairs in the 6-qubit ring
# for i in range(6):
#     j = (i + 1) % 6
#     key = f"S{i}{j}"
#     print(f"⟨S{i}S{j}⟩ = {expect[key]:.5f}")

# # Example: compute an entanglement witness per pair (similar form as before)
# for i in range(6):
#     j = (i + 1) % 6
#     witness = (1 - expect[f"S{i}"] - expect[f"S{j}"] - expect[f"S{i}{j}"]) / 4.0
#     print(f"Entanglement witness W{i}{j} = {witness:.5f}")


