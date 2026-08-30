import sys
import torch
import time
import logging

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(levelname)s : %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def core_selected(core_type: str = {"CPU, GPU"}):
    if core_type=="CPU":
        return "cpu"
    elif core_type=="GPU":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            print("Warning: Your pc have bo GPU! This core selected will use cpu instad.")
            return "cpu"
    else:
        raise KeyError("Your core_type is not valid, use only CPU or GPU to be an parameter.")

my_core = core_selected("GPU")

#1.1 random m*n with float32 -> move to gpu -> how much memory used

opod = one_point_one_data = {"m": 10000,
                             "n": 3000,
                             "core": "GPU"}
opod_core = core_selected(opod["core"])
opom = one_point_one_matrix = torch.rand(opod["m"], opod["n"], dtype=torch.float32, device=opod_core)
print(f"Currently operation on {opod_core}/{opod["core"]} | Data type is {opom.dtype} | Size per elements in bytes : {opom.element_size()} | Memory Usages: {opom.element_size() * opom.nelement()}")

#1.2 new matrix with same dim but use float 16

optm = one_point_two_matrix = torch.rand(opod["m"], opod["n"], dtype=torch.float16, device=opod_core)
print(f"Currently operation on {opod_core}/{opod["core"]} | Data type is {optm.dtype} | Size per elements in bytes : {optm.element_size()} | Memory Usages: {optm.element_size() * optm.nelement()}")


#-----------------

# torch.cuda.memory._record_memory_history(max_entries=100000)

#2.1 createe m*n and n*k with float32 move to gpu and @

tpod = two_point_one_data = {"m": 100,
                             "n": 35,
                             "k": 75,
                             "core": "GPU"}

tpod_core = core_selected(tpod["core"])
tpoma = two_point_one_matrix_a = torch.rand(tpod["m"], tpod["n"], dtype=torch.float32, device=tpod_core)
tpomb = two_point_one_matrix_b = torch.rand(tpod["n"], tpod["k"], dtype=torch.float32, device=tpod_core)

tpom = tpoma @ tpomb

#2.2 cast 2.1) to float16 and @

tptma = two_point_one_matrix_a.to(torch.float16)
tptmb = two_point_one_matrix_b.to(torch.float16)

tptm = tptma @ tptmb

#2.3 cast 2.1) to bfloat16 and @ 

tp3ma = two_point_one_matrix_a.to(torch.bfloat16)
tp3mb = two_point_one_matrix_b.to(torch.bfloat16)

tp3m = tp3ma @ tp3mb

#2.4 compare result of 2.1 2.2 2.3

###############
print(f"\n2.4 The comparisom of matrix multiply with difference data type " \
      f"\n\t Data type {tpom.dtype} have the first (0, 0) result is {tpom[0][0]} " \
      f"\n\t Data type {tptm.dtype} have the first (0, 0) result is {tptm[0][0]} " \
      f"\n\t Data type {tp3m.dtype} have the first (0, 0) result is {tp3m[0][0]} ")

# torch.cuda.memory._dump_snapshot("2_1_to_2_4.pickle")
# torch.cuda.memory._record_memory_history(enabled=None)

#2.5 do again but mul all data with 100

# torch.cuda.memory._record_memory_history(max_entries=100000)

tpf_oma = tpoma * 100.0
tpf_omb = tpomb * 100.0
tpf_om = tpf_oma @ tpf_omb

tpf_tma = tpf_oma.to(torch.float16)
tpf_tmb = tpf_omb.to(torch.float16)
tpf_tm = tpf_tma @ tpf_tmb

tpf_3ma = tpf_oma.to(torch.bfloat16)
tpf_3mb = tpf_omb.to(torch.bfloat16)
tpf_3m = tpf_3ma @ tpf_3mb

print(f"\n2.5 The comparisom of matrix multiply with difference data type (While factor 100)" \
      f"\n\t Data type {tpf_om.dtype} have the first (0, 0) result is {tpf_om[0][0]:.2f} " \
      f"\n\t Data type {tpf_tm.dtype} have the first (0, 0) result is {tpf_tm[0][0]:.2f} " \
      f"\n\t Data type {tpf_3m.dtype} have the first (0, 0) result is {tpf_3m[0][0]:.2f} ")

# torch.cuda.memory._dump_snapshot("2_5.pickle")
# torch.cuda.memory._record_memory_history(enabled=None)

#------------------

#3.1 create m*n and n*k with float32, move to gpu and @. Record execution time on @ only.

Tpod = three_point_one_data = {"m": 3000,
                               "n": 1250,
                               "k": 2230,
                               "core": "GPU"}

Tpod_core = core_selected(Tpod["core"])
Tpoma = three_point_one_matrix_a = torch.rand(Tpod["m"], Tpod["n"], dtype=torch.float32, device=Tpod_core)
Tpomb = three_point_one_matrix_b = torch.rand(Tpod["n"], Tpod["k"], dtype=torch.float32, device=Tpod_core)

begin_time = time.time()
Tpom = Tpoma @ Tpomb
Tpom_time_usage = time.time() - begin_time


#3.2 do 3.1 again but do float16 and bfloat16, compare these 3 execution time.

Tptma_f16 = Tpoma.to(torch.float16)
Tptmb_f16 = Tpomb.to(torch.float16)

begin_time = time.time()
Tptm_F16 = Tptma_f16 @ Tptmb_f16
Tptm_F16_time_usage = time.time() - begin_time

Tptma_b16 = Tpoma.to(torch.bfloat16)
Tptmb_b16 = Tpomb.to(torch.bfloat16)

begin_time = time.time()
Tptm_B16 = Tptma_b16 @ Tptmb_b16
Tptm_B16_time_usage = time.time() - begin_time

print(f"\n[On GPU]\t Float32 {Tpom_time_usage}, Float16 {Tptm_F16_time_usage}, BFloat16 {Tptm_B16_time_usage}")

#3.3 do 3.1-3.2 but do on the cpu only, compare on cpu with on gpu.

Tp3ma = Tpoma.to("cpu")
Tp3mb = Tpomb.to("cpu")

begin_time = time.time()
Tp3m = Tp3ma @ Tp3mb
Tp3m_time_usage = time.time() - begin_time

#--------

Tp3ma_f16 = Tptma_f16.to("cpu")
Tp3mb_f16 = Tptmb_f16.to("cpu")

begin_time = time.time()
Tp3m_F16 = Tp3ma_f16 @ Tp3mb_f16
Tp3m_F16_time_usage = time.time() - begin_time

#----------

Tp3ma_b16 = Tptma_b16.to("cpu")
Tp3mb_b16 = Tptmb_b16.to("cpu")

begin_time = time.time()
Tp3m_B16 = Tp3ma_b16 @ Tp3mb_b16
Tp3m_B16_time_usage = time.time() - begin_time

print(f"[On CPU]\t Float32 {Tp3m_time_usage}, Float16 {Tp3m_F16_time_usage}, BFloat16 {Tp3m_B16_time_usage}")
