import os
import numpy as np
import scipy.sparse as sp

# ========================== 配置区：以后主要改这里 ==========================
DATASET_DIR = "/cnic/work/shixg/GNN/pro_dataset/arxiv"
OUTPUT_FILE = "/cnic/work/shixg/GNN/tc_cc_kernel/test/data/arxiv_tcf_hybrid_output.bin"
FEATURE_DIM = 128
INDPTR_FILE = "arxiv_indptr.bin"
INDICES_FILE = "arxiv_indices.bin"
FEATURE_FILE = "arxiv_features.bin"
# ========================================================================

indptr_path = os.path.join(DATASET_DIR, INDPTR_FILE)
indices_path = os.path.join(DATASET_DIR, INDICES_FILE)
feature_path = os.path.join(DATASET_DIR, FEATURE_FILE)

indptr = np.fromfile(indptr_path, dtype=np.uint64)
indices = np.fromfile(indices_path, dtype=np.uint64)
num_rows = len(indptr) - 1
nnz = len(indices)

print("rows:", num_rows)
print("nnz:", nnz)

X = np.fromfile(feature_path, dtype=np.float32)
expected_x_elements = num_rows * FEATURE_DIM
assert X.size == expected_x_elements, f"X size error: got {X.size}, expected {expected_x_elements}"
X = X.reshape(num_rows, FEATURE_DIM)
print("X shape:", X.shape)

values = np.ones(nnz, dtype=np.float32)
A = sp.csr_matrix((values, indices, indptr), shape=(num_rows, num_rows))

print("computing CPU reference...")
reference = A @ X

assert os.path.exists(OUTPUT_FILE), f"output file not found: {OUTPUT_FILE}"
hybrid = np.fromfile(OUTPUT_FILE, dtype=np.float32)
expected_output_elements = num_rows * FEATURE_DIM

print("hybrid output elements:", hybrid.size)
print("expected elements:", expected_output_elements)

assert hybrid.size == expected_output_elements, f"output size error: got {hybrid.size}, expected {expected_output_elements}"
hybrid = hybrid.reshape(num_rows, FEATURE_DIM)

diff = np.abs(reference - hybrid)

print()
print("max abs error:", diff.max())
print("mean abs error:", diff.mean())
print("p99 abs error:", np.percentile(diff, 99))
print("allclose 1e-5:", np.allclose(reference, hybrid, rtol=1e-5, atol=1e-5))
print("allclose 1e-4:", np.allclose(reference, hybrid, rtol=1e-4, atol=1e-4))
print("allclose 1e-3:", np.allclose(reference, hybrid, rtol=1e-3, atol=1e-3))
print("allclose 1e-2:", np.allclose(reference, hybrid, rtol=1e-2, atol=1e-2))

max_index = np.unravel_index(np.argmax(diff), diff.shape)

print()
print("max error position:", max_index)
print("reference:", reference[max_index])
print("hybrid:", hybrid[max_index])
print("difference:", diff[max_index])