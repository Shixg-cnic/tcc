import numpy as np
import scipy.sparse as sp


dataset_dir = "/cnic/work/shixg/GNN/pro_dataset/arxiv"

cc_output_file = (
    "/cnic/work/shixg/GNN/"
    "tc_cc_kernel/test/data/"
    "arxiv_tcf_cc_output.bin"
)

feature_dim = 128


# --------------------------------------------------
# 1. 读取 CSR
# --------------------------------------------------

indptr = np.fromfile(
    dataset_dir + "/arxiv_indptr.bin",
    dtype=np.uint64,
)

indices = np.fromfile(
    dataset_dir + "/arxiv_indices.bin",
    dtype=np.uint64,
)

num_rows = len(indptr) - 1

print("rows:", num_rows)
print("nnz:", len(indices))


# --------------------------------------------------
# 2. 读取 X
# --------------------------------------------------

X = np.fromfile(
    dataset_dir + "/arxiv_features.bin",
    dtype=np.float32,
)

X = X.reshape(
    num_rows,
    feature_dim,
)

print("X shape:", X.shape)


# --------------------------------------------------
# 3. 构造 A
# --------------------------------------------------

values = np.ones(
    len(indices),
    dtype=np.float32,
)

A = sp.csr_matrix(
    (
        values,
        indices,
        indptr,
    ),
    shape=(
        num_rows,
        num_rows,
    ),
)


# --------------------------------------------------
# 4. CPU reference
# --------------------------------------------------

print("computing CPU reference...")

reference = A @ X


# --------------------------------------------------
# 5. 读取 CUDA Core 结果
# --------------------------------------------------

cc = np.fromfile(
    cc_output_file,
    dtype=np.float32,
)

expected_elements = (
    num_rows * feature_dim
)

print(
    "CC output elements:",
    cc.size,
)

print(
    "expected elements:",
    expected_elements,
)

assert (
    cc.size == expected_elements
)

cc = cc.reshape(
    num_rows,
    feature_dim,
)


# --------------------------------------------------
# 6. 比较
# --------------------------------------------------

diff = np.abs(
    reference - cc
)

print()
print(
    "max abs error:",
    diff.max(),
)

print(
    "mean abs error:",
    diff.mean(),
)

print(
    "p99 abs error:",
    np.percentile(
        diff,
        99,
    ),
)

print(
    "allclose 1e-5:",
    np.allclose(
        reference,
        cc,
        rtol=1e-5,
        atol=1e-5,
    ),
)

print(
    "allclose 1e-4:",
    np.allclose(
        reference,
        cc,
        rtol=1e-4,
        atol=1e-4,
    ),
)


# --------------------------------------------------
# 7. 找误差最大的位置
# --------------------------------------------------

max_index = np.unravel_index(
    np.argmax(diff),
    diff.shape,
)

print()
print(
    "max error position:",
    max_index,
)

print(
    "reference:",
    reference[max_index],
)

print(
    "CC:",
    cc[max_index],
)

print(
    "difference:",
    diff[max_index],
)