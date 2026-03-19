# Device 0: NVIDIA A100-SXM4-40GB

- **Total number of SMs**: 108
- **Maximum number of threads per SM**: 2048
- **Maximum number of threads per block**: 1024
- **Maximum size of each dimension of a block**: 1024 x 1024 x 64
- **Maximum size of each dimension of a grid**: 2147483647 x 65535 x 65535
- **Shared memory per block**: 49152 bytes
- **Total global memory**: 39.4945 GB
- **Number of registers per SM**: 65536
- **Number of registers per block**: 65536
- **Maximum registers per thread**: 64

**PASSED!**
**PASSED!**
**PASSED!**
**PASSED!**
**PASSED!**
**PASSED!**
**PASSED!**
**PASSED!**

---

## Running tests for GEMM
**Matrix dims MxNxK**: 4096 x 4096 x 4096
**Block dims**: 16 x 16

| Algorithm             | Time     | TFLOPS     | P RATIO |
| :-------------------- | :------- | :--------- | :------ |
| CUBLAS_TC_TF32        | 0.001080 | 127.262204 | 100.00% |
| CUBLAS_SGEMM          | 0.007222 | 19.031456  | 14.95%  |
| blockGEMM_sn8         | 0.011239 | 12.228922  | 9.61%   |
| blockGEMM_sn16        | 0.011909 | 11.540643  | 9.07%   |
| vec_GEMM_sn8          | 0.009002 | 15.266831  | 12.00%  |
| vec_GEMM_sn16         | 0.008773 | 15.665869  | 12.31%  |
| conflictFreeGEMM_sn8  | 0.008622 | 15.941020  | 12.53%  |
| conflictFreeGEMM_sn16 | 0.008269 | 16.620166  | 13.06%  |
| doubleBufferGEMM_sn8  | 0.007761 | 17.709302  | 13.92%  |
| doubleBufferGEMM_sn16 | 0.007754 | 17.725333  | 13.93%  |