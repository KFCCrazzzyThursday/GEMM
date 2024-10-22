# Device 0: NVIDIA GeForce RTX 4070 

- **Total number of SMs**: 46  
- **Maximum number of threads per SM**: 1536  
- **Maximum number of threads per block**: 1024  
- **Maximum size of each dimension of a block**: 1024 x 1024 x 64  
- **Maximum size of each dimension of a grid**: 2147483647 x 65535 x 65535  
- **Shared memory per block**: 49152 bytes  
- **Total global memory**: 11.9937 GB  
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
**PASSED!**

---

## Running tests for GEMM  
**Matrix dims MxNxK**: 128 x 128 x 128  
**Block dims**: 16 x 16  

| Algorithm                | Time       | TFLOPS    | P RATIO |
|--------------------------|------------|-----------|---------|
| CUBLAS                   | 0.000133   | 0.031637  | 100.00% |
| naiveGEMM                | 0.000011   | 0.390945  | 1235.73% |
| blockGEMM_sn8            | 0.000028   | 0.148554  | 469.56% |
| blockGEMM_sn16           | 0.000028   | 0.148030  | 467.90% |
| vec_GEMM_sn8             | 0.000020   | 0.207654  | 656.37% |
| vec_GEMM_sn16            | 0.000019   | 0.225163  | 711.72% |
| conflictFreeGEMM_sn8     | 0.000015   | 0.270746  | 855.80% |
| conflictFreeGEMM_sn16    | 0.000014   | 0.290416  | 917.97% |
| doubleBufferGEMM_sn8     | 0.000014   | 0.306755  | 969.62% |
| doubleBufferGEMM_sn16    | 0.000013   | 0.327639  | 1035.63% |

**PASSED!**  
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
**Matrix dims MxNxK**: 256 x 256 x 256  
**Block dims**: 16 x 16  

| Algorithm                | Time       | TFLOPS    | P RATIO |
|--------------------------|------------|-----------|---------|
| CUBLAS                   | 0.000008   | 4.380657  | 100.00% |
| naiveGEMM                | 0.000029   | 1.173691  | 26.79%  |
| blockGEMM_sn8            | 0.000046   | 0.725588  | 16.56%  |
| blockGEMM_sn16           | 0.000046   | 0.725262  | 16.56%  |
| vec_GEMM_sn8             | 0.000035   | 0.961917  | 21.96%  |
| vec_GEMM_sn16            | 0.000033   | 1.022348  | 23.34%  |
| conflictFreeGEMM_sn8     | 0.000029   | 1.141916  | 26.07%  |
| conflictFreeGEMM_sn16    | 0.000028   | 1.185897  | 27.07%  |
| doubleBufferGEMM_sn8     | 0.000024   | 1.404280  | 32.06%  |
| doubleBufferGEMM_sn16    | 0.000022   | 1.497998  | 34.20%  |

**PASSED!**  
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
**Matrix dims MxNxK**: 512 x 512 x 512  
**Block dims**: 16 x 16  

| Algorithm                | Time       | TFLOPS    | P RATIO |
|--------------------------|------------|-----------|---------|
| CUBLAS                   | 0.000028   | 9.487658  | 100.00% |
| naiveGEMM                | 0.000165   | 1.628943  | 17.17%  |
| blockGEMM_sn8            | 0.000079   | 3.393333  | 35.77%  |
| blockGEMM_sn16           | 0.000081   | 3.323333  | 35.03%  |
| vec_GEMM_sn8             | 0.000059   | 4.541440  | 47.87%  |
| vec_GEMM_sn16            | 0.000057   | 4.745721  | 50.02%  |
| conflictFreeGEMM_sn8     | 0.000052   | 5.175087  | 54.55%  |
| conflictFreeGEMM_sn16    | 0.000049   | 5.507335  | 58.05%  |
| doubleBufferGEMM_sn8     | 0.000044   | 6.169342  | 65.02%  |
| doubleBufferGEMM_sn16    | 0.000043   | 6.300402  | 66.41%  |

**PASSED!**  
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
**Matrix dims MxNxK**: 1024 x 1024 x 1024  
**Block dims**: 16 x 16  

| Algorithm                | Time       | TFLOPS    | P RATIO |
|--------------------------|------------|-----------|---------|
| CUBLAS                   | 0.000136   | 15.787256 | 100.00% |
| naiveGEMM                | 0.001130   | 1.899829  | 12.03%  |
| blockGEMM_sn8            | 0.000268   | 8.009450  | 50.73%  |
| blockGEMM_sn16           | 0.000269   | 7.972631  | 50.50%  |
| vec_GEMM_sn8             | 0.000210   | 10.223862 | 64.76%  |
| vec_GEMM_sn16            | 0.000201   | 10.696234 | 67.75%  |
| conflictFreeGEMM_sn8     | 0.000171   | 12.543760 | 79.45%  |
| conflictFreeGEMM_sn16    | 0.000161   | 13.323657 | 84.40%  |
| doubleBufferGEMM_sn8     | 0.000151   | 14.199013 | 89.94%  |
| doubleBufferGEMM_sn16    | 0.000145   | 14.813126 | 93.83%  |

**PASSED!**  
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
**Matrix dims MxNxK**: 2048 x 2048 x 2048  
**Block dims**: 16 x 16  

| Algorithm                | Time       | TFLOPS    | P RATIO |
|--------------------------|------------|-----------|---------|
| CUBLAS                   | 0.001023   | 16.792684 | 100.00% |
| naiveGEMM                | 0.009086   | 1.890818  | 11.26%  |
| blockGEMM_sn8            | 0.001568   | 10.956499 | 65.25%  |
| blockGEMM_sn16           | 0.001625   | 10.573978 | 62.97%  |
| vec_GEMM_sn8             | 0.001234   | 13.921037 | 82.90%  |
| vec_GEMM_sn16            | 0.001249   | 13.750360 | 81.88%  |
| conflictFreeGEMM_sn8     | 0.001097   | 15.658899 | 93.25%  |
| conflictFreeGEMM_sn16    | 0.001043   | 16.466607 | 98.06%  |
| doubleBufferGEMM_sn8     | 0.000980   | 17.538562 | 104.44% |
| doubleBufferGEMM_sn16    | 0.000943   | 18.214506 | 108.47% |

---
