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
