
This document contains search commands (`grep`) to analyze the `mpcgpu` project.

For each category, there are two commands:
1. **Details:** Creates a checklist with file paths, line numbers, and the exact code.
2. **Summary:** Creates a clean, alphabetized list of the unique commands used.

Run these commands in the mpcgpu repo folder, before create a folder called "grep_findings".

---

## 1. Standard CUDA Commands
**Goal:** Find basic memory and management calls.

* **Details (Checklist):**
  `grep -rnP --exclude-dir="grep_findings" 'cuda[A-Z][a-zA-Z0-9_]+' . > grep_findings/1_cuda_standard_details.txt`

* **Summary (Unique Names only):**
  `grep -rhoP --exclude-dir="grep_findings" 'cuda[A-Z][a-zA-Z0-9_]+' . | sort -u > grep_findings/1_cuda_standard_summary.txt`

---

## 2. GPU Hardware & Syntax
**Goal:** Find GPU-specific keywords, memory types, and thread indexes.

* **Details (Checklist):**
  `grep -rnP --exclude-dir="grep_findings" '(__global__|__shared__|__constant__|__device__|__syncthreads|atomic[A-Z][a-zA-Z0-9_]+|threadIdx|blockIdx|blockDim|gridDim)' . > grep_findings/2_gpu_syntax_details.txt`

* **Summary (Unique Names only):**
  `grep -rhoP --exclude-dir="grep_findings" '(__global__|__shared__|__constant__|__device__|__syncthreads|atomic[A-Z][a-zA-Z0-9_]+|threadIdx|blockIdx|blockDim|gridDim)' . | sort -u > grep_findings/2_gpu_syntax_summary.txt`

---

## 3. NVIDIA Math Libraries
**Goal:** Find external NVIDIA libraries (like cuBLAS or cuSOLVER).

* **Details (Checklist):**
  `grep -rnP --exclude-dir="grep_findings" '(cublas|cusolver|cusparse|cufft|curand|thrust|cub|cudnn|nvtx)[A-Za-z0-9_]*' . > grep_findings/3_nvidia_libs_details.txt`

* **Summary (Unique Names only):**
  `grep -rhoP --exclude-dir="grep_findings" '(cublas|cusolver|cusparse|cufft|curand|thrust|cub|cudnn|nvtx)[A-Za-z0-9_]*' . | sort -u > grep_findings/3_nvidia_libs_summary.txt`

---

## 4. Kernel Execution (Where does the GPU start?)
**Goal:** Find every line of code where the CPU tells the GPU to start calculating.

* **Details (Checklist):**
  `grep -rn --exclude-dir="grep_findings" "<<<" . > grep_findings/4_kernel_starts_details.txt`


---

## 5. Kernel Definitions (Where is the GPU code written?)
**Goal:** Find the exact files and lines where the custom GPU math functions are written.

* **Details (Checklist):**
  `grep -rnw --exclude-dir="grep_findings" . -e "__global__" > grep_findings/5_kernel_definitions_details.txt`
