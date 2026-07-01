# Cross-Architecture Benchmarking Plan

Our goal for the next phase is to reproduce the project's core benchmarks (Figures 5, 6, and the component timesplits) across three architectures: **NVIDIA CUDA, NVIDIA HIP, and AMD HIP**.

## Phase 1: Infrastructure & Automation
Before we start collecting data, we need to ensure that benchmarking on different hardware is fully automated and easy to execute.

-> It woudld be useful here to use Felix code as a guide or to adopt this timing method in its entirety, so that the data is comparable to his project.

* **(Joint) Task (Tobias; Masha could give feedback):** Briefly define a standardized output format for our C++ hardware timers (e.g., a simple CSV string outputting architecture, solver, and component times) to ensure our data merges seamlessly. (This could be omitted if we use Felix code)
* **Tobias:** Write a central automation script that handles the compilation for the specified architecture, iterates through all required parameter configurations (knot points, control rates), and reliably logs the outputs.

## Phase 2: Data Collection & Component Benchmarking
To efficiently reproduce the graphs, we will divide the benchmarking tasks along our established code responsibilities so nobody has to dig into unfamiliar code.

### Masha (Solver & Low-Level Math)
* **Figure 5 (CDF of Linear System Solve Time):** Benchmark GBD-PCG solve times at different tolerances ($10^{-4}, 5\cdot10^{-5}, 10^{-5}$) to verify if the AMD architecture maintains the solver's original convergence characteristics.
* **Component Timesplits:** Instrument the code to measure and validate timings for the **Shur** and **Linsys** components across all three architectures.

### Tobias (High-Level Pipeline & QDLDL)
* **Figure 6 (Control Rate vs. Knot Points):** Run the full automated NMPC pipeline (using both QDLDL and GBD-PCG) across the grid ($N \in \{32, 64, 128, 256, 512\}$) at $250\text{Hz}, 500\text{Hz}, \text{and } 1\text{kHz}$ to generate the heatmap data.
* **Component Timesplits:** Instrument the code to measure and validate timings for the **KTT**, **DZ**, and **Line Search** components across all three architectures.

## Phase 3: Evaluation & Architecture Comparison
Once the automated data collection is complete, we will jointly evaluate the results:
* **CUDA vs. NVIDIA HIP:** Verify that our HIP port introduces no significant performance overhead on native NVIDIA hardware (should serve as our baseline validation).
* **NVIDIA HIP vs. AMD HIP:** Evaluate the performance differences on the AMD architecture. We need to pay special attention to whether the known numerical drift on AMD artificially inflates the required iteration counts, thereby affecting the real-time feasibility mapped out in Figure 6.
