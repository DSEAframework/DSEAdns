# Data Streaming for Explicit Algorithms - Direct Numerical Simulation (DSEAdns)

## Introduction

**DSEAdns** is a direct numerical simulation (DNS) implementation within the DSEA (Data Streaming for Explicit Algorithms) framework. It solves the three-dimensional incompressible Navier–Stokes equations using a **fourth-order central differencing scheme** for spatial discretization and a **third-order Runge-Kutta method** for temporal integration.

The simulation setup follows the **Taylor-Green vortex** example presented by Jacobs et al. [1], serving as a reference case for accuracy and performance evaluation.

## Getting Started

### Prerequisites

To build and run DSEAdns, the following tools are required:

- A C++ compiler with **C++17** support
- **CUDA 12.0** or newer
- An **MPI library** with MPI 2.0 support

Optional (for multi-rail communication across compute nodes):

- **UCX** (Unified Communication X) version **1.17** or newer (both headers and libraries)

### Installation and Execution

1. Adjust paths in the `Makefile` to match your system’s CUDA, MPI, and (optionally) UCX installation.
2. Configure the simulation case by editing `run_case.sh`:
   - Select a predefined problem size
   - Choose the kernel file to be used
   - Set the number of workers per GPU
   - Define the number of communication rails
   - Specify the number of supercycles to compute
3. Execute the `run_case.sh` script to start the simulation.

### Simulation Output

To enable simulation output, set the `DOUTPUT` macro to a positive integer. This value defines the interval (in supercycles) between each simulation output.

To change the output directory, modify the path in the `void DS::write_vtr` function in the selected kernel file.

## Kernel Files and Optimization Cycles

Each kernel file corresponds to a specific optimization cycle. Below is an overview of the implemented cycles:

- `dsea_kernel_cycle00_base.cu` Baseline implementation algorithm based on Jacobs et al. [1]. Uses fixed kernel configurations with a 1D block size of 128 threads.
- `dsea_kernel_cycle01_0_fusing.cu `Introduces **kernel fusion**, transitioning from task-specific kernels to data-centric kernels. Reduces the number of kernels to five and adds support for configurable thread block sizes.
- `dsea_kernel_cycle01_1_fusing+temporal_derivative.cu` Moves temporal derivative calculation out of `dns_Res_StageAdvance`, reducing overall memory traffic.
- `dsea_kernel_cycle02_rhoETp_optimized.cu` Splits `dns_rhoETpdxyz` into two kernels to improve memory coalescing. Uses **shared memory** and optimized **floating-point division** to reduce register pressure.
- `dsea_kernel_cycle03_Res_StageAdvance_optimized.cu` Further reduces memory traffic by distributing temporal derivative computation across other kernels, saving one global store/load per result.
- `dsea_kernel_cycle_04_0_shared.cu` Leverages shared memory in additional kernels to eliminate redundant memory access.
- `dsea_kernel_cycle_04_1_shared_schedule.cu` Leverages **instruction-level scheduling**, overlapping shared memory loads with computations to hide memory latency.
- `dsea_kernel_cycle04_2_scaling.cu` Final version with hardcoded, empirically optimized thread block configurations for maximum performance.

> The final implementation achieves a **3× speedup** compared to the baseline.

## References

[1] Christian T. Jacobs, Satya P. Jammy, Neil D. Sandham, *OpenSBLI: A framework for the automated derivation and parallel execution of finite difference solvers on a range of computer architectures*, Journal of Computational Science, Volume 18, 2017, Pages 12–23.
[https://doi.org/10.1016/j.jocs.2016.11.001](https://doi.org/10.1016/j.jocs.2016.11.001)

## Citation

If you use this work in academic or scientific contexts, please cite:

> M. Rose, S. Homes, L. Ramsperger, J. Gracia, C. Niethammer, and J. Vrabec.
> *Cyclic Data Streaming on GPUs for Short Range Stencils Applied to Molecular Dynamics*.
> HeteroPar 2025, accepted.
