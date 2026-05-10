# 经典权威书籍与资源索引

> 矢量计算与计算机体系架构 — 经典书籍、规范与在线资源

---

## 📚 经典权威书籍

### 核心教材

| 书名 | 作者 | 版本 | 获取途径 |
|------|------|------|---------|
| **Computer Architecture: A Quantitative Approach** | John L. Hennessy & David A. Patterson | 6th Ed. (2019) | [Archive.org 合法存档](https://archive.org/details/ComputerArchitectureAQuantitativeApproach5thEdition) · [Elsevier 出版社](https://shop.elsevier.com/books/computer-architecture/hennessy/978-0-443-15406-5) |
| **SIMD Programming Manual for Linux and Windows** | Paul Cockshott & John Renfrew | 2013 | [Archive.org 免费下载](https://archive.org/download/LinuxLibgen/969.SIMD%20Programming%20Manual%20for%20Linux%20and%20Windows.pdf) |
| **The RISC-V Reader: An Open Architecture Atlas** | David Patterson & Andrew Waterman | 2017 | [riscv.org](https://riscv.org/community/risc-v-reader/) · [免费在线阅读](https://www.riscvbook.com/) |
| **Programming Massively Parallel Processors** | David B. Kirk & Wen-mei W. Hwu | 4th Ed. (2022) | [Elsevier](https://shop.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-323-99140-7) · [NVIDIA Developer](https://developer.nvidia.com/blog/) |

---

## 📖 官方规范与手册（免费）

### RISC-V 系列

| 文档 | 来源 | 链接 |
|------|------|------|
| **RISC-V Vector Extension Spec v1.0** | RISC-V International | [riscv.org/specifications](https://riscv.org/technical/specifications/)（PDF 免费下载） |
| **RISC-V Unprivileged ISA Manual** | RISC-V International | [GitHub](https://github.com/riscv/riscv-isa-manual) |
| **RISC-V Parallel SIMD Programming Book** | smartcomputerlab | [GitHub](https://github.com/smartcomputerlab/RISC-V-Parallel-SIMD-and-MIMD-Programming-and-Processing-Book) |

### ARM 系列

| 文档 | 来源 | 链接 |
|------|------|------|
| **ARM Architecture Reference Manual (ARMv9)** | ARM Limited | [ARM Developer](https://developer.arm.com/documentation/)（免费注册下载） |
| **ARM SVE Programmer's Guide** | ARM Limited | [ARM Developer](https://developer.arm.com/documentation/101490/latest/) |
| **ARM SME Programmer's Guide** | ARM Limited | [ARM Developer](https://developer.arm.com/documentation/109246/latest/) |
| **Migrate SIMD code to Arm** | ARM Learning Paths | [learn.arm.com](https://learn.arm.com/learning-paths/cross-platform/vectorization-comparison/) |

### Intel 系列

| 文档 | 来源 | 链接 |
|------|------|------|
| **Intel 64/IA-32 Architecture SDM (Vol.1)** | Intel | [Intel Developer Zone](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)（免费下载） |
| **Intel AMX Overview** | Intel | [Intel CN](https://www.intel.cn/content/www/cn/zh/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html) |
| **Intel Intrinsics Guide** | Intel | [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics/32vol1/710/SYNCHRONIZE.html) |

### NVIDIA 系列

| 文档 | 来源 | 链接 |
|------|------|------|
| **CUDA C++ Programming Guide** | NVIDIA | [NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)（免费在线） |
| **NVIDIA Hopper Architecture Whitepaper** | NVIDIA | [NVIDIA Technical Brief](https://resources.nvidia.com/en-us-tensor-core) |
| **NVIDIA Blackwell Architecture Whitepaper** | NVIDIA | [NVIDIA Developer](https://developer.nvidia.com/blog/) |

---

## 🌐 在线开放课程与讲义

| 资源 | 机构 | 链接 |
|------|------|------|
| **15-418/618 Parallel Computer Architecture** | CMU (Kayvon Fatahalian) | [cs.cmu.edu/~418](https://www.cs.cmu.edu/afs/cs/academic/class/15418-s23/www/) |
| **6.172 Performance Engineering** | MIT (Saman Amarasinghe) | [MIT OpenCourseWare](https://ocw.mit.edu/) |
| **CS61C Great Ideas in Computer Architecture** | UC Berkeley | [cs61c.org](https://cs61c.org/) |
| **RISC-V Vector Extension Tutorial** | PLCT Lab | [riscv.org/community](https://riscv.org/events/) |

---

## 🔬 经典论文

| 论文 | 作者 | 年份 | 获取 |
|------|------|------|------|
| **Systolic Arrays: A Survey** | H. T. Kung | 1982 | [IEEE Computer](https://ieeexplore.ieee.org/) |
| **In-Datacenter Performance Analysis of a Tensor Processing Unit** | Jouppi et al. | 2017 | [ISCA 2017](https://arxiv.org/abs/1704.04760)（arXiv 免费） |
| **The Landscape of Parallel Computing Research: A View from Berkeley** | Asanović et al. | 2006 | [UCB/EECS-2006-183](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-183.html) |
| **Simplified Vector-Thread Architectures** | UC Berkeley | 2006 | [EECS Tech Report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/) |

---

## 💻 工具链与编译器文档

| 资源 | 说明 | 链接 |
|------|------|------|
| **LLVM Auto-Vectorization Doc** | 循环向量化 + SLP 向量化 | [llvm.org/docs/Vectorizers](https://llvm.org/docs/Vectorizers.html) |
| **LLVM RISC-V Vector Extension** | RVV 1.0 LLVM 实现 | [llvm.org/docs/RISCV/RISCVVectorExtension](https://llvm.org/docs/RISCV/RISCVVectorExtension.html) |
| **GCC Vectorization** | GCC 自动向量化 | [gcc.gnu.org/projects/tree-ssa/vectorization.html](https://gcc.gnu.org/projects/tree-ssa/vectorization.html) |
| **RISC-V GNU Toolchain** | RVV 1.0 工具链 | [github.com/riscv-collab/riscv-gnu-toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain) |

---

## 📝 使用说明

### 如何获取这些资源

1. **开源/免费资源**：直接点击表中的链接即可免费获取（arXiv、官方文档、GitHub、Archive.org）
2. **出版社资源**：Elsevier / Springer 等出版社的书籍建议通过图书馆借阅或购买正版
3. ** Archive.org**：部分经典书籍有合法存档版本，可免费在线阅读
4. **NVIDIA/ARM/Intel 开发者网站**：注册开发者账号后可免费下载技术手册

### 推荐阅读顺序

```
第 1 步：Hennessy & Patterson
        → Chapter 4（Data-Level Parallelism）
        → 建立理论基础

第 2 步：SIMD Programming Manual (Cockshott)
        → 实战 SIMD 编程技巧
        → x86/ARM 双平台覆盖

第 3 步：官方规范（按需阅读）
        → 用 AVX-512？读 Intel SDM
        → 用 SVE？读 ARM SVE Guide
        → 用 RVV？读 RISC-V Vector Spec v1.0

第 4 步：GPU 计算
        → Kirk & Hwu《Programming Massively Parallel Processors》
        → NVIDIA CUDA Programming Guide

第 5 步：前沿论文
        → Jouppi ISCA 2017（TPU v1 架构论文）
        → Asanović Berkeley View（并行计算全景）
```

---

*更新日期：2026-05-10*
