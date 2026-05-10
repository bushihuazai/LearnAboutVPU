# 矢量计算深度研究报告

> 计算机体系架构中的矢量计算：理论、范式与业界先进架构
> Created by xxClaw

---

## 📚 目录结构

```
vector/
├── 01_theory/                  # 理论基础
│   └── 01_vector_computing_theory.md    # 矢量计算基础理论
├── 02_paradigms/              # 计算范式
│   ├── 01_simd_paradigm.md        # SIMD 计算范式
│   └── 02_simt_paradigm.md        # SIMT 计算范式
├── 03_architectures/          # 业界先进架构
│   ├── 01_x86_avx512.md          # x86 AVX-512 架构
│   ├── 02_arm_sve_sme.md         # ARM SVE/SME 架构
│   ├── 03_riscv_rvv.md           # RISC-V RVV 1.0 架构
│   └── 04_intel_amx.md           # Intel AMX 架构
├── 04_gpu_ai/                 # GPU 与 AI 加速器
│   ├── 01_gpu_computing.md        # GPU AI 计算架构
│   └── 02_tpu_systolic_array.md  # TPU 脉动阵列架构
├── 05_programming/            # 编程与工具
│   └── 01_auto_vectorization.md    # 自动向量化编程
└── 06_references/             # 参考资源
    └── README.md                   # 经典书籍与文献索引
```

---

## 🗂️ 文件说明

### 01_theory/ — 理论基础

| 文件 | 内容概要 |
|------|---------|
| `01_vector_computing_theory.md` | Flynn 分类法、矢量处理器发展史（Cray-1 → 现代 SIMD）、矢量长度模型（固定/实现定义/运行时动态）、核心概念术语表 |

### 02_paradigms/ — 计算范式

| 文件 | 内容概要 |
|------|---------|
| `01_simd_paradigm.md` | SIMD 核心概念、Subword/Vector SIMD/SIMT/VT 四种模型、主流 SIMD ISA（x86/ARM/RISC-V）、微架构实现、性能优化关键技术 |
| `02_simt_paradigm.md` | SIMT 核心概念（与 SIMD 对比）、GPU Warp 执行模型、CUDA 编程模型映射、性能优化（Tiling/Pipeline/Latency Hiding） |

### 03_architectures/ — 业界先进架构

| 文件 | 内容概要 |
|------|---------|
| `01_x86_avx512.md` | AVX-512 指令集家族（F/CD/BW/DQ/VNNI/BF16 等）、微架构实现、EVEX 编码、掩码寄存器、与 SVE/RVV 对比 |
| `02_arm_sve_sme.md` | SVE 可扩展矢量（VLA）、SVME2 DSP 扩展、SME 矩阵扩展（外积方式）、ZA Tile 寄存器、与 AMX 对比 |
| `03_riscv_rvv.md` | RVV 1.0 设计哲学（Write Once Run Everywhere）、vsetvli 动态配置、LMUL 机制、寄存器模型、与 AVX-512/SVE 对比 |
| `04_intel_amx.md` | AMX TMUL 矩阵引擎、Tile 寄存器模型、调色板配置、TDPBSSD/TDPBF16PS 指令、与 SME/Tensor Core 对比 |

### 04_gpu_ai/ — GPU 与 AI 加速器

| 文件 | 内容概要 |
|------|---------|
| `01_gpu_computing.md` | GPU 架构演进（Tesla → Blackwell）、Tensor Core 架构详解、Hopper Transformer Engine、GPU 内存层次、CUDA 编程模型、多 GPU NVLink |
| `02_tpu_systolic_array.md` | 脉动阵列原理、TPU v1~v5 架构演进、PE 内部结构、TPU 软件栈（Jax/XLA）、脉动阵列 vs Tensor Core vs 向量处理器 |

### 05_programming/ — 编程与工具

| 文件 | 内容概要 |
|------|---------|
| `01_auto_vectorization.md` | LLVM 循环向量化器（Loop Vectorizer）、SLP 向量化器、GCC 向量化、诊断与调试（Rpass/fopt-info）、写出可向量化代码的实践指南 |

---

## 🏛️ 经典权威书籍

### 核心参考书

| 书名 | 作者 | 说明 | 获取途径 |
|------|------|------|---------|
| **Computer Architecture: A Quantitative Approach (6th Ed.)** | Hennessy & Patterson | 矢量计算章节（Chapter 4）权威教材 | [Elsevier](https://shop.elsevier.com/books/computer-architecture/hennessy/978-0-443-15406-5) |
| **SIMD Programming Manual for x86 and ARM** | Daniel Lemire | x86/ARM SIMD 实战手册 | [Lemire's Blog](https://lemire.me/blog/) |
| **RISC-V Instruction Set Manual (Unprivileged)** | RISC-V International | RVV 官方规范 | [riscv.org](https://riscv.org/technical/specifications/) |
| **CUDA C++ Programming Guide** | NVIDIA | GPU SIMT/Tensor Core 编程权威指南 | [NVIDIA Docs](https://docs.nvidia.com/cuda/) |
| **ARM SVE Programmer's Guide** | ARM Limited | SVE/SME 编程权威指南 | [ARM Developer](https://developer.arm.com/documentation/) |
| **The Synthesis Approach to Digital System Design** | Daniel D. Gajski | 早期向量处理理论 | 图书馆/二手 |
| **Systolic Arrays: A Survey** | H. T. Kung (1982) | 脉动阵列理论奠基论文 | [IEEE Computer](https://ieeexplore.ieee.org/) |

### 在线开放资源

| 资源 | 说明 | 链接 |
|------|------|------|
| **RISC-V RVV 1.0 Spec** | 官方 PDF（免费） | [riscv.org/specifications](https://riscv.org/technical/specifications/) |
| **LLVM Vectorizers Doc** | LLVM 自动向量化文档（免费） | [llvm.org/docs](https://llvm.org/docs/Vectorizers.html) |
| **ARM Learning Paths** | ARM SVE 迁移指南（免费） | [learn.arm.com](https://learn.arm.com/) |
| **NVIDIA Technical Blog** | Tensor Core / Hopper 架构（免费） | [nvidia.com/blog](https://developer.nvidia.com/blog/) |
| **Computer Architecture Course (CMU 15-418)** | Kayvon Fatahalian 课程讲义 | [cs.cmu.edu/~418](https://www.cs.cmu.edu/afs/cs/academic/class/15418-s23/www/) |

---

## 📊 架构对比速查表

### 三大矢量 ISA 对比

| 维度 | x86 AVX-512 | ARM SVE/SVE2 | RISC-V RVV 1.0 |
|------|-------------|---------------|---------------------|
| **向量宽度** | 固定 512-bit | 实现定义（128~2048） | 运行时动态（vsetvli） |
| **可移植性** | ❌ 需重编译 | ✅ 二进制兼容 | ✅✅ 跨实现兼容 |
| **LMUL 支持** | ❌ | ❌ | ✅（核心创新） |
| **生态成熟度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **嵌入式适用** | ❌ | ⚠️ | ✅ |

### AI 矩阵加速技术对比

| 维度 | Intel AMX | ARM SME | NVIDIA Tensor Core | Google TPU |
|------|-----------|---------|---------------------|------------|
| **计算范式** | 点积 GEMM | 外积累加 | 混合式 | 脉动阵列 |
| **数据类型** | INT8, BF16 | INT8/16/32, FP16, BF16 | FP16→FP8→FP4 | INT8→BF16 |
| **首现** | 2021 | 2023 | 2017 | 2016 |
| **编程复杂度** | 中 | 高 | 低（框架自动） | 低（Jax/XLA） |

---

## 🛠️ 工具与技能

### 编译器向量化工具

| 工具 | 用途 | 参考文件 |
|------|------|---------|
| **Clang/LLVM** | 循环向量化 + SLP 向量化 | `05_programming/01_auto_vectorization.md` |
| **GCC** | `-ftree-vectorize`，`-fopt-info-vec` | 同上 |
| **ARM Compiler** | SVE 自动向量化 | `03_architectures/02_arm_sve_sme.md` |
| **RISC-V GNU Toolchain** | RVV 1.0 向量化 | `03_architectures/03_riscv_rvv.md` |

### Intrinsic 编程参考

| ISA | Intrinsic 头文件 | 示例 |
|-----|-----------------|------|
| x86 AVX-512 | `<immintrin.h>` | `_mm512_add_ps()` |
| ARM SVE | `<arm_sve.h>` | `svadd_f32_z()` |
| RISC-V RVV | `<riscv_vector.h>` | `vfadd_vv_f32m1()` |
| Intel AMX | `<immintrin.h>` | `_tile_dpbssd()` |

---

## 📅 更新记录

| 日期 | 内容 |
|------|------|
| 2026-05-10 | 初始版本：完成全部 9 个技术文档编写，包含 PlantUML 架构图示 |

---

## 📞 使用建议

1. **按顺序阅读**：先读 `01_theory/` → `02_paradigms/` → `03_architectures/`
2. **PlantUML 图示**：所有 `.md` 文件均含 PlantUML 代码，可使用 VS Code PlantUML 插件渲染
3. **对比学习**：`03_architectures/` 的四个文件互相对照阅读，架构差异一目了然
4. **实践优先**：`05_programming/` 提供可直接编译的 Intrinsic 代码示例

---

*本报告由 WorkBuddy AI 深度研究生成，内容综合自 Hennessy & Patterson、ARM/NVIDIA/Intel 官方文档、以及开源社区技术文章。*
