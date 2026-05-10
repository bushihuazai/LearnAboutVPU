# Google TPU 与脉动阵列架构

> Google Tensor Processing Unit — 脉动阵列驱动的 AI 加速器

---

## 目录

1. [概述](#概述)
2. [脉动阵列（Systolic Array）原理](#脉动阵列systolic-array原理)
3. [TPU v1 ～ v5 架构演进](#tpu-v1--v5-架构演进)
4. [TPU 微架构详解](#tpu-微架构详解)
5. [TPU 编程模型](#tpu-编程模型)
6. [脉动阵列 vs 向量处理器 vs Tensor Core](#脉动阵列-vs-向量处理器-vs-tensor-core)
7. [PlantUML 架构图示](#plantuml-架构图示)
8. [参考文献](#参考文献)

---

## 概述

**TPU（Tensor Processing Unit）** 是 Google 为加速深度神经网络而设计的**专用 ASIC 芯片**，首次发布于 2016 年（TPU v1）。

### TPU 核心设计理念

```plantuml
@startuml
skinparam backgroundColor #FAFAFA

title TPU 设计哲学

note "核心矛盾：\nAI 训练/推理需要巨量矩阵乘法\n通用 CPU/GPU 并非最优" as c1

note "解决方案：\n1. 专用脉动阵列（Systolic Array）\n2. 极大减少内存访问（数据在阵列内流动）\n3. 简化控制逻辑（领域专用）" as c2

c1 -> c2 : Google TPU 方案

@enduml
```

### TPU vs GPU vs CPU 对比

| 维度 | CPU (AVX-512/AMX) | GPU (Tensor Core) | TPU (Systolic Array) |
|------|-------------------|---------------------|--------------------------|
| **架构类型** | 通用 + 矢量扩展 | 通用 + 图形 + AI 加速 | 领域专用（DSA） |
| **矩阵计算方式** | 向量累加 / Tile 乘加 | Tensor Core 投影乘加 | 脉动阵列（数据流） |
| **能效** | 低 | 中 | **极高** |
| **峰值吞吐** | 低 | 高 | 极高（同功耗下） |
| **编程灵活性** | 高 | 中 | 低（主要为矩阵乘法） |
| **首款产品** | — | 2017 (Volta) | 2016 (TPU v1) |

---

## 脉动阵列（Systolic Array）原理

### 核心概念

**脉动阵列（Systolic Array）**：一种由多个相同处理单元（PE，Processing Element）组成的网状结构，数据在时钟驱动下**像脉搏一样在阵列中流动（Systole = 脉搏）**。

```
传统矩阵乘法（CPU/GPU）：
  从内存读取 A[i,k]，B[k,j]
  计算 A[i,k] × B[k,j]
  写回 C[i,j]
  → 大量内存访问！

脉动阵列（TPU）：
  数据从阵列边界"流入"
  每个 PE 完成部分乘加后，数据"脉动"到相邻 PE
  最终结果在阵列另一端"流出"
  → 数据重用极高，内存访问极少！
```

### 脉动阵列结构

```plantuml
@startuml
skinparam backgroundColor #FAFAFA

title 2D 脉动阵列结构（以 4×4 为例）

node "PE[0,0]" as pe00
node "PE[0,1]" as pe01
node "PE[0,2]" as pe02
node "PE[0,3]" as pe03

node "PE[1,0]" as pe10
node "PE[1,1]" as pe11
node "PE[1,2]" as pe12
node "PE[1,3]" as pe13

node "PE[2,0]" as pe20
node "PE[2,1]" as pe21
node "PE[2,2]" as pe22
node "PE[2,3]" as pe23

node "PE[3,0]" as pe30
node "PE[3,1]" as pe31
node "PE[3,2]" as pe32
node "PE[3,3]" as pe33

pe00 -> pe01 : 右移
pe01 -> pe02 : 右移
pe02 -> pe03 : 右移

pe10 -> pe11 : 右移
pe11 -> pe12 : 右移
pe12 -> pe13 : 右移

pe20 -> pe21 : 右移
pe21 -> pe22 : 右移
pe22 -> pe23 : 右移

pe30 -> pe31 : 右移
pe31 -> pe32 : 右移
pe32 -> pe33 : 右移

pe00 <- pe10 : 下移
pe10 <- pe20 : 下移
pe20 <- pe30 : 下移

@enduml
```

### PE（Processing Element）内部结构

```
每个 PE 包含：
  1. 乘法器（Multiplier）
  2. 加法器（Accumulator）
  3. 输入寄存器（存储 A[i,k]）
  4. 权重寄存器（存储 B[k,j]）
  5. 部分和寄存器（存储 C[i,j] 的部分和）

数据流：
  时钟 1:  A[0,0] 和 B[0,0] 加载到 PE[0,0]
  时钟 2:  A[0,0]→右移，B[0,0]→下移；A[0,1] 和 B[1,0] 加载
  ...
  每个 PE 不断执行：acc += A_in × B_in
```

---

## TPU v1 ～ v5 架构演进

```plantuml
@startuml
skinparam backgroundColor #FAFAFA

title Google TPU 架构演进时间线

note
  2016: TPU v1     → 8-bit MAC，推理专用，23 TOPS
  2017: TPU v2     → 训练+推理，HBM，45 TOPS (INT8) / 180 TFLOPS (BF16)
  2018: TPU v3     → 提升 HBM 带宽，420 TFLOPS (BF16)
  2021: TPU v4     → 新互连拓扑，1.1 PFLOPS (BF16)
  2023: TPU v5e    → 推理优化，性价比提升
  2024: TPU v5p    → 训练优化，2.25 PFLOPS (BF16)
  2025: TPU v6（？）→ 预计 FP8 支持
end note

@enduml
```

### 各代 TPU 规格对比

| 型号 | 年份 | 峰值 BF16 | 片内 HBM | 主要用途 |
|------|------|-----------|----------|---------|
| **TPU v1** | 2016 | —（仅 INT8） | 8 GB DDR3 | 推理 |
| **TPU v2** | 2017 | 180 TFLOPS | 16 GB HBM | 训练 + 推理 |
| **TPU v3** | 2018 | 420 TFLOPS | 32 GB HBM | 训练 + 推理 |
| **TPU v4** | 2021 | 1.1 PFLOPS | 32 GB HBM | 训练（大规模） |
| **TPU v5e** | 2023 | 197 TFLOPS | 16 GB HBM | 推理（性价比） |
| **TPU v5p** | 2024 | 2.25 PFLOPS | 95 GB HBM | 训练（大规模） |

---

## TPU 微架构详解

### TPU v1 微架构（经典脉动阵列）

```plantuml
@startuml
skinparam backgroundColor #FAFAFA

title TPU v1 微架构框图

package "Host (CPU)" {
  component "PCIe 3.0 x16" as pcie
}

package "TPU v1 Chip" {
  component "Host Interface\n(PCIe)" as hi
  component "Weight FIFO\n(权重队列)" as wfifo
  component "Weight Memory\n(8 MB)" as wm
  component "Unified Buffer\n(24 MB)" as ub
  
  component "Systolic Array\n(256×256 PE)\n65,536 个 MAC/cycle" as sa
  
  component "Accumulators\n(4 MB, 24×256)" as acc
  component "Activation\n(非线性函数)" as act
  
  component "DDR3 Controller\n(8 GB)" as ddr
}

hi -> wfifo : 权重数据
wfifo -> wm : 存储权重
wm -> sa : 权重流向脉动阵列

hi -> ub : 激活数据（通过 PCIe）
ub -> sa : 激活数据流入脉动阵列

sa -> acc : 部分和累加
acc -> ub : 写回（用于多层网络）
ub -> act : 激活函数
act -> ddr : 输出到 DDR3

ddr <- pcie : DMA 传输结果到 Host

note right of sa
  TPU v1 脉动阵列：
  256 × 256 = 65,536 个 PE
  每个 PE：8-bit MAC
  峰值：65,536 MAC/cycle
  @ 700 MHz → 23 TRILLION MAC/s = 92 TOPS（INT8）
end note

@enduml
```

### TPU v4/v5 脉动阵列改进

```
改进点（相比 v1）：
  1. 支持 BF16 格式（v2 开始）
  2. 脉动阵列尺寸更大（具体未公开，但更大）
  3. 支持稀疏性加速（Sparsity）
  4. 片内互连拓扑改进（光学互连）
  5. 软件栈成熟（Jax / PyTorch XLA）
```

---

## TPU 编程模型

### TPU 软件栈

```plantuml
@startuml
skinparam backgroundColor #FAFAFA

title TPU 软件栈架构

package "用户代码" {
  component "Jax\n(推荐)" as jax
  component "PyTorch\n(XLA 后端)" as torch
  component "TensorFlow\n(原生支持)" as tf
}

package "编译器层" {
  component "XLA\n(Accelerated Linear Algebra)" as xla
}

package "运行时层" {
  component "TPU Runtime\n(驱动 + 固件)" as rt
}

package "硬件层" {
  component "TPU Chip\n(Systolic Array)" as tpu
}

jax --> xla : HLO IR（高级优化 IR）
torch --> xla
tf --> xla

xla --> rt : 编译后的二进制
rt --> tpu : 启动 TPU 执行

note right of xla
  XLA 核心优化：
  1. 算子融合（Fusion）→ 减少内存访问
  2. 内存布局优化
  3. 自动生成脉动阵列配置
end note

@enduml
```

### Jax 代码示例

```python
import jax
import jax.numpy as jnp
from jax import random

# 在 TPU 上执行（Jax 自动检测并使用 TPU）
key = random.PRNGKey(0)

# 矩阵乘法（自动使用脉动阵列）
A = random.normal(key, (4096, 4096), dtype=jnp.bfloat16)
B = random.normal(key, (4096, 4096), dtype=jnp.bfloat16)

# Jax 自动通过 XLA 编译为 TPU 脉动阵列指令
C = jnp.dot(A, B)  # ← 在 TPU 脉动阵列上执行

print(C.shape)  # (4096, 4096)
```

---

## 脉动阵列 vs 向量处理器 vs Tensor Core

### 三种 AI 加速架构对比

```plantuml
@startuml
skinparam backgroundColor #FAFAFA

title 三种 AI 加速架构对比

object "向量处理器\n(Cray-1 / RVV)" as vec {
  计算范式 = 向量指令流
  数据流 = 从内存加载 → 向量寄存器 → 执行 → 写回
  适用 = HPC / 通用数据并行
  代表 = Cray-1, RISC-V RVV
}

object "Tensor Core\n(NVIDIA GPU)" as tc {
  计算范式 = Warp 级矩阵乘加
  数据流 = Register → Tensor Core → Accumulator
  适用 = AI 训练/推理 + HPC
  代表 = NVIDIA Volta+
}

object "脉动阵列\n(Google TPU)" as sa {
  计算范式 = 数据流驱动
  数据流 = 数据在 PE 间流动，极少访问内存
  适用 = AI 矩阵乘法（DSA）
  代表 = Google TPU v1-v5
}

@enduml
```

### 综合对比表

| 维度 | 向量处理器（RVV/SVE） | Tensor Core（GPU） | 脉动阵列（TPU） |
|------|---------------------|---------------------|---------------------|
| **指令集** | 矢量 ISA（vadd.vv 等） | Tensor Core 指令（WMMA） | 无传统 ISA（配置权重 + 激活） |
| **数据复用** | 中（向量寄存器复用） | 高（Register File 大） | **极高**（数据在 PE 间流动） |
| **控制复杂度** | 中 | 高 | **极低**（DSA，控制逻辑少） |
| **能效** | 中 | 高 | **最高** |
| **灵活性** | 高 | 中 | 低（主要为矩阵乘法） |
| **适用领域** | HPC + AI（小模型） | AI（大模型）+ HPC | AI（大模型训练/推理） |

---

## PlantUML 架构图示

### 脉动阵列矩阵乘法原理

```plantuml
@startuml
skinparam backgroundColor #FAFAFA

title 脉动阵列执行 C = A × B（简化 2×2 示例）

note "输入 A (2×2):\nA[0,0] A[0,1]\nA[1,0] A[1,1]" as a
note "输入 B (2×2):\nB[0,0] B[0,1]\nB[1,0] B[1,1]" as b

note "PE[0,0]: acc += A[0,0] × B[0,0]\n           + A[0,1] × B[1,0]" as pe00
note "PE[0,1]: acc += A[0,0] × B[0,1]\n           + A[0,1] × B[1,1]" as pe01
note "PE[1,0]: acc += A[1,0] × B[0,0]\n           + A[1,1] × B[1,0]" as pe10
note "PE[1,1]: acc += A[1,0] × B[0,1]\n           + A[1,1] × B[1,1]" as pe11

a -> pe00 : A 数据右移 + 下移
a -> pe01 : A 数据右移
a -> pe10 : A 数据下移
a -> pe11

b -> pe00 : B 数据下移 + 右移
b -> pe01 : B 数据下移
b -> pe10 : B 数据右移
b -> pe11

@enduml
```

### TPU 与 GPU 架构哲学对比

```plantuml
@startuml
skinparam backgroundColor #FAFAFA

title TPU（DSA）vs GPU（通用）架构哲学

package "GPU 哲学：通用 + 加速" {
  note
    设计目标：
    1. 图形渲染（首要）
    2. 通用计算（次要）
    3. AI 加速（后来增加）
    
    结果：
    - 控制逻辑复杂（GPU 编译器复杂）
    - 编程模型灵活（CUDA 通用）
    - 适用面广但能效不如 DSA
  end note
}

package "TPU 哲学：领域专用" {
  note
    设计目标：
    1. AI 矩阵乘法（首要）
    2. 极致能效（次要）
    3. 简化控制逻辑
    
    结果：
    - 控制逻辑极简（DSA）
    - 编程模型受限（需通过 XLA）
    - 同功耗下 AI 性能最高
  end note
}

@enduml
```

---

## 参考文献

1. Jouppi, N. P., et al. (2017). *In-Datacenter Performance Analysis of a Tensor Processing Unit*. ISCA 2017.
2. Jouppi, N. P., et al. (2023). *Ten Lessons From Three Generations Shaped Google's TPUv4i*. ISCA 2023.
3. Google LLC. *Cloud TPU Architecture Whitepapers*, Latest Versions.
4. "谷歌 AI 的'心脏'长什么样?读懂 TPU 就够了", 腾讯科技, 2026.
5. "谷歌第八代 TPU 双舰齐发", 知乎, 2026.
6. Kung, H. T. (1982). *Why Systolic Architectures?*. IEEE Computer.
7. Hennessy, J. L., & Patterson, D. A. (2019). *Computer Architecture: A Quantitative Approach (6th Ed.)*, Section 4.5 (Domain-Specific Architectures).
8. Google. *Jax Documentation*, https://jax.readthedocs.io/.
