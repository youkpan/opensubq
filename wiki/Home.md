# CRBSA Wiki

> Codebook-Routed Block-Sparse Attention 项目知识库

## 文档导航

### [Architecture.md](Architecture.md) — 架构设计文档
- 问题定义：长文本"不可能三角"
- 数学基础：O(N^2) → O(N) 的重构
- 架构总览：四步并行计算流
- 模块级微架构设计
- 三阶段训练法则
- 分布式系统架构 (Ulysses + P2P)
- 核心参数参考
- 评测基准 (RULER / MRCR / SWE-bench)
- 三条复现路线

### [Code-Design.md](Code-Design.md) — 代码设计文档
- 项目目录结构
- 配置系统 (dataclass + 调试开关)
- 核心模块代码设计：
  - BlockSummarizer — Block 摘要提取
  - CodebookRouter — 倒排索引 + Query 路由
  - CRBSAAttention — 完整 Layer 串联
  - Triton Kernel 骨架
- 调试系统：
  - 8 个调试开关及输出说明
  - DebugContext / DebugCollector / CRBSAProfiler
- 调试与优化路线 (Phase 0~4, 12 周计划)
  - Phase 0: 单模块正确性
  - Phase 1: Kernel 性能验证
  - Phase 2: 路由质量验证
  - Phase 3: 端到端训练
  - Phase 4: 分布式扩展
