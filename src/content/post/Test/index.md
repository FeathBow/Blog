---
title: TEST
description: 
tags: 
publishDate: 2025-06-20 23:35
draft: false
share: true
---
 
# 强化学习入门：PPO 与 VeRL 框架在大型语言模型优化中的全面指南

## 1. 强化学习基础：通往 AI 的门户

强化学习（Reinforcement Learning, RL）是机器学习的一个分支，其核心在于智能体通过与环境的持续交互，在试错中学习如何做出决策，以最大化其累积奖励。与依赖标签数据的监督学习或发现无标签数据模式的无监督学习不同，强化学习专注于序列决策问题 1。

### 1.1 什么是强化学习？智能体 - 环境循环

强化学习过程涉及几个关键组成部分：

- 智能体（Agent）：这是学习者或决策者。智能体在环境中执行动作 1。
    
- 环境（Environment）：智能体所操作的世界或系统。环境根据智能体的动作提供反馈 1。
    
- 状态（State）或观察（Observation）：智能体当前所处的情境或条件。这是智能体感知到的信息 1。在视觉强化学习中，观察可以是图像或视频帧的原始像素输入，也可以是从这些输入中提取的特征 3。观察空间定义了所有可能的观察及其格式 2。例如，在 CartPole 环境中，状态包括小车位置、速度、杆子角度和角速度 1。
    
- 动作（Action）：智能体可以做出的移动或决策。动作会影响环境的下一个状态并产生奖励 1。动作空间可以是离散的（例如，CartPole 中的向左/向右移动 1）或连续的（例如，机器人中的电机扭矩 3）。
    
- 奖励（Reward）：环境提供的反馈，表示在给定状态下采取某个动作的期望程度。奖励可以是正向的或负向的（惩罚），并引导智能体实现其目标 1。智能体的目标是随着时间的推移最大化累积奖励 1。奖励函数为状态 - 动作对分配即时奖励 3。
    
- 策略（Policy）：智能体的策略，将状态映射到动作。它决定了智能体的行为 1。强化学习的目标是找到一个最大化累积奖励的最优策略 9。
    
- 价值函数（Value Function）：估计智能体从给定状态将获得的未来累积奖励 1。它帮助智能体评估特定状态或动作在长期内的好坏。

强化学习过程是一个持续的循环：智能体观察状态，根据其策略采取动作，从环境接收奖励和新状态，并相应地调整其行为以优化未来的奖励 1。

### 1.2 为什么强化学习是强大的优化范式

强化学习在解决传统技术无法处理的复杂问题方面表现出色，尤其是在涉及序列决策和动态环境的问题中 1。它允许通过实时交互进行持续学习，从而实现自适应行为 1。强化学习能够处理结果不确定或随时间变化的不确定环境，这使其在实际应用中非常有用 1。此外，模型能够从环境中不断学习并纠正训练过程中出现的错误 1。

强化学习的核心是奖励函数，它是智能体学习的唯一信号 1。对于习惯于明确指标的传统领域专家来说，如何将“良好”行为（例如，LLM 的有用性、无害性、真实性）转化为可量化的奖励信号，通常通过人类反馈（RLHF）或复杂的奖励模型实现，是至关重要且往往不那么直观的第一步 10。整个强化学习系统的成功都取决于这种抽象定义的质量。

深度强化学习（DRL）通常利用深度神经网络来处理原始像素等高维观察 4。虽然这些网络功能强大，但它们作为“函数逼近器” 9，直接从感知输入中学习复杂的输入 - 输出映射 9。这意味着智能体的内部“推理”过程可能不透明。对于传统领域的专业人士来说，这种缺乏明确规则或可解释步骤的特性可能是一个重大障碍。在优化 VeRL 用于 LLM 时，理解 LLM 为何生成特定响应（其“动作”）对于调试和改进至关重要。DRL 模型的黑箱性质意味着，尽管性能可能提高，但底层机制仍然难以解释，这可能使“VeRL 优化工作”复杂化，如果需要诊断特定行为问题而不仅仅是性能指标。这突出了通过奖励塑造和仔细设计观察空间来间接引导行为的重要性。

## 2. 解密近端策略优化 (PPO)

### 2.1 核心思想：策略梯度方法

策略梯度方法直接优化智能体的策略函数，旨在最大化预期的累积奖励 3。它们使用基于梯度的方法迭代地提高策略性能 3。传统的策略梯度方法每次数据采样只执行一次梯度更新 12。然而，由于策略更新可能过大，它们可能不稳定，从而导致策略进入性能不佳的区域，最终导致性能“崩溃”。信赖域策略优化（TRPO）是早期解决这种不稳定性的尝试，它通过确保策略更新不会过大，将新策略保持在旧策略的“近端”的“信赖域”内 12。尽管 TRPO 有效，但其实现复杂。

### 2.2 PPO 的目标函数：平衡性能与稳定性

近端策略优化（PPO）是一系列策略梯度方法，旨在平衡实现简易性、样本效率和性能 12。它已成为最成功的深度强化学习方法之一 13。PPO 通过与环境交互采样数据，并使用随机梯度上升优化“替代”目标函数 12。PPO 的关键创新在于其裁剪的替代目标函数 14。该目标函数允许在相同数据上进行多轮小批量更新，这比每次采样只执行一次更新的传统策略梯度方法更具样本效率 12。

PPO 的目标函数可以形式化表示为：

JPPO(θ)=Et​[min(rt​(θ)A^t​,clip(rt​(θ),1−ϵ,1+ϵ)A^t​)] 14

- 其中，rt​(θ)=πθold​​(at​∣st​)πθ​(at​∣st​)​ 是新策略与旧策略在给定动作和状态下的概率比。此比率也称为重要性采样比 14。
    
- A^t​ 是时间步 t 的优势估计，通常使用广义优势估计（GAE）计算 14。优势函数衡量了在给定状态下，某个动作相对于平均动作的优势程度 3。
    
- ϵ 是一个小的超参数，通常在 0.1 或 0.2 左右，定义了裁剪范围。

### 2.3 裁剪机制：防止破坏性更新

目标函数中的 clip 函数是 PPO 稳定性的核心。它将重要性采样比 rt​(θ) 限制在一个小的区间内，通常是 [1−ϵ,1+ϵ] 14。如果比率超出此区间，它将被“裁剪”，这意味着策略更新受到限制。这可以防止新策略与旧策略偏离过大，有助于避免可能导致性能下降的剧烈、不稳定的更新 14。目标函数取两个项的最小值：原始的优势加权目标，以及其裁剪版本。这确保了如果策略试图过度增加具有正优势的动作的概率，或者过度减少具有负优势的动作的概率，更新将被限制。

### 2.4 PPO 在现代强化学习中的主要优势

- 简洁性：与 TRPO 相比，PPO 实现起来简单得多，同时保留了 TRPO 的许多优点 12。
    
- 样本效率：它允许在同一批数据上进行多次梯度更新，从而提高了样本效率 12。
    
- 稳定性：裁剪机制提供了“近端”更新，防止剧烈的策略变化，增强了训练稳定性 14。
    
- 经验性能：PPO 在广泛的基准任务中表现出强大的经验性能，包括模拟机器人运动和 Atari 游戏 12。在多智能体设置中，它能够实现与离策略方法相当甚至更优的结果 15。
    
- 鲁棒性：一些研究表明，PPO 对超参数选择的鲁棒性优于传统的策略梯度方法 16。

PPO 虽然旨在保持策略更新的“近端” 12，但一些研究质疑它是否严格限制了概率比或强制执行了明确定义的信赖域约束 13。这意味着 PPO 仍然可能面临性能不稳定的风险 13。这表明 PPO 的成功并非完全来自其理论保证，也得益于其在实践中的鲁棒性和所采用的“优化技巧” 13。例如，“真正近端策略优化”（TR-PPO-RB）论文 13 试图通过引入回滚行为和基于理论证明的信赖域触发条件来解决这个问题。对于用户而言，这意味着 PPO 虽然是一个强大的基线，但它并非万能药。在复杂 LLM 环境中，如果 PPO 遇到困难，“优化工作”可能需要理解这些微妙的不稳定性，并考虑探索 TR-PPO-RB 或其他算法（VeRL 支持多种算法 17）。这也强调了即使对于鲁棒的算法，超参数调整仍然至关重要 15。

16 指出，通过简单地增加每次迭代的价值更新步骤，传统的策略梯度方法本身就可以达到与 PPO 相当甚至更好的性能。这直接挑战了 PPO 的卓越性能完全归因于其策略更新机制的普遍观点。这表明，

改进价值估计对传统策略梯度方法至关重要 16。更准确的价值估计（批评者的作用）提供了更精确的优势估计（

A^t​），从而导致更有效和稳定的策略更新，即使对于更简单的策略梯度方法也是如此。对于 LLM 优化而言，这意味着仅仅关注策略（LLM 的生成过程）可能是不够的。奖励模型（在 RLHF 中充当批评者 11）的质量及其价值估计至关重要。优化奖励模型或增加其训练步骤可以显著提高 LLM 的性能，即使核心强化学习算法（如 PPO）保持不变。这使得部分优化重点从 LLM 策略本身转移到奖励信号的生成。

## 3. 介绍 VeRL：火山引擎大型语言模型强化学习框架

### 3.1 VeRL 的愿景：灵活、高效、生产就绪的 LLM 强化学习

VeRL 框架（Volcano Engine Reinforcement Learning for LLMs）旨在成为一个灵活、高效且生产就绪的强化学习训练库，专门用于大型语言模型（LLMs）17。它是“HybridFlow: A Flexible and Efficient RLHF Framework”论文的开源实现 17。

### 3.2 核心设计原则与理念

VeRL 的设计强调以下几个关键原则 17：

- 灵活性和易用性：通过混合控制器编程模型实现，该模型允许灵活表示和高效执行复杂的后训练数据流。这种模型能够以最少的代码构建 GRPO 和 PPO 等强化学习数据流。
    
- 无缝集成：VeRL 解耦了计算和数据依赖，促进与现有 LLM 框架的无缝集成，如 FSDP、Megatron-LM、vLLM 和 SGLang。它还支持 HuggingFace 模型。
    
- 灵活的设备映射：该框架支持将模型放置在不同的 GPU 集合上，优化了跨不同集群规模的资源利用和可扩展性。
    
- 高性能：VeRL 旨在通过集成最先进的 LLM 训练和推理引擎实现最先进的吞吐量，从而实现最先进的强化学习吞吐量。
    
- 高效的 Actor 模型重分片：3D-HybridEngine 是实现效率的关键组件，通过消除内存冗余并显著减少训练和生成阶段之间转换时的通信开销。

### 3.3 主要组件与功能

VeRL 提供了一套全面的功能来支持 LLM 的强化学习训练 17：

- 训练引擎：支持 FSDP（推荐使用 FSDP2 以获得更好的吞吐量和内存使用，并支持 CPU 卸载以节省内存）和 Megatron-LM（用于训练大型专家混合模型，如 DeepSeek-671b 和 Qwen3-236b）。
    
- Rollout 生成引擎：与 vLLM（当使用 FSDP 作为训练后端时支持 vLLM 0.8.2 及以上版本）、SGLang（积极开发用于多轮代理强化学习、VLM RLHF、基于服务器的强化学习和部分 Rollout）以及 HF Transformers（与 Qwen-3、Llama3.1、Gemma2 等流行 HuggingFace 模型集成）集成。
    
- 强化学习算法：VeRL 支持广泛的强化学习算法，包括 PPO、GRPO、ReMax、REINFORCE++、RLOO、PRIME、DAPO、DrGRPO、KL_Cov & Clip_Cov、PF-PPO、VAPO。
    
- 奖励机制：支持基于模型的奖励和基于函数的奖励（可验证奖励，适用于数学和编码等任务）。它还支持视觉 - 语言模型（VLMs）用于多模态强化学习（例如，Qwen2.5-vl 和 Kimi-VL）。
    
- 多模态与代理强化学习：明确支持 VLM 和具有工具调用能力的多轮交互。
    
- LLM 对齐策略：包括自博弈偏好优化（SPPO）。
    
- 优化技术：集成了 Flash attention 2、序列打包、通过 DeepSpeed Ulysses 实现的序列并行、LoRA（低秩适应）、Liger-kernel，并可扩展到 671B 模型和数百个 GPU，支持专家并行和多 GPU LoRA 强化学习以节省内存。
    
- 实验跟踪：与 wandb、swanlab、mlflow 和 tensorboard 集成。

### 3.4 VeRL 在大型语言模型（LLM）对齐与优化中的作用

VeRL 专门为 LLM 的后训练量身定制，这是 LLM 与人类偏好对齐并提高其在复杂任务上性能的关键步骤 17。该框架对各种强化学习算法的支持，包括 PPO 及其变体，以及灵活的奖励机制（基于模型、基于函数、基于 VLM），直接解决了 LLM 对齐的挑战 17。其高性能和可扩展性设计（例如，FSDP、Megatron-LM、3D-HybridEngine）对于训练和优化需要大量计算资源的超大型 LLM 至关重要 17。其开源性质和详细文档 17 旨在实现 LLM 强化学习工业级解决方案的民主化，例如基于 VeRL 构建的 DAPO 算法在 AIME 2024 上取得了最先进的成果 14。

VeRL 不仅仅是一个通用的强化学习框架；它明确为 LLM 和 RLHF 而设计 17。包含像 DAPO 14 这样先进的强化学习算法（引入了“解耦裁剪和动态采样策略优化”，并包含 Clip-Higher 和 Token-Level Policy Gradient Loss 等特定技术），表明 VeRL 旨在促进 LLM 对齐领域的尖端研究和实际应用。这表明 VeRL 走在 RLHF 的前沿。对于用户的“VeRL 优化工作”而言，这意味着该框架不仅提供了 PPO 的实现，还提供了一个平台，用于实验和实现更复杂的 RLHF 算法，以解决 LLM 训练中已知的挑战（例如，熵崩溃、奖励噪声、训练不稳定 14）。这暗示“优化”可能涉及利用这些高级算法，甚至在 VeRL 生态系统内贡献新的算法。

VeRL 强调与现有 LLM 框架（FSDP、Megatron-LM、vLLM、SGLang、HF Transformers）的“无缝集成”及其“混合控制器编程模型” 17 是一个关键的设计选择。使用强化学习训练大型 LLM 需要管理庞大的模型（高达 671B 参数 17）和高效的 Rollout 生成推理。一个单一的强化学习框架将难以应对。混合方法允许专业组件（例如，FSDP 用于训练，vLLM 用于推理）高效协同工作。计算和数据依赖的解耦 17 直接实现了这种无缝集成，这反过来又使 VeRL 能够实现 LLM 任务所需的高性能和可扩展性。3D-HybridEngine 进一步减少了内存冗余和通信开销 17，直接影响了效率。对于用户而言，这表明 LLM 强化学习不仅仅是关于强化学习算法本身，还涉及底层的分布式系统和模型服务基础设施。“优化工作”很可能涉及理解这些组件如何交互以及优化它们在 VeRL 中的配置，而不仅仅是调整强化学习超参数。这也意味着分布式计算和 LLM 服务框架方面的专业知识将是有益的。

## 4. 探索 VeRL 框架：架构与关键文件

### 4.1 整体框架结构与混合控制器模型

VeRL 的架构围绕“混合控制器编程模型”构建，该模型结合了单一控制器和多控制器范式 18。这种设计能够灵活表示和高效执行 LLM 的复杂后训练数据流。该框架使用 Ray 进行分布式工作器管理 18。VeRL 中的“角色”代表在同一进程中运行的一组工作器 18。预定义的角色包括

Actor、Rollout、ActorRollout、Critic、RefPolicy、RewardModel 和 ActorRolloutRef 18。

role_worker_mapping 字典建立了每个角色对应的工作器类 18。资源池用于管理全局 GPU 资源，允许灵活的设备映射和角色在同一 GPU 上的共置 18。

### 4.2 VeRL 中的数据处理：DataProto 与 RLHFDataset

VeRL 内部的数据交换通过 DataProto 类进行标准化 18。

- DataProto：该类作为函数之间数据交换的标准协议。它包含两个主要组件：

- batch：一个 tensordict.TensorDict 对象，它是 PyTorch 张量的字典式容器。它非常适合存储具有相同批量大小的张量，并允许对内容进行集体操作 18。
    
- meta_info：一个标准的 Python 字典，用于存储额外的元信息 18。

- DataProto 提供了数据操作的核心 API，包括 concat（连接 DataProto 对象）、make_iterator（创建用于小批量处理的迭代器，与 PyTorch 数据集兼容）、select（选择数据子集）和 to（将数据移动到指定的 PyTorch 设备）18。
    
- RLHFDataset：该类专门设计用于从 Parquet 文件中加载和预处理 RLHF 数据 18。它负责将文件本地缓存，将数据读取到 HuggingFace 数据集中，对提示进行分词，并可选地通过  
    ProcessorMixin 处理多模态图像/视频数据 18。它还支持按最大长度过滤提示和从检查点恢复。
    
- collate_fn：RLHFDataset 中的一个实用函数，用于将一批样本字典整理成批处理张量和数组，为模型输入做准备 18。

### 4.3 为任务定义奖励函数

用户必须根据其用于 PPO 训练的数据集或应用程序定义特定的奖励函数 18。VeRL 支持

基于模型的奖励（添加 RewardModel 角色，通常使用 Hugging Face 的 AutoModelForSequenceClassification 结构）和基于函数的奖励（用户使用 _select_rm_score_fn 等函数对每个数据集的奖励进行分类）18。

verl/utils/reward_score 目录中提供了数学和编码等任务的已实现奖励函数示例 18。

### 4.4 理解工作器类与资源管理

- 工作器类：

- ActorRolloutRefWorker：一个预实现的工作器，可根据配置充当独立的 Actor、Rollout、ActorRollout HybridEngine 或 ActorRolloutRef HybridEngine 18。
    
- CriticWorker、RewardModelWorker、Reference model workers：为 PyTorch FSDP 和 Megatron-LM 后端预实现 18。
    
- verl.single_controller.Worker：表示一个分布式工作器，管理其自身的初始化、通信设置和设备配置 18。
    
- verl.single_controller.WorkerGroup：用于管理分布式系统中工作器集合的基类 18。

- 资源管理：

- verl.single_controller.ResourcePool：管理跨多个节点的资源池，跟踪进程计数和 GPU 分配 18。它计算池中所有节点的总世界大小、局部世界大小和局部排名。
    
- RayWorkerGroup：扩展了 WorkerGroup，提供 Ray 特有的功能，用于创建和管理 Ray Actor 组 18。

- RayPPOTrainer 是核心组件，负责初始化工作器、在分配的 GPU 上初始化模型，并执行 PPO 训练 18。

### 4.5 VeRL 中 PPO 示例架构的逐步解析

main_ppo.py 入口点（如 VeRL 文档中所述 18）概述了设置和运行 PPO 进行 LLM 后训练的步骤。

- 步骤 1：定义数据：用户预处理数据集并将其存储在 Parquet 文件中。RLHFDataset 加载并分词这些文件，至少需要一个 prompt 字段。data_preprocess 目录中提供了示例 18。
    
- 步骤 2：定义奖励函数：用户定义特定于任务的奖励函数（例如，用于 GSM8k、MATH）或使用基于模型的 RM。RewardManager 选择适当的函数 18。
    
- 步骤 3：定义工作器类：构建 role_worker_mapping 以将工作器类（例如，ActorRolloutRefWorker、CriticWorker）分配给角色。定义用于 GPU 分配的资源池 ID 和规范 18。
    
- 步骤 4：定义奖励模型/函数：指定任务是使用基于模型的 RM 还是基于函数的 RM。如果基于模型，则添加 RewardModel 角色；如果基于函数，则使用 _select_rm_score_fn 18。
    
- 步骤 5：定义、初始化和运行 PPO 训练器：RayPPOTrainer 使用用户配置、分词器、工作器映射、资源池和奖励函数进行初始化。trainer.init_workers() 在 GPU 上初始化模型，trainer.fit() 执行训练 18。
    
- VeRL 旨在通过重用现有 Ray 模型工作器、资源池和奖励函数，轻松扩展到其他强化学习算法 18。

DataProto、RLHFDataset、不同的工作器角色（Actor、Critic、RewardModel）和资源池的详细分解 18 揭示了高度模块化的架构。这不仅仅是代码组织的问题；它直接促进了“VeRL 优化工作”。如果用户想要优化奖励函数，他们可以专注于

RewardModelWorker 及其相关数据和逻辑，而无需重新设计整个 PPO 训练器或 Rollout 机制。同样，如果他们需要自定义模型，他们可以扩展现有的 FSDP 或 Megatron 后端 18。这种模块化意味着“优化”可以是有针对性的。VeRL 的设计允许对特定组件（例如，更好的奖励模型、更高效的 Rollout 引擎、自定义策略网络）进行集中实验和改进，而不会破坏整个强化学习管道，这对于需要不同组件专业知识的复杂 LLM 任务至关重要。

tensordict.TensorDict 在 DataProto 的 batch 组件中的使用 18 是一个微妙但强大的设计选择。

TensorDict 构建在 PyTorch 生态系统之上，允许将张量字典作为单个张量进行操作，非常适合分布式设置中的数据共享 18。这是一个直接影响大规模强化学习（尤其是 LLM）性能和内存效率的技术细节。

TensorDict 处理具有相同批量大小的张量的能力及其与 PyTorch 数据集的兼容性 18 意味着数据可以在不同的分布式工作器（例如，Actor、Critic、奖励模型）之间高效传递，而无需昂贵的数据转换或冗余内存复制。这直接促进了 VeRL 的“高性能”和“高效 Actor 模型重分片”目标 17。对于用户而言，这意味着理解 PyTorch 的

tensordict 库对于 VeRL 中的高级数据操作和调试将非常有价值。它还表明 VeRL 的性能提升不仅来自算法选择，还来自与底层 PyTorch 分布式原语的深度集成，这是高性能深度学习框架中的一个常见趋势。

## 5. 强化学习数据：从原始到就绪

### 5.1 强化学习数据的性质：观察、动作与奖励

在强化学习中，数据是通过智能体与环境的交互生成的，形成（状态、动作、奖励、下一状态）元组序列，通常称为“轨迹”或“回合” 1。

- 观察（状态）：

- 视觉数据：对于机器人或游戏等任务，观察通常是高维视觉输入，例如图像或视频帧的原始像素数据 3。深度强化学习将强化学习与深度神经网络结合，以处理这些高维状态空间，直接从原始像素数据中学习 4。
    
- 基于特征的数据：除了原始像素，观察也可以是表示环境状态的低维特征向量（例如，CartPole 中的小车位置、杆子角度 1）。这些特征可以是手工设计的，也可以由神经网络提取。
    
- 序列数据：强化学习观察本质上是序列性的，这意味着当前观察通常依赖于过去的观察和动作。这种时间依赖性至关重要 3。在视觉强化学习中，智能体的状态可以重新定义为多个连续观察的序列（例如，堆叠  
    k 帧）以部分恢复时间依赖性 5。

- 动作：

- 离散动作：有限的一组不同选择（例如，向左移动、向右移动、跳跃）1。Atari 游戏通常具有离散动作空间 12。
    
- 连续动作：可以在某个范围内取任何值的动作（例如，电机扭矩、转向角）3。机器人控制任务通常涉及连续动作空间 12。

- 奖励：

- 标量奖励：通常是一个表示即时反馈的单一数值（例如，+1，-100）1。
    
- 稀疏奖励：很少获得的奖励，通常只在长序列动作的末尾获得（例如，只有在解决迷宫时才获得奖励，而不是每一步正确动作都获得奖励）。这是视觉强化学习中的一个常见挑战 3。
    
- 密集奖励：更频繁地给予的奖励，提供更即时的反馈。
    
- 奖励塑造：设计奖励函数以引导学习，可能通过添加中间奖励，同时旨在保持最优策略 3。逆向强化学习（IRL）可以从专家演示中推断奖励函数 3。

表：常见强化学习数据类型及其表示

|   |   |   |   |   |
|---|---|---|---|---|
|类别|类型|表示形式|示例/上下文|说明|
|观察|原始像素（图像/视频）|高维数组（例如，RGB 图像的 HxWx3，堆叠帧的 NxHxWx3）|Atari 游戏 7、相机输入机器人 3、视觉导航 3|通常由 CNN 处理；需要大量计算资源。|
|观察|特征向量（本体感受/传感器数据）|低维数值数组（例如，位置、速度、关节角度）|CartPole 1、DeepMind Control Suite 8|可与视觉数据结合以获得更丰富的状态表示。|
|动作|离散|整数标签（例如，左、右、跳跃的 0、1、2）|Atari 游戏 7、CartPole 1|适用于具有有限选择集的任务。|
|动作|连续|范围内的浮点值（例如，电机扭矩、转向角）|机器人运动 12、DeepMind Control Suite 8|需要能够处理连续空间的策略优化方法。|
|奖励|标量|单一数值（例如，+1，-100）|迷宫导航 1、CartPole 1、ViZDoom 22|可以是稀疏的或密集的；对引导学习至关重要。|
|奖励|人类反馈/偏好|人类提供的排名、比较或分数|LLM 的 RLHF 11、学习奖励函数 10|用于训练奖励模型，尤其适用于主观任务。|

### 5.2 强化学习中关键的数据预处理与清洗技术

数据预处理在机器学习中至关重要，它能将原始数据转换为可理解和可用的格式，从而提高数据质量和模型性能 24。对于强化学习，特别是视觉和序列数据，这一点尤为重要。

#### 5.2.1 一般数据卫生

24

- 处理缺失值：缺失数据可能导致分析偏差。策略包括插补（用均值、中位数、众数或预测模型填充）或删除行/列 25。
    
- 移除重复项：重复记录可能导致结果偏差。识别并消除相同或近似重复的条目 26。
    
- 纠正不一致的格式：标准化格式（例如，日期格式、字符串大小写）以保持一致性 26。
    
- 异常值检测与处理：异常值可能扭曲结果。Z-score 或四分位距（IQR）等技术有助于识别和处理异常 26。

#### 5.2.2 数据转换

24

- 缩放与归一化：将数值调整到共同的尺度（例如，Min-Max 缩放的 0 到 1，标准化后的零均值/单位方差），确保没有单一特征因其尺度而主导模型 26。这对于依赖距离度量或梯度下降的算法至关重要。
    
- 编码分类变量：将分类数据转换为适合模型训练的数值形式（例如，独热编码、标签编码）26。
    
- 特征工程与提取：从现有数据中创建新的、信息丰富的特征，或选择重要的特征以提高模型性能并降低维度 24。对于视觉数据，这可能涉及提取语义或几何特征 21。

#### 5.2.3 视觉与序列数据特有技术

- 图像大小调整与裁剪：确保模型输入图像尺寸一致 28。裁剪可以聚焦图像的相关部分 28。
    
- 色彩校正与降噪：通过调整亮度、对比度以及应用滤镜（高斯模糊、中值模糊、拉普拉斯滤镜）来增强图像质量并减少噪声 28。
    
- 帧堆叠：对于序列视觉观察，堆叠多个连续帧（例如，4 帧）来表示状态，提供时间信息，帮助智能体理解运动和时间依赖性 5。这是 Atari 强化学习中的常见做法。
    
- 处理视觉观察中的噪声：真实世界的视觉数据富含不相关噪声。像任务相关掩码采样（TRMS）这样的技术利用分割模型来识别和过滤掉与任务不相关的部分，专注于高级抽象而不是原始像素，以提高鲁棒性和效率 5。

#### 5.2.4 视觉强化学习中的数据增强：扩展数据集

数据增强涉及通过操作现有样本来生成高质量的人工数据，以人工方式扩大训练数据集并增强泛化能力 26。这对于强化学习尤其重要，因为其样本效率低且存在泛化差距 30。

- 常见技术 26：

- 几何变换：旋转、平移、翻转（水平/垂直）、缩放、剪切，以模拟相机方向或物体姿态的变化 28。
    
- 颜色抖动/光度变换：随机调整色调、饱和度、亮度或对比度，以模拟不同的光照条件 28。
    
- 添加噪声：添加随机高斯、泊松、椒盐噪声，使网络对传感器失真具有鲁棒性 28。
    
- 模糊：添加高斯或运动模糊，以模拟失焦镜头或相机移动 28。
    
- 图像混合/神经融合：组合多张图像 33。

- 高级数据增强：AutoAugment 使用强化学习来寻找最优的数据增强策略，将其框定为决策过程，其中控制器 RNN（智能体）选择子策略（操作、概率、幅度），并根据网络性能（奖励）进行更新 31。
    
- 目的：数据增强提高了模型准确性，为训练所有参数提供了足够的数据（尤其是在数据收集困难时），防止过拟合，并节省了收集更多真实世界数据的时间/成本 32。

强化学习，特别是在机器人和视觉任务中，由于真实世界交互的成本和安全问题，通常严重依赖模拟进行数据收集 34。然而，纯粹在模拟中训练的模型在部署到真实世界时往往表现不佳——这就是“模拟到现实”的鸿沟 36。数据增强 29 和生成合成数据 34 是弥合这一鸿沟的明确策略，通过多样化训练数据，使策略对真实世界的变化更具泛化性和鲁棒性。真实世界数据收集的高成本和危险性导致了对模拟的依赖。模拟保真度的不完善造成了模拟到现实的鸿沟。数据增强和合成数据生成通过增加数据多样性并使智能体对变化具有鲁棒性，直接解决了这一鸿沟，从而提高了真实世界的性能。对于用户的“VeRL 优化工作”而言，这意味着尽管 VeRL 专注于 LLM，但通过增强实现数据多样性和鲁棒性的原则普遍适用。如果 LLM 与模拟环境（例如，基于文本的游戏、编码环境）交互，那么增强“观察”（例如，不同措辞的提示、噪声输入）或“奖励”可以显著提高其对真实世界用户交互的泛化能力。挑战从视觉像素噪声转向了语言或语义噪声。

传统上，状态是环境的完整表示 1。然而，在视觉强化学习中，原始像素输入是高维且通常有噪声的 5。状态的概念从单个原始图像演变为“k 个连续观察序列”（帧堆叠）以捕获时间动态 5。此外，像 TRMS 5 这样的先进技术表明，“状态”不仅仅是原始输入，而是从视觉信息中提取的

经过处理的、与任务相关的抽象，有效地“屏蔽”了不相关的噪声。这种趋势是学习更紧凑、更鲁棒、与任务相关的状态表示，而不是仅仅依赖原始、高维输入。这是一种向更高层次抽象的转变，类似于人类通过关注相关元素来处理视觉信息的方式 5。对于 LLM 而言，这意味着“状态”（提供给 LLM 的上下文）可能需要类似的“预处理”或“抽象”。未来优化工作可能涉及采用技术，仅从提示或对话历史中提取最“任务相关”的信息，从而为 LLM 创建“掩码”或“抽象”的输入状态。这可以通过减少噪声并使 LLM 专注于关键信息来提高效率和性能。

### 5.3 整理您的强化学习数据集库：关键资源与论文

表：推荐的公共强化学习数据集与模拟环境

|   |   |   |   |   |
|---|---|---|---|---|
|环境/套件|描述|数据类型焦点|相关性|链接/仓库|
|OpenAI Gym|用于 RL 算法开发和基准测试的标准化环境工具包 19。包括经典控制、Atari、MuJoCo、机器人、Box2D 19。|多样（像素、特征向量、离散/连续动作）。|许多 RL 算法的经典基准，初学者的良好起点 1。|[https://gymnasium.farama.org/](https://gymnasium.farama.org/) 2|
|DeepMind Control Suite (DMCS)|由 MuJoCo 物理引擎驱动的一套连续控制任务，具有标准化结构和可解释奖励 8。|特征向量、像素观察（通过包装器）、连续动作 8。|连续 RL 算法的基准测试，高保真物理模拟 8。|[https://github.com/deepmind/dm_control](https://github.com/deepmind/dm_control) 8|
|Atari Arcade Learning Environment (ALE)|用于开发 Atari 2600 游戏 AI 智能体的框架，常使用原始像素输入 12。|原始像素观察、离散动作。|DRL 的历史基准，特别是深度 Q 网络（DQN）36。|[https://github.com/mgbellemare/Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) 19|
|D4RL|专门用于离线强化学习的环境和数据集集合，策略从静态数据集学习，无需在线交互 19。包括 Maze2D、AntMaze、Adroit（机器人手）、Gym、Flow（交通）、FrankKitchen（操作）41。|混合（特征向量、人类演示、异构策略）。|对于在线数据收集困难的数据驱动 RL 研究和实际应用至关重要 41。|[https://sites.google.com/view/d4rl-anonymous/](https://sites.google.com/view/d4rl-anonymous/) 41|
|SAPIEN / ManiSkill|逼真、物理丰富的模拟环境，用于机器人视觉和交互任务，拥有大型关节对象数据集（PartNet-Mobility）35。|RGBD + 分割视觉数据、运动标注、机器人状态、连续动作。|机器人学习的高保真模拟，支持 GPU 并行化数据收集 35。|[https://sapien.ucsd.edu/](https://sapien.ucsd.edu/) 35|
|CARLA Simulator|用于自动驾驶研究的开源模拟器，提供逼真的城市环境和传感器数据 19。|点云、语义标签、自我运动、RGB 图像、地面真实掩码 44。|自动驾驶系统的高保真模拟，支持各种交通和天气条件 44。|[https://carla.org/](https://carla.org/) 19|
|NVIDIA Isaac Lab / Isaac Sim|用于机器人学习的开源框架，具有高保真物理模拟和 RTX 渲染，与 Isaac Sim 集成 34。|合成数据、真实机器人捕获数据、感知驱动（视觉）数据。|加速机器人技能开发，危险场景的安全验证场，降低真实世界数据收集成本 34。|[https://developer.nvidia.com/isaac-robot-learning](https://developer.nvidia.com/isaac-robot-learning) 34|
|Unity ML-Agents|用于在 Unity 编辑器中创建 RL 环境和训练智能体的工具包 39。|多样（向量观察、视觉观察、离散/连续动作）。|用户友好的游戏 AI 和机器人平台，允许自定义环境 47。|[https://unity.com/products/machine-learning](https://unity.com/products/machine-learning) 39|

表：强化学习数据与环境设计领域的有影响力论文

|   |   |   |   |   |
|---|---|---|---|---|
|论文标题|作者|年份|关键贡献/相关性|对用户的背景说明|
|"Reinforcement Learning: An Introduction"|Richard S. Sutton 和 Andrew G. Barto|2018（第二版）|强化学习的奠基性教科书，全面解释核心概念 36。|提供了理解强化学习的理论基础，是深入研究特定算法或框架前的必要准备。|
|"Playing Atari with Deep Reinforcement Learning"|Volodymyr Mnih 等|2013|引入深度 Q 网络（DQN），展示 DRL 如何直接从原始像素输入中学习复杂任务，如 Atari 游戏 5。|一篇里程碑式的论文，展示了 DRL 在视觉数据方面的强大能力，为后续许多进展奠定了基础。|
|"Trust Region Policy Optimization (TRPO)"|John Schulman 等|2015|引入了用于稳定策略优化的信赖域方法，是 PPO 的前身 36。|为 PPO 的发展提供了背景，解释了 PPO 旨在解决的问题（策略梯度的稳定性）。|
|"Proximal Policy Optimization Algorithms"|John Schulman 等|2017|引入 PPO 的开创性论文，详细阐述了其裁剪替代目标和经验优势 12。|用户查询第一部分的核心论文。|
|"DeepMind Control Suite"|Yuval Tassa 等|2018 (arXiv:1801.00690)|引入了 MuJoCo 中广泛使用的连续控制任务基准套件，实现了 RL 算法的标准化评估 8。|提供了一组实用的环境，用于测试和比较连续控制 RL 算法，与机器人技术相关。|
|"D4RL: Datasets for Deep Data-Driven Reinforcement Learning"|Justin Fu 等|2020|引入了一系列用于离线 RL 的基准和数据集，侧重于真实世界数据特征 41。|直接回应了用户关于“数据情况”和“数据集仓库”的查询，强调了静态、预收集数据的重要性。|
|"A Comprehensive Survey of Data Augmentation in Visual Reinforcement Learning"|Guozheng Ma 等|2022 (arXiv:2210.04561)|提供了视觉 RL 中数据增强技术的统一框架和分类，解决了样本效率和泛化差距问题 30。|对于理解如何“清洗数据”和准备视觉输入以进行 RL 至关重要，特别是为了提高鲁棒性和样本效率。|
|"Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)"|14|2024|提出了一种基于 VeRL 的算法，通过解决熵崩溃和奖励噪声等挑战，在大规模 LLM RL 中取得了最先进的成果 14。|直接关联到 VeRL，并提供了 LLM 优化的前沿见解，展示了 VeRL 在高级任务中的实际应用。|

## 6. VeRL 优化之旅的下一步

### 6.1 初始设置与实验的实用建议

在深入研究 VeRL 之前，确保对核心强化学习概念有扎实的掌握。可以利用对初学者友好的资源，例如 Hugging Face 的深度强化学习课程、DeepMind x UCL 的强化学习入门或 FreeCodeCamp 的强化学习课程 48。这些课程涵盖了 Q 学习、策略梯度和 Actor-Critic 方法 48。

首先从 OpenAI Gym/Gymnasium 中较简单的环境（如 CartPole）开始 1。这有助于理解智能体 - 环境交互循环（

env.step()、观察、奖励），而无需复杂的视觉输入 2。

一旦熟悉了强化学习基础知识，请查阅 VeRL 的官方文档，了解安装和快速入门指南 17。VeRL 文档中 PPO 示例架构 18 是理解其结构以及 PPO 如何在框架中实现的出色起点。

了解 VeRL 期望数据集以 Parquet 文件形式存储，并且至少需要一个 prompt 字段 18。熟悉 VeRL 示例中的

data_preprocess 目录，以获取有关准备 GSM8k 或 MATH 等数据集的指导 18。

对于您的特定 LLM 优化任务，仔细定义奖励函数。探索 VeRL 的 verl/utils/reward_score 目录，查找基于函数奖励的示例 18。如果使用基于模型的奖励，请了解

RewardModelWorker 如何支持 Hugging Face 模型 18。

利用 VeRL 与 wandb、mlflow 或 tensorboard 等工具集成的实验跟踪功能 17。这对于监控训练进度、比较不同运行和调试至关重要。

从小模型或 VeRL 中较简单的任务开始，然后再扩展到 671B 模型或数百个 GPU。迭代地设计奖励函数和调整超参数。

强化学习开发的高度敏感性决定了其迭代性质。与可以评估静态验证集性能的传统监督学习不同，强化学习训练对超参数、奖励函数设计和探索策略高度敏感。微小的变化可能导致截然不同的结果，而且通常智能体的行为是非直观的。强化学习固有的试错性质 1 加上其对设计选择的敏感性，导致了对强大实验跟踪的需求。没有它，几乎不可能系统地理解智能体表现好坏的原因，或者重现成功的结果。跟踪指标、配置甚至智能体行为成为主要的诊断工具。对于用户的“VeRL 优化工作”而言，这意味着从第一天起建立一个强大的实验跟踪系统与编写代码同样重要。这是应用于强化学习开发的“科学方法”，允许系统地进行假设检验和数据驱动的改进，这与重视方法论的“传统方向”思维方式非常契合。

### 6.2 进一步学习资源与社区参与

- VeRL 文档：深入了解 VeRL API、编程模型和高级用法的主要资源 17。请注意有关添加模型、多轮 Rollout 支持和扩展到其他强化学习算法的部分 17。
    
- 相关研究论文：深入研究与 VeRL 相关的论文，例如“HybridFlow”（VeRL 的基础）和 DAPO 14。这些论文提供了对 VeRL 中实现的设计选择和高级技术的理解。
    
- 数据预处理调查：查阅视觉强化学习中数据增强的综合调查 30 和一般数据预处理的调查 24。虽然有些侧重于图像分类，但清洗、转换和增强的原则高度相关。
    
- 强化学习社区：参与在线强化学习社区（例如，Reddit 上的 r/reinforcementlearning 36），进行讨论、故障排除，并了解最新研究。
    
- 开源贡献：考虑探索 GitHub 上 VeRL 的“良好入门问题”（good first issues）17。为开源项目做贡献是加深理解和获得实践经验的绝佳方式。
    
- 持续学习：强化学习和 LLM 领域发展迅速。持续关注新研究、框架和最佳实践。

VeRL 是一个相对较新的框架（文章最新更新日期：2025 年 6 月 6 日 17），LLM 的强化学习领域也处于前沿。VeRL 支持多种算法，包括许多最新算法 17，并且是研究论文的开源版本 17，这表明这是一个高度动态和研究驱动的环境。“项目路线图”和“良好入门问题” 17 进一步强调了这一点。LLM 的快速发展正在推动新的强化学习方法（例如，多模态强化学习、代理强化学习 17）。这表明“优化”不会是一个静态过程，而是持续参与新方法和社区发展。对于用户而言，这意味着报告虽然提供了坚实的基础，但持续学习和参与 VeRL 社区以及更广泛的强化学习/LLM 研究对于长期成功至关重要。“优化工作”将不仅仅是应用现有工具，还包括适应并可能为不断发展的最先进技术做出贡献。这强调了与开源生态系统保持联系的重要性。

#### 引用的著作

1. Reinforcement Learning - GeeksforGeeks, 访问时间为 六月 23, 2025， [https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/](https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/)
    
2. Reinforcement Learning with Gymnasium: A Practical Guide ..., 访问时间为 六月 23, 2025， [https://www.datacamp.com/tutorial/reinforcement-learning-with-gymnasium](https://www.datacamp.com/tutorial/reinforcement-learning-with-gymnasium)
    
3. Reinforcement learning for vision tasks | Images as Data Class ..., 访问时间为 六月 23, 2025， [https://library.fiveable.me/images-as-data/unit-5/reinforcement-learning-vision-tasks/study-guide/UTWVoVk6ymTCYZ8r](https://library.fiveable.me/images-as-data/unit-5/reinforcement-learning-vision-tasks/study-guide/UTWVoVk6ymTCYZ8r)
    
4. Reinforcement learning | Computer Vision and Image Processing Class Notes - Fiveable, 访问时间为 六月 23, 2025， [https://library.fiveable.me/computer-vision-and-image-processing/unit-6/reinforcement-learning/study-guide/xsWdJ6HKQejt7UXi](https://library.fiveable.me/computer-vision-and-image-processing/unit-6/reinforcement-learning/study-guide/xsWdJ6HKQejt7UXi)
    
5. Learning Robust Representations for Visual ... - OpenReview, 访问时间为 六月 23, 2025， [https://openreview.net/pdf/33086987ac2960e94d11ccf4760f2a2af2a0e279.pdf](https://openreview.net/pdf/33086987ac2960e94d11ccf4760f2a2af2a0e279.pdf)
    
6. Tutorial: Deep Reinforcement Learning, 访问时间为 六月 23, 2025， [https://icml.cc/2016/tutorials/deep_rl_tutorial.pdf](https://icml.cc/2016/tutorials/deep_rl_tutorial.pdf)
    
7. Getting Started With OpenAI Gym: The Basic Building Blocks | DigitalOcean, 访问时间为 六月 23, 2025， [https://www.digitalocean.com/community/tutorials/getting-started-with-openai-gym](https://www.digitalocean.com/community/tutorials/getting-started-with-openai-gym)
    
8. [1801.00690] DeepMind Control Suite - ar5iv, 访问时间为 六月 23, 2025， [https://ar5iv.labs.arxiv.org/html/1801.00690](https://ar5iv.labs.arxiv.org/html/1801.00690)
    
9. A Beginner's Guide to Deep Reinforcement Learning - GeeksforGeeks, 访问时间为 六月 23, 2025， [https://www.geeksforgeeks.org/a-beginners-guide-to-deep-reinforcement-learning/](https://www.geeksforgeeks.org/a-beginners-guide-to-deep-reinforcement-learning/)
    
10. Enhancing Rating-Based Reinforcement Learning to Effectively Leverage Feedback from Large Vision-Language Models - arXiv, 访问时间为 六月 23, 2025， [https://arxiv.org/html/2506.12822v1](https://arxiv.org/html/2506.12822v1)
    
11. What is RLHF? - Analytics Vidhya, 访问时间为 六月 23, 2025， [https://www.analyticsvidhya.com/blog/2023/05/reinforcement-learning-from-human-feedback/](https://www.analyticsvidhya.com/blog/2023/05/reinforcement-learning-from-human-feedback/)
    
12. Proximal Policy Optimization Algorithms, 访问时间为 六月 23, 2025， [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
    
13. Truly Proximal Policy Optimization 1 INTRODUCTION - Proceedings of Machine Learning Research, 访问时间为 六月 23, 2025， [http://proceedings.mlr.press/v115/wang20b/wang20b.pdf](http://proceedings.mlr.press/v115/wang20b/wang20b.pdf)
    
14. An Open-Source LLM Reinforcement Learning System at Scale - DAPO, 访问时间为 六月 23, 2025， [https://dapo-sia.github.io/static/pdf/dapo_paper.pdf](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)
    
15. The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games - NIPS, 访问时间为 六月 23, 2025， [https://papers.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf](https://papers.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf)
    
16. (PDF) Improving Value Estimation Critically Enhances Vanilla Policy Gradient, 访问时间为 六月 23, 2025， [https://www.researchgate.net/publication/392105000_Improving_Value_Estimation_Critically_Enhances_Vanilla_Policy_Gradient](https://www.researchgate.net/publication/392105000_Improving_Value_Estimation_Critically_Enhances_Vanilla_Policy_Gradient)
    
17. volcengine/verl: verl: Volcano Engine Reinforcement ... - GitHub, 访问时间为 六月 23, 2025， [https://github.com/volcengine/verl](https://github.com/volcengine/verl)
    
18. Welcome to verl's documentation! — verl documentation, 访问时间为 六月 23, 2025， [https://verl.readthedocs.io/en/latest/index.html](https://verl.readthedocs.io/en/latest/index.html)
    
19. Machine Learning Datasets | Papers With Code, 访问时间为 六月 23, 2025， [https://paperswithcode.com/datasets?v=lst&mod=environment&page=1](https://paperswithcode.com/datasets?v=lst&mod=environment&page=1)
    
20. Visual Reinforcement Learning with Residual Action, 访问时间为 六月 23, 2025， [https://ojs.aaai.org/index.php/AAAI/article/view/34097/36252](https://ojs.aaai.org/index.php/AAAI/article/view/34097/36252)
    
21. XPG-RL: Reinforcement Learning with Explainable Priority Guidance for Efficiency-Boosted Mechanical Search - arXiv, 访问时间为 六月 23, 2025， [https://arxiv.org/html/2504.20969v2](https://arxiv.org/html/2504.20969v2)
    
22. Default scenarios/environments - ViZDoom Documentation, 访问时间为 六月 23, 2025， [https://vizdoom.farama.org/environments/default/](https://vizdoom.farama.org/environments/default/)
    
23. Preparing data for RLHF | Python, 访问时间为 六月 23, 2025， [https://campus.datacamp.com/courses/reinforcement-learning-from-human-feedback-rlhf/foundational-concepts?ex=8](https://campus.datacamp.com/courses/reinforcement-learning-from-human-feedback-rlhf/foundational-concepts?ex=8)
    
24. Data Preprocessing: Concepts, Importance, & Tools | Astera, 访问时间为 六月 23, 2025， [https://www.astera.com/type/blog/data-preprocessing/](https://www.astera.com/type/blog/data-preprocessing/)
    
25. Data Preprocessing in Machine Learning: Steps & Best Practices - lakeFS, 访问时间为 六月 23, 2025， [https://lakefs.io/blog/data-preprocessing-in-machine-learning/](https://lakefs.io/blog/data-preprocessing-in-machine-learning/)
    
26. Data Preprocessing: A Complete Guide with Python Examples - DataCamp, 访问时间为 六月 23, 2025， [https://www.datacamp.com/blog/data-preprocessing](https://www.datacamp.com/blog/data-preprocessing)
    
27. Data Preprocessing for Supervised Learning Techniques - Importance and Best Practices, 访问时间为 六月 23, 2025， [https://moldstud.com/articles/p-data-preprocessing-for-supervised-learning-techniques-importance-and-best-practices](https://moldstud.com/articles/p-data-preprocessing-for-supervised-learning-techniques-importance-and-best-practices)
    
28. Best Practices for Image Preprocessing | Keylabs, 访问时间为 六月 23, 2025， [https://keylabs.ai/blog/best-practices-for-image-preprocessing-in-image-classification/](https://keylabs.ai/blog/best-practices-for-image-preprocessing-in-image-classification/)
    
29. Get Started with Image Preprocessing and Augmentation for Deep ..., 访问时间为 六月 23, 2025， [https://www.mathworks.com/help/images/get-started-with-image-preprocessing-and-augmentation-for-deep-learning.html](https://www.mathworks.com/help/images/get-started-with-image-preprocessing-and-augmentation-for-deep-learning.html)
    
30. arxiv.org, 访问时间为 六月 23, 2025， [https://arxiv.org/abs/2210.04561](https://arxiv.org/abs/2210.04561)
    
31. (PDF) A Comprehensive Survey on Data Augmentation, 访问时间为 六月 23, 2025， [https://www.researchgate.net/publication/380635431_A_Comprehensive_Survey_on_Data_Augmentation](https://www.researchgate.net/publication/380635431_A_Comprehensive_Survey_on_Data_Augmentation)
    
32. Your 101 Guide to Data Augmentation Techniques - ProjectPro, 访问时间为 六月 23, 2025， [https://www.projectpro.io/article/data-augmentation/777](https://www.projectpro.io/article/data-augmentation/777)
    
33. A Comprehensive Survey on Data Augmentation - arXiv, 访问时间为 六月 23, 2025， [https://arxiv.org/html/2405.09591v3](https://arxiv.org/html/2405.09591v3)
    
34. Robot Learning in Simulation | Use Case | NVIDIA, 访问时间为 六月 23, 2025， [https://www.nvidia.com/en-us/use-cases/robot-learning/](https://www.nvidia.com/en-us/use-cases/robot-learning/)
    
35. SAPIEN, 访问时间为 六月 23, 2025， [https://sapien.ucsd.edu/](https://sapien.ucsd.edu/)
    
36. Must read papers for Reinforcement Learning : r/reinforcementlearning - Reddit, 访问时间为 六月 23, 2025， [https://www.reddit.com/r/reinforcementlearning/comments/1is773d/must_read_papers_for_reinforcement_learning/](https://www.reddit.com/r/reinforcementlearning/comments/1is773d/must_read_papers_for_reinforcement_learning/)
    
37. What is OpenAI Gym? - Milvus, 访问时间为 六月 23, 2025， [https://milvus.io/ai-quick-reference/what-is-openai-gym](https://milvus.io/ai-quick-reference/what-is-openai-gym)
    
38. DeepMind Control Suite Dataset - Papers With Code, 访问时间为 六月 23, 2025， [https://paperswithcode.com/dataset/deepmind-control-suite](https://paperswithcode.com/dataset/deepmind-control-suite)
    
39. Master Reinforcement Learning: a curated list of resources dedicated to RL | Kaggle, 访问时间为 六月 23, 2025， [https://www.kaggle.com/discussions/general/466977](https://www.kaggle.com/discussions/general/466977)
    
40. AtariARI Dataset | Papers With Code, 访问时间为 六月 23, 2025， [https://paperswithcode.com/dataset/atariari](https://paperswithcode.com/dataset/atariari)
    
41. D4RL - Google Sites, 访问时间为 六月 23, 2025， [https://sites.google.com/view/d4rl-anonymous/](https://sites.google.com/view/d4rl-anonymous/)
    
42. D4RL Dataset - Papers With Code, 访问时间为 六月 23, 2025， [https://paperswithcode.com/dataset/d4rl](https://paperswithcode.com/dataset/d4rl)
    
43. Sapien, 访问时间为 六月 23, 2025， [https://sapien.hillbot.ai/](https://sapien.hillbot.ai/)
    
44. Dataset - CarlaSC - GitHub Pages, 访问时间为 六月 23, 2025， [https://umich-curly.github.io/CarlaSC.github.io/dataset/](https://umich-curly.github.io/CarlaSC.github.io/dataset/)
    
45. isp-uv-es/IPL-CARLA-dataset - Hugging Face, 访问时间为 六月 23, 2025， [https://huggingface.co/datasets/isp-uv-es/IPL-CARLA-dataset](https://huggingface.co/datasets/isp-uv-es/IPL-CARLA-dataset)
    
46. Example Learning Environments - Unity ML-Agents Toolkit - GitHub Pages, 访问时间为 六月 23, 2025， [https://unity-technologies.github.io/ml-agents/Learning-Environment-Examples/](https://unity-technologies.github.io/ml-agents/Learning-Environment-Examples/)
    
47. Making a New Learning Environment - Unity ML-Agents Toolkit, 访问时间为 六月 23, 2025， [https://unity-technologies.github.io/ml-agents/Learning-Environment-Create-New/](https://unity-technologies.github.io/ml-agents/Learning-Environment-Create-New/)
    
48. 5 Free Courses on Reinforcement Learning - MachineLearningMastery.com, 访问时间为 六月 23, 2025， [https://machinelearningmastery.com/5-free-courses-on-reinforcement-learning/](https://machinelearningmastery.com/5-free-courses-on-reinforcement-learning/)
    
49. Reinforcement Learning in 3 Hours | Full Course using Python - YouTube, 访问时间为 六月 23, 2025， [https://www.youtube.com/watch?v=Mut_u40Sqz4](https://www.youtube.com/watch?v=Mut_u40Sqz4)

**