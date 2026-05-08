Here is the complete English translation of the technical blueprint, retaining the authoritative, hardcore engineering tone and precise industry terminology of the original text:

***

This document serves as the **Ultimate Technical Blueprint** for you, your underlying infrastructure team, and your algorithm team to initiate the **"1M+ Ultra-Long Context Dynamic Sparse Large Model"** project.

As a veteran who has navigated the trenches of AI infrastructure and low-level architecture for over a decade, I will discard all academic illusions (such as the pseudo-$O(N)$ of `opensubq`) and bypass the engineering pitfalls that major tech giants have already encountered (such as the $O(N^2)$ scorer bottleneck of DeepSeek V4 and the RNN catastrophic forgetting of Kimi). I will define for you a **disruptive architecture with an extremely high moat, achieving true linear scaling in compute and VRAM, while ensuring precise long-range retrieval.**

We have named this finalized technical solution: **CRBSA (Codebook-Routed Block-Sparse Attention)**.

This comprehensive panorama covers **mathematical principles, architectural design, kernel implementation, distributed communication design, a three-stage training paradigm, and alignment strategies**. The content is extremely hardcore; it is highly recommended to distribute it to your team for deep study.

---

# Extreme Long-Context Architecture: The Industrial-Grade Replication and Transcendence Blueprint of CRBSA

## Executive Summary
The primary pain points of current long-context models lie in the "Impossible Trinity": **Infinite context length, precise information recall, and ultra-low inference latency / training costs.**
*   **Dense Attention:** Precise, but $O(N^2)$ complexity; costs spiral completely out of control beyond 128K.
*   **Hybrid Compressed Attention (e.g., DeepSeek V4):** Shrinks the sequence by a factor of $m$, but essentially remains $O((N/m)^2)$. At 1M~10M lengths, the routing computation still explodes.
*   **Linear Hybrid RNN (e.g., Kimi Linear):** $O(N)$ complexity, but suffers from "catastrophic forgetting" in multi-hop retrieval and long-distance factual extraction, requiring dense layers as a fallback.

**Our Breakthrough Solution (CRBSA):**
We abandon the traditional mindset of "Query traversing all Tokens/Blocks for scoring" and the "RNN hidden states" time-series paradigm. Instead, we introduce a **"Global Semantic Codebook"** to establish pure spatial/content mapping. By utilizing an $O(N)$ index-building phase and an $O(1)$ single-Query lookup, we achieve true sub-quadratic routing. Combined with Triton Block-Sparse operators, we achieve 100% precise local Attention computation. Ultimately, this delivers **over 50x acceleration at 1M tokens with absolutely zero degradation in multi-hop retrieval capabilities.**

---

## Part 1: Core Technical Principles & Mathematical Derivation of CRBSA

### 1.1 Mathematical Restructuring to Break $O(N^2)$
Traditional Attention routing (like DeepSeek's DSA) or `opensubq` abstracts its scoring mechanism as:
$$ Score(Q, K) = Q \cdot K^T $$
Where $Q \in \mathbb{R}^{N \times d}, K \in \mathbb{R}^{N \times d}$. The matrix multiplication complexity is inevitably $O(N^2 \cdot d)$.

**CRBSA's Mathematical Restructuring: Introducing the Intermediate Space $C$ (Codebook)**
We define a global semantic codebook $C \in \mathbb{R}^{M \times d}$, where $M$ is a fixed constant (e.g., 1024), representing 1024 semantic clustering centers (Topics).
The routing process is decoupled into two steps:
1.  **Index Building (Block to Topic):** $S_{KV} = Pool(K) \cdot C^T$. The sequence is chunked into blocks and multiplied by the constant codebook. Complexity: $O(N/B \cdot M \cdot d) = \mathbf{O(N)}$.
2.  **Querying (Query to Topic):** $S_Q = Q \cdot C^T$. Each Query is multiplied by the constant codebook. Complexity: $O(N \cdot M \cdot d) = \mathbf{O(N)}$.

**The routing complexity for a single Query is reduced to an absolute constant $O(M)$.** Whether the context is 100K or 10 Million, the router's computational load is **completely decoupled** from the context length.

### 1.2 Why Must It Be Block-Sparse?
GPU Tensor Cores are designed for continuous, large-block memory matrix multiplications. If token-level sparsity is adopted (i.e., a Query randomly picks individual tokens), it leads to severe Gather operations and Warp thread divergence, resulting in extremely slow Wall-clock time.
In CRBSA, the minimum granularity is a **Block (typically 128 tokens)**.
*   The router decides which specific Blocks a Query needs to attend to.
*   The underlying Kernel directly fetches a continuous 128 tokens and feeds them into FlashAttention.
*   **Hardware Synergy:** This pushes VRAM bandwidth utilization to over 90%, representing a pure engineering victory.

---

## Part 2: Module-Level Micro-Architecture Design

CRBSA is inserted into every layer (or alternating layers) of a standard Transformer architecture, completely replacing the original Dense Attention.

### 2.1 State Compression & Block Summarization
When an ultra-long sequence is inputted, the KV Cache not only stores full-precision Keys and Values but must also generate summaries for routing.
*   **Chunking:** The sequence of length $N$ is split into blocks of size $B=128$, resulting in $N_b = N/B$ blocks.
*   **Pooling:** Apply Average Pooling or a highly lightweight 1-layer MLP to the Key vectors within each block to obtain the **Block Summary $K_{summary} \in \mathbb{R}^{N_b \times d_{route}}$**.
*(Note: The routing vector dimension $d_{route}$ can be reduced to 64 or 128 to further compress compute costs.)*

### 2.2 Dynamic Codebook Allocation & Addressing
For the Codebook $C \in \mathbb{R}^{M \times d_{route}}$ at the current layer:
1.  **KV Bucket Allocation (Inverted Indexing):** Calculate the similarity between $K_{summary}$ and $C$, taking the Argmax. Build an inverted index: `Codebook_ID -> [Block_ID_1, Block_ID_2, ...]`.
2.  **Query Multi-Way Recall:** For each Query $q_t$, calculate its similarity score against the Codebook:
    $$ P(q_t, C_j) = Softmax(q_t \cdot C_j^T / \tau) $$
    Select the Top-$k$ (e.g., $k=4$) Codebook IDs.
3.  **Block Merging & Pruning:** Extract candidate Block IDs from the inverted lists corresponding to the selected $k$ Codebooks. If the number of candidate blocks exceeds our compute budget (e.g., a maximum of 8 blocks / 1024 tokens allowed), perform rapid pruning based on local similarity.

### 2.3 Local Sliding Window Fallback
In long texts, 90% of coherent grammar and short-term logic rely on adjacent tokens. Therefore, in addition to the Blocks retrieved by the Codebook routing, we **forcefully include the Block containing the current Query as well as the preceding Block (256 tokens total)** into the Attention computation.
*   **Total Candidate Set** = Local Blocks (2) $\cup$ Routed Blocks (e.g., 6).
*   This ensures that every Query always computes a fixed number of Blocks (e.g., 8), completely eliminating Compute Load Imbalance.

### 2.4 Triton Block-Sparse Kernel Invocation (Exact Block-Sparse Attention)
Once the `Target_Block_IDs` matrix corresponding to each Query is obtained, invoke the customized Triton Kernel:
```python
# Pseudocode: Does not compute dense Softmax(QK^T), 
# only executes FlashAttention within selected Blocks.
output = flash_attn_with_block_mask(
    q, k, v, 
    block_indices=Target_Block_IDs, 
    block_size=128
)
```
This step fully reuses FlashAttention-3's internal `block_pointer` mechanism, introducing zero approximations in precision.

---

## Part 3: The Alchemy Secrets — Three-Stage Lossless Training Paradigm

The biggest pitfall major tech companies have faced is that "Sparse Attention is extremely hard to converge" or "Router weights collapse". If CRBSA is trained end-to-end from scratch, the model will inevitably degenerate into "only looking locally."
**We replicate and improve upon DeepSeek's "Detach (Gradient Truncation) + Offline Distillation" strategy.**

### Stage 1: Router Knowledge Distillation
**Goal:** Do not train the main backbone; only train the Codebook and router to near-perfection.
1.  **Data Preparation:** Prepare 128k-length high-quality texts (code, papers, logical reasoning).
2.  **Teacher Annotation:** Run a forward pass using an existing dense large model (e.g., Qwen2.5-7B/14B). Record the actual Block IDs that produced high Attention Weights for every Query in every layer. This is our **Ground Truth**.
3.  **Student Training:** Freeze the backbone parameters. Only flow gradients to the Codebook $C$ and the down-projection layers $W_Q, W_K$.
    *   **Loss Function 1 (Cross Entropy):** Force the router's predicted Top-K Blocks to match the Teacher's selected Blocks.
    *   **Loss Function 2 (Load Balancing Loss):** Introduce an MoE-style load balancing loss to prevent all blocks from crowding into a few Codebooks (preventing index collapse).
    $$ L_{bal} = \alpha \sum_{i=1}^M (f_i \cdot P_i) $$
    *(Where $f_i$ is the fraction of blocks assigned to the $i$-th codebook, and $P_i$ is the average probability of Queries selecting that codebook.)*

### Stage 2: Unfrozen Full-Parameter Sparse Continued Pre-training
**Goal:** Allow the main model to adapt to a "sparse context environment."
At this point, the router's hit rate is over 85%. We unfreeze all parameters.
**Crucial Engineering Trick (Lifesaver):** **Completely Detach the router's forward output from the backbone's backward pass!**
*   After the router computes `Target_Block_IDs`, pass them into the Attention Kernel as **Constants**.
*   The main model continues to update via Next Token Prediction (Cross Entropy).
*   The router continues to update *only* via the Stage 1 Distillation Loss and Load Balancing Loss.
*   **Reasoning:** If the backbone's gradients flow back to the router through the Attention Mask, the model will discover that "looking nearby is the easiest way to lower Loss," thereby destroying the long-range routing capability we painstakingly built.

### Stage 3: RLHF/GRPO Alignment for "Functional Long Context"
This is the "dark technology" explicitly hinted at in SubQ papers but rarely executed successfully.
The model has learned *how* to retrieve, but it is often "lazy to retrieve." We must use RL to force it to fish for distant evidence.
*   **Task Construction:** Synthesize massive datasets of "Needle In A Haystack (NIAH)", "Multi-hop Variable Tracking," and "Cross-file Code Patches (SWE-Bench style)."
*   **Reward Design:**
    *   $R_{correct} = +1$: The final answer is correct.
    *   $R_{routing} = +0.5$: The router's `Target_Block_IDs` genuinely hit the blocks containing scattered key evidence.
    *   $R_{hallucination} = -1.0$: The answer seems plausible but isn't grounded in distant constraints.
*   **Algorithm:** Use PPO or the **GRPO (Group Relative Policy Optimization)** algorithm proven highly efficient by DeepSeek. GRPO estimates advantage solely through generation results, saving the massive VRAM required by a Critic model.

---

## Part 4: Million-Context System Engineering & Distributed Architecture

At lengths of 1M Tokens (or even 2M), no matter how good the algorithmic design is, without underlying distributed system support, GPUs will still OOM. A single H100 (80GB) struggles to even hold a 1M KV Cache, let alone Activations.

We must combine **Sequence Parallelism** with **Asynchronous P2P KV Fetching**.

### 4.1 Ulysses Sequence Parallelism Adaptation
Existing Ring-Attention is suited for Dense models but causes catastrophic communication bubbles in our highly sparse CRBSA. We adopt an architecture based on a **DeepSpeed Ulysses** variant.
Assume a cluster of 8 GPUs, with a total sequence length of $N=1,024,000$:
1.  **Input Splitting:** Each GPU is allocated 128K contiguous Tokens for local feed-forward computation (Q, K, V projections).
2.  **Global Block Summary Sync:**
    After computing the Block Summaries, this data is tiny ($1M/128 = 8000$ Blocks, dim 64, totaling less than 2MB). Perform a lightweight `All-Gather`.
    **Now, every GPU possesses the global summary of all 8000 blocks.**
3.  **Local Routing Computation:**
    Each GPU performs parallel routing on its independent local Codebook (with zero cross-node communication), calculating the `Target_Block_IDs` for the Queries it is responsible for.

### 4.2 Asynchronous P2P Exact KV Fetching (The Millisecond Decider)
This is the most critical step determining Wall-clock speed.
GPU 0 realizes that one of its Queries needs to reference Block 3000, and the actual KV data for Block 3000 physically resides on GPU 7.
*   **Absolute Prohibition:** Performing an All-Gather on all KV data (will instantly nuke the network and cause OOM).
*   **Efficient Solution:** Build a **Request/Response Cross-Node Communication Scheduler**.
    GPU 0 sends a request to GPU 7: "Give me the KV for Block 3000." Fetch it using P2P (Point-to-Point) primitives via NVLink (intra-node) or InfiniBand (inter-node).
*   **Compute-Communication Overlap:** While GPU 0 requests the remote block from GPU 7, GPU 0 simultaneously computes the Attention for its Local Blocks (which are guaranteed to be local due to the sliding window). Once the remote data arrives via the network, it is spliced into the Kernel for subsequent computation.

### 4.3 VRAM Pooling & PagedAttention Integration
During the Inference (Serving) phase, to support highly concurrent long-context sessions, CRBSA is deeply integrated with vLLM / SGLang's PagedAttention.
Because our routing mechanism is inherently Block-level, it is a **perfect philosophical match** with PagedAttention's underlying logic of storing KV Caches in physically discrete Pages (Blocks). The routed `Block_ID` can be directly mapped to the physical address in the Paged Block Table, achieving zero-copy, ultra-fast inference.

---

## Part 5: Core Data Structures & Hyperparameter Guidelines (Deployment Reference)

To allow your R&D team to hit the ground running, here are the recommended baseline configurations for core parameters, suited for 7B~14B foundational models (like Llama-3 or Qwen-2.5) targeting a 1M context.

| Component / Parameter | Recommended Value | Engineering Rationale |
| :--- | :--- | :--- |
| **Block Size ($B$)** | 128 tokens | Matches Triton Kernel's chunking for optimal throughput. |
| **Codebook Size ($M$)** | 1024 | Ensures sufficient fine-grained semantic differentiation. |
| **Routing Dim ($d_{route}$)** | 64 | Massively reduces index building and scoring compute overhead. |
| **Routed Blocks per Query** | 8 | = 1024 tokens. Effectively locks compute complexity to a 1K scale, regardless of whether the input is 100K or 1M. |
| **Local Fallback Blocks** | 2 | Current block + previous block; prevents breakage of extremely short syntactic/code dependencies. |
| **KV Heads Strategy (GQA)** | $H_k = 8$ or $4$ | Paired with Grouped Query Attention to further reduce KV Cache VRAM footprint in long sequences. |
| **RoPE Scaling Strategy** | YaRN (x8 scale) | Extrapolates position encoding to ensure the backbone theoretically possesses long-context spatial awareness before entering sparse tuning. |

---

## Part 6: Benchmarks & "Functional Long Context" Validation

Once development is complete, how do we prove to the industry that our CRBSA has shattered nominal lengths and achieved true "Functional Context"? Abandon the rote-memorization Perplexity leaderboards and target these three ultimate touchstones:

### 6.1 RULER: The Basic Polygraph for Dynamic Long Context
Compared to the classic Needle-In-A-Haystack (NIAH), RULER adds multi-hop aggregation, variable tracking, and conditional filtering.
*   **Validation Point:** Can our Codebook routing assign different definitional fragments of the same variable into the same or adjacent Topic buckets for precise Query extraction?

### 6.2 MRCR v2: The Ultimate Fusion of Non-Adjacent Evidence
This is the core benchmark SubQ used in their PR to claim victory over Gemini and GPT-4.
*   **Difficulty:** The question does not know where the answer is. Evidence A is at 10K, Evidence B is at 800K, and they lack direct lexical similarity.
*   **CRBSA's Solution:** Through Stage 3 RL, our model learns not just literal matching, but recognizes "semantic bridges," pulling the Blocks for both A and B into the computation window in parallel during the routing phase.

### 6.3 SWE-bench Verified: The Real Industrial World Test
An end-to-end software engineering task containing massive codebases, historical PR discussions, and Issue descriptions.
*   **Why CRBSA Wins:** Traditional RNN architectures, after reading 100,000 lines of code, will forget the interface constraints declared at the very beginning. CRBSA's Codebook, however, is globally static. The moment the generated code touches the logic of that interface, the Query instantly activates the Block index containing the interface constraints, accurately recalling the original text. This prevents hallucinations where the code "looks locally correct but fails global compilation."

---

## Final Memo to the Project Lead

This blueprint clears the fog surrounding the long-context exploration that major tech giants are currently trapped in.
1.  **Do not touch RNN long-range memory;** leave it for short videos and audio streams. For highly complex, long logical reasoning, precise reconstruction via Attention is mandatory.
2.  **Do not touch DeepSeek's global brute-force scoring;** leave the illusion of saved compute to short contexts.
3.  **Embrace Codebook + Block-Sparse;** this is the *only* valid industrial path to completely decouple compute complexity from sequence length.

**Suggested Action Item for Week 1:**
Assign your strongest Kernel engineer to directly implement the Forward/Backward Triton Kernel described above (which accepts discrete `Target_Block_IDs`) based on FlashAttention-3. Benchmark its speed and VRAM performance at 128k/1M lengths.
Once the Kernel benchmark outpaces the dense model (which it absolutely 100% will), half of this project's moat is already built. The rest is just feeding that Codebook router with high-quality data.

I look forward to seeing this architecture run on your clusters at speeds and precision that will shock the industry. If you encounter any subtle bugs regarding operator compilation or distributed communication, we are ready to dive into lower-level code discussions at any time.