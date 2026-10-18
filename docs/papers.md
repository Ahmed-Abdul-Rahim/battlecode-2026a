# Papers registry

Living reference list for the team. Every technique we adopt should cite an ID from this file in its PR or issue.

**Bar:** Q1 journals (SCImago SJR) or CORE/ICORE A* / A conferences. Anything below the bar stays listed but is flagged so nobody mistakes it for a vetted source.

**Columns:** `Rank` is the venue rank (ICORE2026 for conferences, SJR 2025 best quartile for journals). `Verified` is the date the venue *and* rank were checked online; `pending` means it was taken from memory and still needs checking. `Status` is one of `used`, `planned`, `deferred`, `rejected`, `reference`.

## How to add a paper

1. Take the next free ID in the right section.
2. Fill in every column. Check the venue rank on [portal.core.edu.au](https://portal.core.edu.au/conf-ranks/) (conferences) or [scimagojr.com](https://www.scimagojr.com/) (journals) and set `Verified` to today's date.
3. Link the code path or issue in "Where used" once something depends on it.

## Findings

- **No A*/A/Q1 paper applies unsupervised domain adaptation to MOBA games directly** (searched 2026-10-18). We compose UDA methods (P-010, P-011) with MOBA generalisation work (P-004, P-005). Source domain = Honor of Kings data, target = cambc self-play; adaptation happens in the shared feature schema (`ml/features/schema.yaml`) and per match through draft-memory FiLM parameters (P-032).
- **The best-known MOBA feature-engineering study (P-007) is in a Q2 journal.** Baseline features therefore come from the observation designs of the A* Honor of Kings papers (P-001, P-003, P-004), with P-007 used only for the net-worth / team-sum idea.
- **Feature selection follows the backward wrapper line** (P-020, P-021), with an mRMR pre-ranking (P-022), a fuel-cost penalty (P-026) and stability-driven group sizing (P-024).

## MOBA and game AI

| ID | Paper | Venue | Rank | Verified | Key idea | Where used | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P-001 | Ye et al., [Mastering Complex Control in MOBA Games with Deep Reinforcement Learning](https://www.researchgate.net/publication/338115939_Mastering_Complex_Control_in_MOBA_Games_with_Deep_Reinforcement_Learning) | AAAI 2020 | A* | pending | Action masks, target-attention head, unit-feature observation design | Feature schema baseline; target-selection head | planned |
| P-002 | Ye et al., [Towards Playing Full MOBA Games with Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/06d5ae105ea1bea4d800bc96491876e9-Abstract.html) | NeurIPS 2020 | A* | 2026-10-18 | Curriculum self-play, teacher→student policy distillation, MCTS drafting over a large hero pool | Distillation pipeline; draft scorer | planned |
| P-003 | Wu et al., [Hierarchical Macro Strategy Model for MOBA Game AI](https://www.researchgate.net/publication/335808250_Hierarchical_Macro_Strategy_Model_for_MOBA_Game_AI) | AAAI 2019 | A* | pending | Macro (where to go) vs micro (how to fight) split | Meta-gate + MoE hierarchy | used |
| P-004 | Wei et al., [Honor of Kings Arena: an Environment for Generalization in Competitive RL](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4dbb61cb68671edc4ca3712d70083b9f-Abstract-Datasets_and_Benchmarks.html) | NeurIPS 2022 (D&B) | A* | 2026-10-18 | Generalisation across controlled heroes and opponents is hard; evaluate on held-out heroes | Held-out map/opponent gate protocol | used |
| P-005 | Qu et al., [Hokoff: Real Game Dataset from Honor of Kings and its Offline RL Benchmarks](https://arxiv.org/abs/2408.10556) ([code](https://github.com/tencent-ailab/hokoff)) | NeurIPS 2023 (D&B) | A* | 2026-10-18 | Offline HoK 1v1 / 3v3 trajectories, no gamecore licence needed | UDA source domain | planned |
| P-006 | Vinyals et al., [Grandmaster level in StarCraft II using multi-agent RL](https://www.nature.com/articles/s41586-019-1724-z) | Nature 2019 | Q1 | pending | League training with exploiter agents; memory under fog of war | Teammate's Qwen harness (exploiter bots) | planned |
| P-007 | Hodge et al., [Win Prediction in Multi-Player Esports: Live Professional Match Prediction](https://pure.york.ac.uk/portal/en/publications/win-prediction-in-multi-player-esports-live-professional-match-pr/) | IEEE Trans. Games 2021 | **Q2 (below bar)** | 2026-10-18 | Engineered live features: net worth = gold + item value, team sums and differences | Economy feature group | reference |
| P-008 | Liu et al., [Mini Honor of Kings: A Lightweight Environment for MARL](https://arxiv.org/abs/2406.03978) | arXiv 2024 | **unranked** | 2026-10-18 | Laptop-scale HoK environment | Possible alternative UDA source | reference |

## Domain adaptation and generalisation

| ID | Paper | Venue | Rank | Verified | Key idea | Where used | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P-010 | Ganin et al., [Domain-Adversarial Training of Neural Networks](https://jmlr.org/papers/v17/15-239.html) | JMLR 2016 | Q1 | 2026-10-18 | Gradient-reversal layer + domain classifier aligns source and target features without target labels | `ml/uda/dann.py` | planned |
| P-011 | Higgins et al., [DARLA: Improving Zero-Shot Transfer in RL](https://arxiv.org/abs/1707.08475) | ICML 2017 | A* | 2026-10-18 | Learn disentangled, domain-invariant state features before learning to act | Egocentric, size-normalised feature schema | used |
| P-012 | Cobbe et al., [Leveraging Procedural Generation to Benchmark RL](https://proceedings.mlr.press/v119/cobbe20a.html) | ICML 2020 | A* | 2026-10-18 | Measure the generalisation gap on held-out levels | `maps/heldout/` gate split | used |
| P-013 | Tobin et al., [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) | IROS 2017 | A | pending (rank 2026-10-18) | Train across randomised environments so the target looks like one more variation | Map generation for training | planned |

## Feature engineering and selection

| ID | Paper | Venue | Rank | Verified | Key idea | Where used | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P-020 | Kohavi & John, [Wrappers for Feature Subset Selection](https://doi.org/10.1016/S0004-3702(97)00043-X) | Artificial Intelligence 1997 | Q1 | 2026-10-18 | Wrapper search scored by the downstream model; backward elimination keeps interacting features | `ml/features/select.py` | used |
| P-021 | Pudil, Novovičová & Kittler, [Floating Search Methods in Feature Selection](https://doi.org/10.1016/0167-8655(94)90127-9) | Pattern Recognition Letters 1994 | Q1 (CV & PR); Q2 (AI) | 2026-10-18 | Sequential *floating* backward selection: after each removal, try re-adding | `ml/features/select.py` | used |
| P-022 | Peng, Long & Ding, [Feature Selection Based on Mutual Information: Max-Dependency, Max-Relevance, Min-Redundancy](https://doi.org/10.1109/TPAMI.2005.159) | IEEE TPAMI 2005 | Q1 | 2026-10-18 | mRMR filter; pairs well with a wrapper in two stages | Ranking used to split groups | used |
| P-023 | Guyon & Elisseeff, [An Introduction to Variable and Feature Selection](https://jmlr.org/papers/v3/guyon03a.html) | JMLR 2003 | Q1 | pending | Filter / wrapper / embedded taxonomy; nested-subset methods | `docs/feature-selection.md` | reference |
| P-024 | Nogueira, Sechidis & Brown, [On the Stability of Feature Selection Algorithms](https://jmlr.org/papers/v18/17-514.html) | JMLR 2018 | Q1 | 2026-10-18 | Stability index with confidence intervals across resamples | Stop / refine rule in group search | used |
| P-025 | Meinshausen & Bühlmann, [Stability Selection](https://doi.org/10.1111/j.1467-9868.2010.00740.x) | J. R. Stat. Soc. B 2010 | Q1 | pending | Keep features chosen consistently across subsamples | Bootstrap reruns in group search | reference |
| P-026 | Nan, Wang & Saligrama, [Feature-Budgeted Random Forest](https://proceedings.mlr.press/v37/nan15.html) | ICML 2015 | A* | 2026-10-18 | Minimise error under a test-time feature-cost budget | Fuel-cost penalty λ·Σcost | used |

## Communication, conditioning and memory-as-parameters

| ID | Paper | Venue | Rank | Verified | Key idea | Where used | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P-030 | Foerster et al., [Learning to Communicate with Deep Multi-Agent RL](https://arxiv.org/abs/1605.06676) | NeurIPS 2016 | A* | pending (arXiv checked) | DIAL: discretise/regularise unit, so discrete messages can be trained with gradients | Learned `TeamChannel` (8-byte memory) | planned |
| P-031 | Wang et al., [Learning Efficient Multi-agent Communication: An Information Bottleneck Approach](http://proceedings.mlr.press/v119/wang20i/wang20i.pdf) | ICML 2020 | A* | 2026-10-18 | Bandwidth limit treated as a message-entropy budget | 6 message bytes in shared memory | planned |
| P-032 | Perez et al., [FiLM: Visual Reasoning with a General Conditioning Layer](https://ojs.aaai.org/index.php/AAAI/article/view/11671) | AAAI 2018 | A* | 2026-10-18 | Feature-wise affine modulation (γ, β) from conditioning input | θ_match FiLM block in draft memory | used |

## Champion representations

| ID | Paper | Venue | Rank | Verified | Key idea | Where used | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P-040 | Lee et al., [DraftRec: Personalized Draft Recommendation for Winning in MOBA Games](https://arxiv.org/abs/2204.12750) | WWW 2022 | A* | 2026-10-18 | Champion and player embeddings, match network predicting outcome from the draft | Draft scorer; synergy and counter terms | planned |
| P-041 | Rabinowitz et al., [Machine Theory of Mind](https://proceedings.mlr.press/v80/rabinowitz18a/rabinowitz18a.pdf) | ICML 2018 | A* | 2026-10-18 | Character embedding learned from observed behaviour | Champion-vector fine-tuning from usage | planned |
| P-042 | Grover et al., [Learning Policy Representations in Multiagent Systems](https://mlanthology.org/icml/2018/) | ICML 2018 | A* | 2026-10-18 | Embeddings of agents learned from interaction episodes | Champion-vector fine-tuning from usage | planned |

## Efficient inference, mixture of experts, distillation

| ID | Paper | Venue | Rank | Verified | Key idea | Where used | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P-050 | Shazeer et al., [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) | ICLR 2017 | A* | pending | Top-k gating: only the selected experts are computed | Top-1 router over heuristic experts | used |
| P-051 | Rusu et al., [Policy Distillation](https://arxiv.org/abs/1511.06295) | ICLR 2016 | A* | pending | Compress a large policy into a small one | `ml/distill/` | planned |
| P-052 | Jacob et al., [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) | CVPR 2018 | A* | pending | int8 weights and activations with int32 accumulation; quantisation-aware training | `bc-brain` int8 kernel | used |
| P-053 | Kaya et al., [Shallow-Deep Networks: Understanding and Mitigating Network Overthinking](https://arxiv.org/abs/1810.07052) | ICML 2019 | A* | pending | Early exits: spend computation only when the input needs it | Meta-gate (heuristic vs MoE) | used |
| P-054 | Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | NeurIPS 2014 workshop | **unranked** | pending | Soft-target distillation | `ml/distill/` | reference |

## Pathfinding

| ID | Paper | Venue | Rank | Verified | Key idea | Where used | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P-060 | Harabor & Grastien, Online Graph Pruning for Pathfinding on Grid Maps (JPS) | AAAI 2011 | A* | pending | Jump point search on uniform-cost grids | Candidate for tier-3 long-range search | planned |
| P-061 | Sturtevant et al., Memory-Based Heuristics for Explicit State Spaces | IJCAI 2009 | A* | pending | Differential / true-distance heuristics from precomputed fields | Static fields as the A* heuristic | planned |
| P-062 | Koenig & Likhachev, D* Lite | AAAI 2002 | A* | pending | Incremental repair instead of replanning from scratch | Tier-2 local repair | used |
| P-063 | Sharon et al., Conflict-Based Search for Optimal Multi-Agent Pathfinding | Artificial Intelligence 2015 | Q1 | pending | Optimal multi-agent paths | Rejected: no shared vision, 8-byte memory, fuel cost | rejected |
| P-064 | Silver, Cooperative Pathfinding (WHCA*) | AIIDE 2005 | **below bar** | pending | Windowed space-time reservations | Rejected for the same reasons as P-063 | rejected |

## Sequence models (deferred by the team)

| ID | Paper | Venue | Rank | Verified | Key idea | Where used | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P-070 | Gu et al., [Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/abs/2111.00396) | ICLR 2022 | A* | pending | Structured state-space layers | — | deferred |
| P-071 | Dao & Gu, [Transformers are SSMs (Mamba-2)](https://proceedings.mlr.press/v235/dao24a.html) | ICML 2024 | A* | 2026-10-18 | Structured state-space duality; faster selective SSM layer | — | deferred |
| P-072 | Gu & Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | COLM 2024 | **unranked (new venue)** | pending | Input-dependent (selective) SSM | — | deferred |
| P-073 | Orvieto et al., [Resurrecting Recurrent Neural Networks for Long Sequences (LRU)](https://arxiv.org/abs/2303.06349) | ICML 2023 | A* | pending | Diagonal linear recurrence; cheap per-step update | Candidate in-bot belief memory | deferred |
| P-074 | Lv et al., [Decision Mamba: A Multi-Grained State Space Model with Self-Evolution Regularization for Offline RL](https://proceedings.neurips.cc/paper_files/paper/2024/file/288b63aa98084366c4536ba0574a0f22-Paper-Conference.pdf) | NeurIPS 2024 | A* | 2026-10-18 | Mamba for offline RL trajectories | — | deferred |
| P-075 | Chen et al., [Decision Transformer](https://arxiv.org/abs/2106.01345) | NeurIPS 2021 | A* | pending | RL as sequence modelling | — | deferred |

## LLM-driven self-improvement

| ID | Paper | Venue | Rank | Verified | Key idea | Where used | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P-080 | Romera-Paredes et al., [Mathematical Discoveries from Program Search with Large Language Models (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) | Nature 2024 | Q1 | 2026-10-18 | LLM proposes programs; an automatic evaluator keeps only the improvements | `bctools gate` + `/improve` loop | used |
| P-081 | Ma et al., [Eureka: Human-Level Reward Design via Coding LLMs](https://arxiv.org/abs/2310.12931) | ICLR 2024 | A* | 2026-10-18 | LLM writes code, reads evaluation feedback, reflects and rewrites | `/improve` loop | used |
| P-082 | Shinn et al., [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | NeurIPS 2023 | A* | pending | Written self-reflection stored as memory between attempts | Experiment issues record what was tried | used |
| P-083 | Hu et al., [Multimodal LLM-assisted Evolutionary Search for Programmatic Control Policies (MLES)](https://arxiv.org/abs/2508.05433) | arXiv 2025 | **unranked** | 2026-10-18 | Visual feedback (rendered behaviour) guides LLM policy edits | MCP `render_frame` in the loop | reference |
