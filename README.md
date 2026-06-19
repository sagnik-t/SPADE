# SPADE — Structure-Preserving Adversarial Data Generation

SPADE is a structure-preserving generative framework for synthetic **collaborative filtering (CF)** data. It produces new pseudo-users and pseudo-items whose interactions respect the latent preference geometry, sparsity topology, and discrete rating space of a real dataset — and it ships two structure-aware diagnostics, **PGPS** and **NDI**, for checking that the structure actually survived. The accompanying paper is *"Beyond Marginals: Structure-Aware Evaluation and Generation of Synthetic CF Data."*

The guiding idea: synthetic CF data should be **generated and judged in terms of latent preference geometry**, not just marginal statistics (rating histograms, degree distributions) or downstream task scores. A dataset can match the marginals perfectly and still carry a degraded collaborative signal. SPADE constrains generation in latent space and enforces every domain constraint *by construction*, so there is no post-hoc rounding, clipping, or quantization.

---

## Why this exists

Existing synthetic-CF methods are typically evaluated two ways, and both miss the point:

- **Distributional similarity** (KL, Earth Mover's Distance, hypothesis tests) only looks at marginal or low-order statistics. It says nothing about higher-order interaction structure.
- **Downstream performance** (train on synthetic, test on real) entangles data quality with model capacity, and a strong recommender can paper over real structural defects.

Neither asks whether the synthetic data preserves *preference structure* — the relative relationships (similar users, items with overlapping audiences, local preference orderings) that make collaborative filtering work in the first place. SPADE targets exactly that, on both the generation and the evaluation side.

### Design requirements

A principled CF generator must satisfy several constraints that pull in different directions. SPADE addresses each one structurally rather than as a correction step:

| Requirement | How SPADE satisfies it |
|---|---|
| **Controlled expansion** — bounded `U' > U`, `I' > I`, no identity leakage | Sampling fixed counts of latent entities from learned generators |
| **Rating-space invariance** — ratings stay exactly in the original discrete set | Categorical (softmax) rating decoder; ratings are discrete by construction |
| **Sparsity preservation** — match real interaction density and long tails | A learned probabilistic interaction gate trained with negative sampling |
| **Fidelity ↔ generalization** — keep structure, but generate novel interactions | Generation happens in latent space, not by copying real entities |
| **Factorized user/item modeling** — reflect the bipartite structure | Independent latent generators for users and items |

---

## How it works

SPADE factorizes the joint distribution as

```
p(u, i, r) = p(z_u) · p(z_i) · p(g_ui | z_u, z_i) · p(r | z_u, z_i, g_ui = 1)
```

and realizes it as a three-stage pipeline. Stages I and II are trained; Stage III is inference-only.

```
(u, i, r)  →  (z_u, z_i)  →  (z̃_u, z̃_i)  →  (g̃_ui, r̃_ui)
  real        Stage I        Stage II         Stage III
  data       encode real    sample new      decode interactions
             entities       entities        + ratings
```

### Stage I — Representation learning

Deterministic encoders `E_u`, `E_i` (embedding lookups followed by MLPs) map users and items to latent vectors `z_u, z_i ∈ R^d`. Determinism is deliberate: it keeps the latent geometry sharp rather than over-smoothed. Two decoders sit on top:

- **Interaction gate** `p(g_ui = 1 | z_u, z_i) = σ(f_g(z_u, z_i))` models *whether* an interaction exists, so sparsity is learned in its own right instead of being a side effect of rating prediction. It is trained with **negative sampling** to handle the missing-not-at-random (MNAR) nature of CF data: an unobserved pair usually means the user never saw the item, not that they disliked it. For each observed positive, `n_neg` unobserved pairs per user are sampled uniformly as negatives (a BPR-style objective), rather than treating every empty cell as a true negative.
- **Rating decoder** `p(r | z_u, z_i) = Categorical(softmax(f_r(z_u, z_i)))` emits logits over the discrete rating set, guaranteeing valid ratings by construction.

The stage minimizes `L_rep = L_gate + L_rating + λ(‖z_u‖² + ‖z_i‖²)`, where the rating loss is computed only on observed interactions and the embedding-norm penalty prevents degenerate solutions. After training, encoders and decoders are **frozen**; the learned `Z_u`, `Z_i` become the empirical latent samples for Stage II.

### Stage II — Latent distribution modeling

Two **independent WGAN-GP** generators `G_u: ε → z_u` and `G_i: ε → z_i` (with `ε ~ N(0, I)`) are trained to match the empirical Stage I latent distributions, enabling controlled expansion without memorizing real identities. WGAN-GP (Wasserstein loss + gradient penalty) is used for training stability on continuous embedding spaces, plus a **moment-matching** regularizer encouraging mean/covariance alignment between real and generated latents.

Training the two generators independently is a deliberate simplification (analogous to a product-of-experts assumption): it trades some joint fidelity for stability and scalability, relying on the shared interaction decoder to couple users and items at synthesis time. The cost of this assumption is examined in the *joint vs. factorized* ablation. A **joint** variant — one WGAN-GP over the concatenated `[z_u; z_i]` of observed interactions — is implemented for that comparison.

### Stage III — Synthetic interaction generation

The practitioner picks target sizes `U' = α·U` and `I' = β·I`. SPADE samples `U'` synthetic user latents and `I'` synthetic item latents from `G_u`, `G_i`, then decodes interactions.

Scoring all `U' × I'` pairs through the gate is quadratic (≈83M forward passes for ML-1M at α=β=2), so SPADE uses **sparse candidate sampling**: for each synthetic user it retrieves the top-`C` nearest synthetic items by cosine similarity via **approximate nearest-neighbor (ANN)** search, and evaluates the gate only over those candidates. `C = ⌈I' · ρ · γ⌉`, where `ρ` is the real dataset's observed sparsity and `γ ≥ 1` (default 5) is an oversampling buffer ensuring enough interactions survive Bernoulli sampling. Existence is sampled `g̃_ui ~ Bernoulli(p)`, and where `g̃_ui = 1` a rating is drawn from the categorical decoder.

The result is a synthetic dataset where entity counts are fixed by sampling, ratings are discrete by categorical decoding, sparsity is enforced by the gate, and generalization comes from latent-space sampling — with no post-hoc correction anywhere.

---

## Evaluation metrics

All structure-aware metrics operate in a **shared reference latent space** fixed by a recommender (matrix factorization and/or neural CF) trained on the real data. Since synthetic and real entities share no index axis, synthetic entities are placed into this space by **transductive inference**: a ridge least-squares linear map `W = (ZᵀZ + λI)⁻¹ZᵀR` is fit from Stage I latents to the reference embeddings on real entities, then applied to synthetic latents. Because the metrics are conditioned on the reference model, their absolute values are **relative diagnostics** — compare generators within an experiment, not across reference models.

- **PGPS — Preference Geometry Preservation Score** *(the novel, headline metric)*. Item-level. For each synthetic item it finds its nearest real *anchor*, then measures the overlap between the synthetic item's real-item neighborhood and the anchor's. Mean overlap ∈ [0, 1]. Near 0 means no geometric relation to the real manifold; near 1 means trivial copying/memorization. The healthy target is comfortably above the random baseline `k/|I|` and below the copy ceiling — so the headline quantity is the **lift**, `PGPS − k/|I|`.
- **NDI — Neighborhood Distortion Index**. User-level complement. The fraction of a real user's `k` nearest neighbors that synthetic users displace once added to the pool. As implemented it is **exactly the synthetic intrusion rate** (the paper states this identity and does *not* claim NDI as a novel metric). Read it two-sided: moderate is healthy; very high means crowding/memorization-level proximity; very low means synthetic users fail to integrate with the real manifold.
- **TS-TR — Train-on-Synthetic, Test-on-Real utility**. Each recommender is scored on a held-out split of its *own* universe (synthetic-trained on a synthetic holdout, real-trained on the real test split), and reported as the ratio `RelPerf`. `RelPerf ≈ 1` means the synthetic data carries as much learnable collaborative signal as the real data. Standard ranking metrics: Recall@k, NDCG@k, MAP.
- **Latent Wasserstein (W₂)**. Global distribution alignment between real and synthetic user/item embeddings — catches support mismatch and large-scale geometric distortion that NDI can miss.
- **Degree-distribution KS**. Kolmogorov–Smirnov distance between real and synthetic user/item degree distributions — a sanity check against unrealistic densification or collapse.

PGPS is the metric the framework leans on while developing a model; the others corroborate it. Headline empirical findings (full numbers in the paper, on **MovieLens-100K** and **MovieLens-1M**): SPADE reproduces real interaction **degree structure** far more faithfully than strong baselines while keeping preference geometry clear of trivial copying, with downstream utility competitive with the strongest baselines.

---

## Repository layout

```
src/spade/
  config/        Dataclass configs + CLI parsing (one config per stage; ExperimentConfig composes them)
  data/          Dataset loaders (ML-100K/ML-1M, on-demand download), interaction store, k-core filtering, leakage-safe splits, negative sampling
  models/        Encoders, interaction gate, rating decoder, WGAN-GP generators & critics, losses, the SPADE model
  training/      RepresentationTrainer, GenerativeTrainer, JointGenerativeTrainer, checkpoint save/load
  synthesis/     Stage III: ANN candidate sampling, the synthesizer, the SyntheticDataset container
  eval/          PGPS, NDI, reference space + transductive map, distributions (W₂, degree-KS), downstream TS-TR, the eval suite
  baselines/     random, marginal, noise_mf (noise-perturbed MF), ganrs, vae, plus the SPADE adapter; BASELINE_REGISTRY
  experiments/   Per-cell pipeline, the resumable run matrix, ablations, aggregation, paper-table export
  utils/         Env loading (.env), logging, seeding, W&B helpers, JAX key helpers

scripts/         Command-line entry points (see Usage)
tests/           pytest suite (unit + reproducibility + an `env` hardware-check marker)
data/            Downloaded datasets and cached splits (git-ignored)
results/         matrix/ per-cell JSON + stage cache; tables/ exported CSV/Markdown/LaTeX
```

### Baselines

`BASELINE_REGISTRY` exposes five comparison generators, all sized to the *same* `U'`/`I'` and target density as SPADE so only the generation *mechanism* differs:

| Name | Mechanism |
|---|---|
| `random` | Uniform random interactions at the target density |
| `marginal` | Marginal-matching (reproduces rating/degree marginals) |
| `noise_mf` | Matrix factorization + isotropic embedding noise |
| `ganrs` | GANRS: DeepMF embeddings → vanilla GAN over interaction tuples → K-Means identifier recovery |
| `vae` | VAE over interaction tuples → K-Means recovery |

---

## Installation

Python **3.12** (`>=3.12`), managed with Poetry. The project targets a CUDA 12 GPU for full runs but ML-100K runs fine on CPU.

```bash
# 1. Install dependencies (creates the project virtualenv)
poetry install

# 2. Configure environment
cp .env.example .env        # then fill in W&B key / CUDA / JAX settings as needed
```

Core stack: **JAX** (`jax[cuda12]`) + **Flax** + **Optax** for models and training, **faiss-cpu** for ANN candidate search, **scikit-learn**/**SciPy** for evaluation, **pandas** for data, **Weights & Biases** for run tracking, **python-dotenv** for config.

`.env` (copied from `.env.example`, git-ignored) is loaded automatically at every script entry point. Key variables: `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_MODE` (`online`/`offline`/`disabled`), `WANDB_PROJECT`, `CUDA_VISIBLE_DEVICES`, `XLA_PYTHON_CLIENT_MEM_FRACTION`. Shell exports take precedence over `.env` values, so CI secrets are never clobbered.

---

## Usage

Every script shares the same nested-dataclass CLI: pass `--data.dataset`, `--seed`, `--representation.epochs`, `--synthesis.alpha`, `--wandb-mode disabled`, etc. Pass `--wandb-mode disabled` to turn off run tracking, or `offline` to log locally.

### Quickest path — full pipeline for one (dataset, seed)

`smoke_pipeline.py` trains both stages, synthesizes, runs the full evaluation suite, and asserts the invariants that must hold on a real run (artifact wiring, entity counts, rating support, no duplicate pairs, bit-for-bit determinism under a fixed seed, and PGPS sitting above its random baseline and below the trivial-copy ceiling). It prints a PASS/FAIL line per check and exits non-zero on any failure, so it doubles as a reproducibility gate.

```bash
poetry run python scripts/smoke_pipeline.py --data.dataset ml-100k --seed 42
# --skip-train  reuses existing stage exports and only re-checks synthesis + eval
```

### Step by step

```bash
# 1. Download, filter (k-core), split leakage-safely, and cache the splits
poetry run python scripts/prepare_data.py --data.dataset ml-100k --seed 42

# 2. Stage I — train representation; exports Z_u/Z_i + the gate/decoder + rating vocab
poetry run python scripts/train_representation.py --data.dataset ml-100k --seed 42 \
    --representation.epochs 100

# 3. Stage II — train the two WGAN-GP latent generators from the Stage I export
poetry run python scripts/train_generative.py --data.dataset ml-100k --seed 42 \
    --generative.epochs 500

# 4. Stage III — synthesize a discrete dataset at the chosen expansion ratios
poetry run python scripts/synthesize.py --data.dataset ml-100k --seed 42 \
    --synthesis.alpha 2 --synthesis.beta 2

# 5. Evaluate against the full metric suite -> evaluation_seed_42.json
poetry run python scripts/evaluate.py --data.dataset ml-100k --seed 42

# (optional) generate a single baseline for comparison
poetry run python scripts/run_baseline.py --baseline ganrs --data.dataset ml-100k --seed 42
```

### Full experiment matrix

`run_experiments.py` trains SPADE once per `(dataset, seed)`, runs SPADE plus baselines through the full suite across all requested seeds and ablations, and writes `mean ± std` tables. The matrix is **resumable** — finished cells are cached as JSON and skipped on re-run. Stage caches are keyed by a signature of the relevant config, so an ablation that changes a stage gets its own cache while ablations that only touch synthesis reuse the base stages.

```bash
poetry run python scripts/run_experiments.py \
    --datasets ml-100k --generators spade random marginal noise_mf ganrs vae \
    --seeds 0 1 2 3 4 \
    --ablations base alpha_1.5 alpha_3.0 latent_reg_off gating_off \
                joint_generator continuous_decoder
```

Re-export the paper tables (CSV + Markdown + booktabs LaTeX) from existing results without re-running anything:

```bash
poetry run python scripts/export_tables.py --results-dir results/matrix --out-dir results/tables
```

### Ablations available

`gating_off` (no interaction gate), `joint_generator` (single joint WGAN-GP over `[z_u; z_i]`), `continuous_decoder` (regress-and-round ratings instead of categorical), `latent_reg_off` (drop the embedding-norm penalty), and the expansion-ratio sweep (`alpha_1.5`, `alpha_3.0`, …).

---

## Configuration reference

Configs are plain dataclasses in `src/spade/config/configs.py`; `ExperimentConfig` composes them and every field is overridable from the CLI. Defaults follow the paper's experimental setup. Selected knobs:

- **Data** (`--data.*`): `dataset` (`ml-100k` | `ml-1m` | `amazon`*), `val_frac`/`test_frac` (0.1/0.1), `min_user_interactions`/`min_item_interactions` (5, k-core), `n_neg` (5 negatives per positive).
- **Representation** (`--representation.*`): `latent_dim` (64), encoder/gate/decoder hidden sizes, `lr` (1e-3), `epochs` (100), `l2_lambda` (1e-5), `early_stop_patience` (10), `continuous_decoder` (ablation flag).
- **Generative** (`--generative.*`): `noise_dim` (64), generator/critic hidden sizes, `n_critic` (5), `gp_lambda` (10.0), `moment_lambda` (1.0), `lr` (1e-4), WGAN-GP Adam betas (0.0, 0.9), `epochs` (500), `joint` (ablation flag).
- **Synthesis** (`--synthesis.*`): `alpha`/`beta` expansion ratios (2.0/2.0), `gamma` ANN oversampling buffer (5.0), `ann_metric` (cosine), `gating` (on; off = ablation).
- **Eval** (`--eval.*`): `reference_models` (`mf`, `ncf`), `topk` ([10, 20]), `neighbor_k` (10), `map_ridge` (1e-2, transductive map strength), `tstr_model` (`mf` | `ncf` | `bpr`), reference-model sizes/epochs.

`*` `amazon` is registered but deferred — the exact Reviews subset and core filter are not yet pinned, and that choice sets the sparsity `ρ` driving Stage III's candidate count, so it must be chosen deliberately. **MovieLens-100K** and **MovieLens-1M** are fully implemented with on-demand download.

---

## Testing

```bash
poetry run pytest                 # default suite (excludes hardware checks)
poetry run pytest -m env          # hardware/environment checks (CPU+GPU jax/flax/optax/faiss)
```

The suite covers config parsing, data/splitting, models, generative training (including the joint variant), synthesis (including continuous-decoder rating snapping), the evaluation metrics, the experiment matrix, and reproducibility. `pytest` is configured via `pyproject.toml` (`testpaths=tests`, `pythonpath=src`); the `env` marker is excluded from the default run.

---

## Reproducibility notes

Synthesis is **deterministic under a fixed seed** (verified bit-for-bit by the smoke pipeline). The default protocol runs **five seeds** (`0–4`) and reports `mean ± std`. Leakage-safe splits are materialized per seed before training. Reference-model conditioning means PGPS/NDI are reported under two reference models (MF and NCF) to show sensitivity. All run curves and per-cell metrics are logged to Weights & Biases unless disabled.

---

## Citation

If you use SPADE or its metrics, please cite the paper:

> Taraphdar, S. *Beyond Marginals: Structure-Aware Evaluation and Generation of Synthetic CF Data.*

## License

MIT — see `pyproject.toml`. Author: Sagnik Taraphdar (`code.sagnik@gmail.com`).
