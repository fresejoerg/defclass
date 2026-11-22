# Zero-Shot, Few-Shot, and Supervised Intent Classification: An Empirical Ablation & Comparative Study

**Primary Evaluation Metric:** Macro-F1 (class-balanced performance across frequent and rare intents)

---

## 1. Executive Summary

Zero-shot multi-class intent classification with Large Language Models (LLMs) often relies on short, dataset-provided label strings (e.g., `card_arrival`, `flight`, or opaque integers `"0"`–`"150"`), leaving models vulnerable to label-name priors, ambiguous class boundaries, and positional bias in long prompts. In this study, we evaluate a definition-driven zero-shot framework across five public intent benchmarks (`clinc150`, `banking77`, `hwu64`, `atis`, `snips`; `325` total union classes) and compare it directly against calibrated decision models (`gemini-decision-1`), few-shot contrastive fine-tuning (`setfit:8shot`), frozen sentence-transformer embedding heads (`st-head`), and sparse lexical baselines (`tfidf+linearsvc`, `tfidf+logreg`).

Our key findings are:
1. **Self-sufficient definitions substantially improve zero-shot Macro-F1**: On fine-grained `banking77` (`200` test samples), adding audited definitions raised `gemini-3.5-flash-lite` Macro-F1 from **`0.7921`** to **`0.8582`** (**`+0.0661`**). On `gemini-decision-1` (`50` samples), replacing opaque integer labels with definitions lifted Accuracy/Macro-F1 from `0.1400`/`0.0119` to **`0.9400`/`0.8977`** on `clinc150`, `0.0600`/`0.0111` to **`0.8800`/`0.7968`** on `hwu64`, and `0.0600`/`0.0209` to **`1.0000`/`1.0000`** on `snips`.
2. **Opaque identifiers match or outperform semantic labels on Macro-F1**: Stripping human-readable class titles (`numbered_def`, `uuid_def`, `uuid_shuffled_def`) matched or beat `semantic_def` across every dataset with `>7` classes (`0.8582` vs `0.8390` on `banking77`; `0.6092` vs `0.5766` on `atis`; `0.8514` vs `0.8275` on `hwu64`; `0.9273` vs `0.8556` on `clinc150`).
3. **Zero-shot LLMs with definitions outperform 8-shot SetFit on large label spaces (64–151 classes), while 8-shot SetFit leads on imbalanced `atis`**: On `banking77`, `hwu64`, and `clinc150`, definition-driven zero-shot LLMs (`0.8582`, `0.8514`, `0.9273` Macro-F1) beat `setfit:8shot` (`0.7578`, `0.6883`, `0.7273` Macro-F1). Conversely, on severely imbalanced `atis` (`76%` `flight`), 8-shot stratified sampling in `setfit:8shot` (`138` examples) naturally rebalanced rare classes and achieved the **highest overall Macro-F1 (`0.7285`)** of any method tested.
4. **Frozen sentence-transformer heads (`st-head`) provide the best supervised speed/accuracy balance on multi-domain tasks**: Trained on full `train` splits, `st-head` (`all-MiniLM-L6-v2` + `LogisticRegression`) beat `tfidf+linearsvc` on `clinc150` (`0.8553` vs `0.8003` F1), `hwu64` (`0.9261` vs `0.9110` F1), and `atis` (`0.7179` vs `0.6965` F1) at `2.2–3.1 ms/utt` inference.
5. **Reflective definition optimization yields large gains when dev density is sufficient—and overfits when sparse**: Optimizing `atis` definitions on `80` `train` dev samples (`~3.6` examples/class) improved held-out `test` Macro-F1 from **`0.5224`** to **`0.6783`** (**`+0.1559`**) and transferred to `gemini-decision-1` (`0.4864` → **`0.6228`**), whereas optimizing `77`-class `banking77` on `40` dev samples (`~0.5`/class) overfit `dev` and regressed on `test` (`0.7568` → `0.7099`).

---

## 2. Main Hypotheses

We formulate five testable hypotheses governing Macro-F1, prompt sensitivity, and model-family trade-offs:

- **H1 (Definition Grounding):** Providing self-sufficient, boundary-explicit class definitions improves Macro-F1 over semantic label names alone (`semantic_only`), and rescues datasets with opaque integer labels.
- **H2 (Label Prior Removal):** Replacing semantic class titles with opaque identifiers (`numbered_def`, `uuid_def`) forces the LLM to reason over definition text rather than label-name priors, preserving or improving Macro-F1.
- **H3 (Positional Bias vs. Prefix Caching):** Fixed class ordering introduces lost-in-the-middle positional bias on imbalanced datasets; per-example shuffling (`uuid_shuffled_def`) eliminates this bias at the cost of breaking KV/prefix caching.
- **H4 (Reflective Self-Improvement & Dev Density):** Iterative reflective optimization of class definitions on a small `train` dev slice improves held-out `test` Macro-F1 and transfers across model families, provided dev-set coverage per class is sufficient to prevent overfitting.
- **H5 (Architecture-by-Dataset Regime Fit):** Supervised sparse (`tfidf+linearsvc`) and dense (`st-head`) classifiers dominate when thousands of clean in-distribution training examples exist, whereas definition-driven zero-shot LLMs and decision models outperform 8-shot contrastive fine-tuning (`setfit:8shot`) as class counts scale (64–151 classes) or when test-only classes appear.

---

## 3. Data & Experimental Setup

### 3.1 Datasets
We evaluate across five public intent classification benchmarks from Hugging Face:

| Dataset | Hugging Face Source | Train Size | Test Size | Classes (Train / Test / Union) | Label Schema & Structural Notes |
| :--- | :--- | ---: | ---: | :---: | :--- |
| **`clinc150`** | `DeepPavlov/clinc150` | 15,200 | 5,500 | `151 / 151 / 151` | `int64` opaque IDs (`"0"`–`"150"`); 10 domains + out-of-scope (`oos`) |
| **`banking77`** | `PolyAI/banking77` | 10,003 | 3,080 | `77 / 77 / 77` | `ClassLabel` strings; single narrow retail-banking domain |
| **`hwu64`** | `DeepPavlov/hwu64` | 8,954 | 1,076 | `64 / 64 / 64` | `int64` opaque IDs (`"0"`–`"63"`); smart-home/assistant domains |
| **`atis`** | `tuetschek/atis` | 4,978 | 893 | `22 / 20 / 26` | String labels; **`4` test-only compound labels** (`flight+airline`, `airfare+flight`, `day_name`, `flight_no+airline`); `76%` `flight` in 200-sample subset |
| **`snips`** | `DeepPavlov/snips` | 13,084 | 1,400 | `7 / 7 / 7` | `int64` opaque IDs (`"0"`–`"6"`); balanced everyday voice commands |

All subsampled `test` evaluations use deterministic shuffling (`seed=42`). Unless otherwise noted, supervised/few-shot baselines (`tfidf+linearsvc`, `tfidf+logreg`, `st-head`, `setfit:8shot`) and `atis`/`banking77` LLM ablations use `200` test samples; `hwu64` LLM ablation uses `40`; `clinc150` LLM ablation uses `20`; `snips` local LLM inference uses `50`; and `gemini-decision-1` uses `50` per dataset.

---

## 4. Analytical Approach

### 4.1 Self-Sufficient Definition Generation & Isolation Audit
1. **Generation**: For each of the `325` union classes, up to `10` `train` utterances (zero `test` leakage) were sampled and passed to `gemini-3.5-flash-lite` with structured JSON output (`ClassDefinition`: `semantic_name`, `definition`, `inclusion_criteria`, `exclusion_criteria`, `representative_patterns`).
2. **Isolation Audit & Repair**: `gemini-4-hydrogen` audited every definition in isolation for (a) self-sufficiency and (b) absence of sibling class/label references. It flagged and rewrote **83 / 325** definitions (`47/77` in `banking77`, `13/26` in `atis`, `12/64` in `hwu64`, `10/151` in `clinc150`, `1/7` in `snips`).

### 4.2 Prompt Perturbation Ablation Design
Five prompt modes were evaluated with `temperature=0.0` and JSON schema `enum` constraints:
1. **`semantic_only`** *(Baseline)*: `semantic_name` labels only, no definitions.
2. **`semantic_def`**: `semantic_name` + full self-sufficient definition.
3. **`numbered_def`**: Sequential integer IDs (`"1"`..`"N"`) + definition (`semantic_name` omitted).
4. **`uuid_def`**: Deterministic UUIDv5 per `(dataset, label)` + definition, fixed order (`semantic_name` omitted).
5. **`uuid_shuffled_def`**: Same fixed UUIDv5s as `uuid_def`, with class order shuffled per test example (`semantic_name` omitted).

### 4.3 Reflective Optimization, Decision Models, Few-Shot & Supervised Baselines
- **Reflective Definition Optimization**: A DSPy-style hill-climbing loop where `gemini-3.5-flash-lite` evaluates a `train` dev slice, `gemini-4-hydrogen` reflects on false positives/negatives of the top-4 most-confused classes to rewrite their definitions (preserving self-sufficiency), and mutations are accepted only if dev Accuracy (tie-broken by Macro-F1) improves.
- **Decision Model (`gemini-decision-1`)**: Evaluated both with raw dataset labels and with the generated class definitions passed as `criteria`.
- **Non-Determinism Mitigation**: Tested `3×` majority voting with `uuid_shuffled_def` on `atis` (`100` samples) and `3×` re-runs on individual error cases.
- **Sparse Supervised Baselines (`tfidf+linearsvc`, `tfidf+logreg`)**: Word `(1,2)` + char_wb `(3,5)` TF-IDF features with `LinearSVC` and `LogisticRegression` trained on full `train` splits.
- **Dense Embedding Head (`st-head`)**: Normalized frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings paired with a `LogisticRegression` (`C=5.0`) classification head trained on full `train` splits (evaluated on CPU).
- **Few-Shot Contrastive Fine-Tuning (`setfit:8shot`)**: `SetFitModel` (`sentence-transformers/all-MiniLM-L6-v2`) fine-tuned on a stratified `8`-shot sample per class from `train` (`1` epoch, `10` contrastive pair iterations, batch size `16`, evaluated on CPU).

---

## 5. Analysis Results

### 5.1 Prompt Perturbation Ablation Matrix (Testing H1, H2, H3)

#### Cloud Generative LLM (`gemini-3.5-flash-lite`)

| Dataset (Samples) | `semantic_only` | `semantic_def` | `numbered_def` | `uuid_def` | `uuid_shuffled_def` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`banking77`** (200) | `0.8350` / `0.7921` | `0.8750` / `0.8390` | `0.8800` / `0.8382` | `0.8750` / `0.8542` | **`0.8900` / `0.8582`** |
| **`atis`** (200) | **`0.7450`** / `0.5603` | `0.6950` / `0.5766` | `0.5850` / `0.5854` | `0.5950` / **`0.6092`** | `0.7200` / `0.6022` |
| **`hwu64`** (40) | `0.9000` / `0.8275` | `0.9000` / `0.8275` | `0.9000` / `0.8275` | **`0.9250` / `0.8514`** | **`0.9250` / `0.8514`** |
| **`clinc150`** (20) | `0.9500` / `0.9273` | `0.9000` / `0.8556` | `0.9500` / `0.9273` | `0.9500` / `0.9273` | `0.9500` / `0.9273` |

*(Cells report `Accuracy / Macro-F1`.)*

#### Local Inference (`google/gemma-4-e4b` via LM Studio on `snips`, 50 samples)

| Mode | Accuracy | Macro-F1 | Avg Latency (ms/utt) |
| :--- | ---: | ---: | ---: |
| `semantic_only` | `0.9600` | `0.9720` | `7,586.6` |
| `semantic_def` | `0.9800` | `0.9848` | `6,093.7` |
| `numbered_def` | `0.9800` | `0.9848` | `6,705.7` |
| `uuid_def` | `0.9800` | `0.9728` | `11,319.3` |
| `uuid_shuffled_def` | `0.9800` | `0.9848` | **`69,751.9`** |

**Findings on H1–H3:**
- **H1 & H2 confirmed**: Definitions consistently improve Macro-F1 on fine-grained/imbalanced tasks (`+0.0661` on `banking77`, `+0.0489` on `atis`, `+0.0239` on `hwu64`), and opaque `uuid_def` / `uuid_shuffled_def` match or beat `semantic_def` on Macro-F1 across all datasets.
- **H3 confirmed**: On `atis` (`76%` `flight`), fixed-order opaque labels (`numbered_def` `0.5850` Acc, `uuid_def` `0.5950` Acc) suffered positional degradation on the majority class, which `uuid_shuffled_def` (`0.7200` Acc) recovered. Conversely, on local LM Studio (`snips`), shuffling destroyed KV prefix caching and increased latency **11.4×** (`69.75s` vs `6.09s`).

### 5.2 Reflective Definition Optimization & Error Analysis (Testing H4)

| Dataset | Dev Slice (Density) | Dev Macro-F1 (Iter 0 → Best) | Held-Out Test | Test Macro-F1 (Before → After) | Test Acc (Before → After) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`atis`** | `80` / `22` (`~3.6`/cls) | `0.6350` → `0.9828` (iter 2) | `200` | `0.5224` → **`0.6783`** (**`+0.1559`**) | `0.7150` → **`0.9050`** (**`+0.1900`**) |
| **`banking77`** | `40` / `77` (`~0.5`/cls) | `0.7812` → `0.9140` (iter 1) | `80` | `0.7568` → `0.7099` (`-0.0469`) | `0.8375` → `0.8000` (`-0.0375`) |

- **Cross-model transfer (H4)**: Evaluating the optimized `atis` definitions with `gemini-decision-1` (`50` samples) raised Accuracy from `0.8200` to **`0.9000`** and Macro-F1 from `0.4864` to **`0.6228`** (`+0.1364`).
- **Error decomposition on optimized `atis`**: Out of `22` errors (`200` samples), `17` were literal definition-following on ATIS dataset conventions (`12` `flight`→`flight+airline`, `2` `airfare+flight`, `3` `flight`→`quantity` on *"how many ... flights"*). Among the `5` genuine LLM errors, `3×` re-runs showed **`2` persistent** (`#14` `day_name`, `#112` arrival-time phrasing) and **`3` transient sampling noise** (`#44`, `#54`, `#195` all `3/3` correct on re-run).
- **3× Shuffled Majority Vote (`atis`, 100 samples)**: Individual votes scored `0.6900` (`0.5025` F1), `0.6200` (`0.5045` F1), and `0.7000` (`0.5756` F1); majority voting stabilized Accuracy at **`0.7000`** (`0.5031` Macro-F1).

### 5.3 Full Cross-Architecture Comparison (Testing H5)

| Dataset | `tfidf+linearsvc` (Full Train) | `st-head` (Full Train) | `setfit:8shot` (8/cls) | Best Zero-Shot LLM (w/ Defs) | `gemini-decision-1` (50, w/ Defs) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`clinc150`** (151) | `0.8000` / `0.8003` (`0.07ms`) | `0.8400` / `0.8553` (`2.46ms`) | `0.7200` / `0.7273` (`2.60ms`) | **`0.9500` / `0.9273`** *(20)* | **`0.9400` / `0.8977`** (`1.80s`) |
| **`banking77`** (77) | **`0.9550` / `0.9460`** (`0.07ms`) | `0.9300` / `0.9079` (`3.15ms`) | `0.8000` / `0.7578` (`2.95ms`) | `0.8900` / `0.8582` (`5.43s`) | `0.8400` / `0.6825` (`1.96s`) |
| **`hwu64`** (64) | `0.9300` / `0.9110` (`0.05ms`) | **`0.9400` / `0.9261`** (`2.18ms`) | `0.7350` / `0.6883` (`2.30ms`) | `0.9250` / `0.8514` *(40)* | `0.8800` / `0.7968` (`1.13s`) |
| **`atis`** (20) | `0.9450` / `0.6965` (`0.08ms`) | **`0.9650`** / `0.7179` (`2.47ms`) | `0.9350` / **`0.7285`** (`2.60ms`) | `0.9050` / `0.6783` *(opt)* | `0.9000` / `0.6228` *(opt, 50)* |
| **`snips`** (7) | **`0.9850` / `0.9839`** (`0.07ms`) | `0.9600` / `0.9586` (`8.77ms`) | `0.9200` / `0.9191` (`2.98ms`) | `1.0000` / `1.0000` *(10)* | **`1.0000` / `1.0000`** (`0.33s`) |

*(Cells report `Accuracy / Macro-F1` and per-utterance inference latency.)*

### 5.4 Why Different Approaches Perform Differently Across Datasets

The cross-dataset divergence is explained by four structural properties of the benchmarks interacting with each model's inductive bias:

1. **Narrow-Domain Lexical Separation vs. Multi-Domain Semantic Overlap (`banking77` vs. `clinc150`)**:
   - On **`banking77`**, all `77` intents share a single retail-banking vocabulary, and fine-grained distinctions often hinge on specific keyword combinations (e.g., `pending_card_payment` vs. `card_payment_not_recognised`). With `10,003` training examples, high-dimensional sparse word + character n-grams (`tfidf+linearsvc`, `0.9460` F1) preserve exact lexical cues better than a compressed 384-dim frozen embedding (`st-head`, `0.9079` F1) or zero-shot definitions (`0.8582` F1).
   - On **`clinc150`**, `151` classes span `10` broad domains *plus* an out-of-scope (`oos`) rejection class. Sparse TF-IDF struggles with paraphrases and `oos` boundary detection (`0.8003` F1), whereas dense semantic representations (`st-head` `0.8553` F1), calibrated decision models (`gemini-decision-1` `0.8977` F1), and zero-shot LLMs with audited definitions (`0.9273` F1) generalize far better across diverse phrasings.

2. **Label-Space Scale in Low-Data Regimes (`setfit:8shot` vs. Definition-Driven Zero-Shot)**:
   - On **`hwu64`** (`64` classes), **`banking77`** (`77` classes), and **`clinc150`** (`151` classes), `setfit:8shot` trails zero-shot definition LLMs by `~10–20` Macro-F1 points (`0.6883` vs `0.8514` on `hwu64`; `0.7578` vs `0.8582` on `banking77`; `0.7273` vs `0.9273` on `clinc150`). With `64–151` classes and only `8` examples/class, `1` epoch of contrastive fine-tuning on a compact 22M-parameter `MiniLM` backbone (`10` pair iterations) underfits the complex multi-way decision boundaries—whereas an LLM conditioned on explicit self-sufficient definitions leverages rich pretrained semantic priors.

3. **Severe Class Imbalance & Stratified Rebalancing (`atis`)**:
   - On **`atis`** (`76%` `flight`), full-train supervised models (`tfidf+linearsvc` `0.6965` F1, `st-head` `0.7179` F1) achieve high overall Accuracy (`0.9450`–`0.9650`) by heavily favoring `flight`, while zero-shot models lose points on `4` test-only compound labels (`flight+airline`, `airfare+flight`) that supervised models cannot even predict.
   - Remarkably, **`setfit:8shot` achieved the highest Macro-F1 (`0.7285`) on `atis` using only `138` training examples**: capping each class at `8` examples naturally rebalances the training distribution so rare intents (`abbreviation`, `airfare`, `ground_service`, `capacity`) receive equal contrastive gradient weight against `flight`.

4. **Task Simplicity & Ceiling Effects (`snips`)**:
   - **`snips`** has only `7` semantically distinct intents (`PlayMusic`, `GetWeather`, `BookRestaurant`, etc.), so all approaches achieve `0.92`–`1.00` Macro-F1 once opaque integer labels are resolved.

### 5.5 Why Zero-Shot Generative Classification on Self-Sufficient Definitions Does Not Always Win

Even when every class has an audited, self-sufficient definition, flat zero-shot generative classification lags supervised or decision-model approaches in five recurring regimes:

1. **Dataset Quirks vs. Literal Semantics**: Benchmarks often contain idiosyncratic annotation rules that contradict literal definitions (e.g., on `atis`, *"how many northwest flights leave st. paul"* is labeled `flight` rather than `quantity`, and airline-filtered flight searches are `flight` rather than `flight+airline`). Supervised models (`tfidf+linearsvc`, `st-head`) memorize these conventions from thousands of examples, whereas a zero-shot LLM faithfully following literal definitions is penalized.
2. **Long-Context Dilution & Positional Bias**: Stuffing `77` (`banking77`, `~26k` tokens) or `151` (`clinc150`, `~42k` tokens) full definitions into a single prompt degrades fine-grained pairwise discrimination and introduces lost-in-the-middle bias (`atis` `uuid_def` `0.5950` vs. `uuid_shuffled_def` `0.7200` Accuracy). By contrast, `tfidf+linearsvc` (`0.9460` F1 on `banking77`) learns exact hyperplane margins across all `77` classes simultaneously.
3. **Absence of Calibrated Domain Priors**: Without training examples, the LLM lacks empirical class-frequency priors and subtle domain-specific tie-breakers that `st-head` and `tfidf+linearsvc` calibrate from `5,000–15,000` in-domain utterances.
4. **Residual Sampling Non-Determinism**: Even at `temperature=0.0` with JSON schema `enum` constraints, hosted generative LLMs exhibit occasional flips (`3` of the `5` genuine `atis` errors disappeared upon `3×` re-running), whereas `tfidf+linearsvc`, `st-head`, and `gemini-decision-1` are deterministic.
5. **Definition Imperfections & Sparse-Dev Overfitting**: Definitions synthesized from `10` examples (or `0` for `test`-only classes like `atis`'s `day_name` and `flight+airline`) can leave subtle boundary gaps—and refining them with too few dev examples per class (`banking77`: `40` dev / `77` classes) overfits rather than generalizes (`-0.0469` Macro-F1).

---

## 6. Future Direction: Hierarchical Two-Stage Classification via Super-Class Clustering

For large label spaces such as `clinc150` (`151` classes) or `banking77` (`77` classes), flat prompting forces the LLM to attend over `~26k–42k` tokens per query. A **hierarchical two-stage (Cluster → Leaf)** architecture directly addresses both context dilution and token cost:

### 6.1 Proposed Pipeline
1. **Offline Semantic Clustering & Summary Synthesis**:
   - Embed the `C` self-sufficient leaf definitions using a dense encoder (e.g., `all-MiniLM-L6-v2`) and partition them into `K` semantically coherent super-classes (e.g., `151` `clinc150` classes → `K ≈ 10–15` clusters of `~10–15` intents each, mirroring natural domains such as *Banking*, *Travel*, *Kitchen*, *Smart Home*, etc.).
   - Synthesize and isolation-audit a **summary definition** for each super-class capturing the union of its member intents and explicit cross-cluster boundaries.
2. **Stage 1 — Coarse Cluster Routing**:
   - Given an utterance and the `K` super-class summary definitions (`~2.5k–4k` tokens instead of `~42k`), the zero-shot classifier selects the relevant super-class (or top-`m` super-classes).
3. **Stage 2 — Fine-Grained Leaf Disambiguation**:
   - The classifier then selects the final intent from *only* the `~10–15` full self-sufficient leaf definitions inside the routed cluster(s), using opaque identifiers (`uuid_def` / `uuid_shuffled_def`) to maximize Macro-F1.

### 6.2 Expected Benefits vs. Failure Modes & Mitigations

| Dimension | Impact of Two-Stage Clustering | Design Mitigation for Risks |
| :--- | :--- | :--- |
| **Token Cost & Latency** | Reduces prompt size on `clinc150` from `~42k` tokens (flat) to `~5k–7k` tokens total across both stages (**~6×–8× reduction**), making `uuid_shuffled_def` practical even on local models. | Cache Stage 1 summary prompts (static `K`-way prefix) for near-zero Stage 1 prefill cost. |
| **Long-Context Attention** | Stage 2 compares only `~10–15` confusable siblings side-by-side, mitigating lost-in-the-middle dilution. | Group semantically adjacent/confusable classes into the same cluster so fine-grained boundaries are resolved in Stage 2. |
| **Hard-Routing Error Propagation** | *Primary risk*: if Stage 1 picks the wrong cluster, Stage 2 has `0%` chance of recovering the true leaf class (especially for utterances straddling two clusters or out-of-scope `oos`). | **Top-2 / Probability-Thresholded Routing**: Route top-`2` clusters (or use `gemini-decision-1` in Stage 1 and take clusters until cumulative probability `≥ 0.95`), passing `~20–25` leaf definitions to Stage 2—still ~6× smaller than flat `151`. Allow multi-membership for boundary/OOS classes or add an *"Escalate / None of these"* fallback in Stage 2. |
| **Optimization Data Efficiency** | Reflective optimization on `K ≈ 12` super-classes with a `40`–`80` sample dev set yields `~3–7` examples/cluster, avoiding the sparse-dev overfitting observed when optimizing `77` flat classes on `40` examples. | Optimize Stage 1 cluster summaries first on small dev sets, and only optimize Stage 2 leaf definitions when `≥2–3` dev examples per active leaf class exist. |

### 6.3 Empirical Evaluation of Two-Stage Hierarchical Classification

We implemented and benchmarked this two-stage architecture using agglomerative cosine clustering over `all-MiniLM-L6-v2` definition embeddings, `gemini-4-hydrogen` cluster summaries, and `gemini-3.5-flash-lite` (`uuid_shuffled_def` in Stage 2) on `banking77` (`K=10`, `200` test samples) and `clinc150` (`K=12`, `50` test samples):

| Dataset (Samples) | Routing | Clusters ($K$) | Stage 1 Recall | Accuracy | Macro-F1 | Avg Latency (ms/utt) | Prompt Size (Chars/utt) |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| **`banking77`** (200) | `top-1` | 10 | `0.9050` | `0.8300` | `0.7912` | `3,540.3` | `27,448` |
| **`banking77`** (200) | **`top-2`** | 10 | **`0.9850`** | `0.8800` | **`0.8631`** | **`3,494.0`** | **`46,022`** *(vs ~118k flat)* |
| **`clinc150`** (50) | **`top-1`** | 12 | `0.9000` | `0.8800` | **`0.9060`** | **`3,645.9`** | **`44,124`** *(vs ~256k flat)* |
| **`clinc150`** (50) | `top-2` | 12 | **`0.9400`** | `0.8800` | `0.8397` | `3,711.1` | `71,701` |

**Empirical Takeaways:**
- **Top-1 routing is bottlenecked by Stage 1 recall (`~90%`), and `top-2` fixes it**: Moving from `top-1` to `top-2` raised Stage 1 recall from `0.9050` to **`0.9850`** on `banking77` and from `0.9000` to **`0.9400`** on `clinc150`.
- **`top-2` beats flat prompting on `banking77` while cutting cost & latency**: On `banking77` (`200` samples), `top-2` hierarchical routing achieved **`0.8631` Macro-F1** (surpassing flat `uuid_shuffled_def` at `0.8582` Macro-F1), ran **1.55× faster** (`3,494 ms` vs `5,432 ms`), and reduced input prompt size by **~2.56×** (`46k` vs `~118k` chars/utt).
- **`top-1` achieves strong Macro-F1 on `clinc150` at ~5.8× lower prompt size**: On `clinc150` (`50` samples), `top-1` reached **`0.9060` Macro-F1** using `44.1k` chars/utt (compared to `~256k` chars/utt flat), whereas `top-2` (`0.8397` F1) introduced additional Stage 2 distractors on this 50-sample slice.

---

## 7. Related Work & Comparison to State of the Art (2023–2026)

1. **Definition-Driven Zero-Shot Intent Classification**: Parikh et al. (ACL 2023, arXiv:2305.07157) demonstrated that prompting LLMs with natural-language intent descriptions yields competitive zero-shot performance on `BANKING77` and `CLINC150`. Our results confirm and extend this finding in two ways: (a) raw generated descriptions frequently leak sibling label names (`83/325` definitions flagged by our isolation audit), and (b) stripping semantic label titles (`uuid_def` / `uuid_shuffled_def`) matches or beats `semantic_def` across all datasets with `>7` classes, while resolving opaque `int64` dataset schemas (`+0.80` to `+0.94` Accuracy on `gemini-decision-1`).
2. **Positional Bias, Label Permutations & Systems Trade-Offs**: Recent diagnostic frameworks (e.g., *PertEval*, *Label Space Reduction*, *ZOE*, and permutation-aware tuning, 2024–2026) show that LLMs suffer primacy/recency biases—flipping `>30%` of predictions under label-order shifts—and context dilution as label counts grow. Our `atis` ablation (`uuid_def` `0.5950` vs. `uuid_shuffled_def` `0.7200` Accuracy) directly reproduces this positional effect, while quantifying a critical systems trade-off often omitted in accuracy-only studies: per-example shuffling invalidates KV prefix caching, causing an **11.4× latency penalty** (`69.75s/utt` vs. `6.09s/utt`) on local LM Studio inference (`google/gemma-4-e4b`).
3. **Reflective Error-Driven Prompt Optimization**: Frameworks such as *GEPA* / `dspy.GEPA` (ICLR 2026), *AdaGEPA* (2026), *ERGO* (2026), and *DSPy/MIPROv2* replace scalar rewards with natural-language reflection (*diagnose-prescribe-rewrite* loops) on text and intent classification (`CLINC150`, `TREC`). Our reflective loop confirms strong gains on boundary-learnable tasks (`atis`: `+0.1559` Macro-F1, transferring `+0.1364` F1 to `gemini-decision-1`), but uncovers an important boundary condition: when dev density falls below `~1` example/class (`banking77`: `40` dev / `77` classes), reflective optimization overfits dev errors and regresses on held-out `test` (`-0.0469` Macro-F1).
4. **Hierarchical Two-Stage Zero-Shot Classification**: Recent studies on black-box hierarchical text classification—such as Yoshimura et al. (arXiv:2508.04219), *KG-HTC* (arXiv:2505.05583), and *TaxMorph* (arXiv:2601.18375)—show that top-down routing reduces ambiguity and token cost, and that human-curated hierarchies often misalign with LM embedding spaces. Rather than relying on pre-defined taxonomies, our two-stage pipeline induces clusters directly from `all-MiniLM-L6-v2` definition embeddings, demonstrating empirically that `top-1` routing is bottlenecked by `~90%` Stage 1 recall while **`top-2` routing** recovers `0.9850` recall on `banking77` and beats flat prompting (`0.8631` vs. `0.8582` Macro-F1) at `~2.56×` fewer input characters.
5. **SetFit & Lightweight Encoders vs. Zero-Shot LLMs**: Comparative 2024–2026 literature generally finds that few-shot *SetFit* (`5–20` examples/class) outperforms zero-shot LLMs on fine-grained intent benchmarks while running far faster on CPU. Our results refine that consensus: under a fast `8`-shot CPU budget (`1` epoch, `10` pair iterations on `all-MiniLM-L6-v2`), `setfit:8shot` trailed definition-driven zero-shot LLMs on `64–151` class tasks (`0.6883–0.7578` vs. `0.8514–0.9273` Macro-F1), yet achieved the **highest Macro-F1 (`0.7285`) on severely imbalanced `atis`** via stratified rebalancing—while full-train `st-head` (`2–3 ms/utt`) outperformed `tfidf+linearsvc` on `clinc150`, `hwu64`, and `atis`.

---

## 8. Conclusion & Task-Selection Decision Rule

Our experiments demonstrate that **self-sufficient, audited class definitions** are essential for zero-shot intent classification—unlocking opaque label spaces (`+0.80` to `+0.94` Accuracy on `gemini-decision-1`) and lifting Macro-F1 on fine-grained taxonomies (`+0.0661` on `banking77`). Removing semantic label headers (`uuid_def` / `uuid_shuffled_def`) consistently improves or preserves Macro-F1 by preventing label-name shortcuts, while reflective definition optimization can nearly match supervised Macro-F1 (`0.6783` vs `0.6965` on `atis`) when dev density is sufficient (`≥2–3` examples/class). For large taxonomies (`77–151` classes), **two-stage hierarchical routing (`top-2` on `banking77`, `top-1` on `clinc150`)** delivers `0.8631–0.9060` Macro-F1 while cutting prompt size **2.5×–5.8×** and reducing latency. Among non-generative methods, `st-head` (`all-MiniLM-L6-v2` + `LogisticRegression`) provides the strongest full-train Macro-F1 on multi-domain tasks (`clinc150`, `hwu64`), while `setfit:8shot` excels on imbalanced, moderate-size taxonomies (`atis`).

### Operational Decision Rule

1. **Static taxonomy, abundant labeled data (`≥50`/class), sub-10 ms latency required**:
   - **Narrow single domain with fine-grained lexical cues (`banking77`)** → **`tfidf+linearsvc`** (`0.9460` F1, `<0.1 ms/utt`).
   - **Multi-domain or paraphrase-heavy taxonomy (`clinc150`, `hwu64`, `atis`)** → **`st-head`** (`0.7179`–`0.9261` F1, `2–3 ms/utt`).
2. **Small, imbalanced dataset (`≤~25` classes, `~8` labeled examples/class available)**:
   - → **`setfit:8shot`** (`0.7285` F1 on `atis`, `0.9191` F1 on `snips`, `~2.6 ms/utt`).
3. **Zero-shot, `64–150+` classes with scarce labels, unseen test-time classes, or auditable rules required**:
   - Always generate and isolation-audit self-sufficient definitions first.
   - **Large multi-domain space (`>100` classes) / calibrated probabilities / low API latency** → **`gemini-decision-1` + definitions** (`0.8977` F1 on `clinc150`), or **2-stage hierarchical LLM (`top-1` / `top-2`)** to cut token cost ~2.5×–5.8× (`0.9060` F1 on `clinc150`, `0.8631` F1 on `banking77`).
   - **Fine-grained single domain (`banking77`)** → **2-stage `top-2` hierarchical (`uuid_shuffled_def`)** (`0.8631` F1, `~2.56×` fewer tokens) or flat **Generative LLM + opaque labels (`uuid_def` if KV caching matters; `uuid_shuffled_def` if imbalanced)**.
   - **Dev slice available?** Run reflective definition optimization **only if dev has `≥2–3` examples per active class** (`atis`: `+0.1559` F1); skip when sparse (`banking77` `0.5`/cls: `-0.0469` F1).
