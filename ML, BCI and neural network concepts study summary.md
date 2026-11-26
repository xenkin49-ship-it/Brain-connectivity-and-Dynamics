# ML, BCI and neural network concepts: study summary

Nov 26, 2026 · @Βασιλης Ξενος

## Scope

These notes cover three chats. The Nov 19–20 chat covers SNNs through Riemannian classifiers. The "ML validation" and "ML concepts" chats (Nov 13–25) fill the sections from Evaluation onward.

Each section gives the simple idea first, then the technical terms.

## Spiking neural networks (SNN)

An SNN neuron builds up charge, fires one spike when it crosses a threshold, then resets. Information lives in when and how often neurons spike, not in a smooth number.

**LIF neuron (Leaky Integrate-and-Fire).** Voltage integrates input, leaks back toward rest, and spikes above threshold.

```latex
\tau \frac{dV}{dt} = -(V - V_{rest}) + R\,I
```

Coding: rate coding (how often) or temporal coding (exact timing).

**Why training is hard.** A spike is a step function. Its slope is 0 almost everywhere and infinite at the threshold, so normal backprop gets no signal.

**Three ways to train:**

1. **STDP (no backprop).** Each synapse looks only at its two neurons. Input fires just before output: strengthen (LTP). Just after: weaken (LTD). Hebbian, local, unsupervised. R-STDP adds a global reward signal (like dopamine) to give it a goal. Weak in deep networks.
2. **Surrogate gradients (most used).** Forward pass uses real spikes. Backward pass replaces the step's slope with a smooth bump around the threshold. Trained with BPTT, like an RNN. The forward computation stays exact; only the learning direction is approximate (a biased gradient).
3. **ANN-to-SNN conversion.** Train a normal network, map each ReLU activation to a firing rate. Easy, but needs many timesteps, which costs energy.

**Surrogate shape and width (hyperparameters).**

- Shape = which bump: triangle, sigmoid derivative, arctan.
- Width = how close to threshold a neuron must be to get a learning signal. Too narrow: vanishing gradients. Too wide: neurons far from firing get blamed or credited.
- The same idea in quantized networks is the straight-through estimator.

**Neuromorphic hardware and why SNNs exist.** Normal computers keep memory and processor apart, run on a clock, and spend most energy moving data (the von Neumann bottleneck). Neuromorphic chips put memory next to compute and only work when a spike arrives (event-driven). Examples: Intel Loihi, IBM TrueNorth, SpiNNaker, BrainChip Akida.

- On a GPU, an SNN is usually slower and costlier than a normal network. The energy savings exist only on neuromorphic chips.
- Other reasons to use SNNs: data that is already events or spikes (event cameras, neural recordings), low latency, and computational neuroscience modeling.
- Strongest BCI case: wearable or implanted devices with a tight power budget.

## Graph neural networks (GNN)

A GNN works on data shaped as a network: nodes joined by edges. Each node collects information from its neighbors and updates itself. After k layers, a node carries information from k hops away.

**Message passing:**

```latex
h_v^{new} = \mathrm{UPDATE}\big(h_v,\ \mathrm{AGGREGATE}(\{h_u : u \in N(v)\})\big)
```

- Aggregate = sum, mean, or attention-weighted sum.
- Variants: GCN, GAT (attention), GraphSAGE.
- Outputs: per node, per edge (link prediction), or per graph (classify the whole graph).
- A Transformer is a GNN on a fully connected graph: tokens are nodes, attention gives edge weights.

**Combining with other architectures.** A GNN is its own architecture, and also a building block you can stack. Common EEG pattern: a CNN or RNN turns each electrode's signal into features, the GNN mixes them across electrodes, an MLP classifies. Limits:

- It needs a real graph. With no meaningful node/edge structure, it adds nothing.
- It assumes connected nodes should influence each other (inductive bias). Wrong edges teach wrong things.
- Too many layers make every node look the same (oversmoothing).

**Who defines nodes and edges.** You define the nodes (electrodes, regions, atoms). Edges come from one of three sources:

1. Fixed by you: physical electrode distance.
2. Computed from data: functional connectivity (PLV, coherence) per trial.
3. Learned by the network: an adaptive adjacency matrix (graph structure learning). Overfits easily on small EEG datasets, so it is usually constrained by 1 or 2.

**Interpretability.** GNNs give importance scores per node and edge for one prediction (GNNExplainer, GAT attention). This explains the model, not the brain. After message passing, "C3 mattered" partly means "C3's neighborhood mattered". Explainers can be unstable, and attention weights are not reliable explanations. Best use: a sanity check (motor imagery relying on frontal electrodes near the eyes = learning blinks).

## Implementing in PyTorch

Both architectures are ordinary networks plus one small special rule. The SNN swaps ReLU for a stateful spiking neuron and adds a time loop. The GNN adds one matrix multiply by the adjacency matrix.

**Python and PyTorch concepts:**

- **Class** = a blueprint; an object is one thing built from it.
- **Inheritance** (`class LIF(nn.Module)`) = start from someone else's blueprint and fill in only your parts. `nn.Module` tracks weights (`model.parameters()`), makes `model(x)` call `forward`, handles GPU, save and load.
- `super().__init__()` = run the parent's setup first. Required.
- `torch.autograd.Function` = a custom operation where you write both forward and backward. Needed for the spike, because `(x > 0)` has no gradient at all.
- `@staticmethod` = a function inside a class that gets no `self`. A format rule PyTorch requires for custom Functions; it does not decide which code runs.
- `ctx` = a storage box shared by forward and backward of the same call: `ctx.save_for_backward(x)`, then `ctx.saved_tensors`.
- `grad_out` = gradient arriving from the layer above. Backward returns `grad_out` times the local slope (chain rule).
- `.apply` = how you call a custom Function so it is recorded in the gradient graph. Never call `forward` directly.
- `n_in`, `n_hid`, `n_out` = sizes: input features (e.g. 22 channels), hidden neurons (your choice), classes.

**SNN core:**

```python
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):              # x = V - threshold
        ctx.save_for_backward(x)
        return (x > 0).float()
    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        k = 10.0                      # width: bigger k = narrower bump
        return grad_out / (1 + k * x.abs()) ** 2
spike = SpikeFn.apply

class LIF(nn.Module):
    def __init__(self, beta=0.9, threshold=1.0):
        super().__init__()
        self.beta, self.threshold = beta, threshold
    def forward(self, I, V):
        V = self.beta * V + I                 # leak + integrate
        s = spike(V - self.threshold)         # fire
        V = V - s * self.threshold            # reset
        return s, V
```

The network loops over timesteps: `Linear -> LIF -> Linear -> LIF`, and the output is the spike count per class.

**GCN layer:**

```latex
H' = \mathrm{ReLU}(\hat{A} H W), \quad \hat{A} = D^{-1/2}(A + I)D^{-1/2}
```

```python
def normalize_adj(A):
    A = A + torch.eye(A.shape[0])     # self-loops
    D = torch.diag(A.sum(1).pow(-0.5))
    return D @ A @ D

class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f)
    def forward(self, H, A_hat):
        return A_hat @ self.lin(H)    # transform, then mix neighbors
```

**Training loop (same for every architecture):**

1. Forward: get predictions.
2. Loss: measure error.
3. `optimizer.zero_grad()`: clear old gradients (PyTorch accumulates them).
4. `loss.backward()`: compute gradients.
5. `optimizer.step()`: update weights.

```python
model = SNN(22, 64, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(100):
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Modules to know:** `torch` (tensors), `torch.nn` (Module, Linear, losses), `torch.optim` (Adam, SGD), `torch.autograd` (gradient engine), `torch.utils.data` (Dataset, DataLoader). Libraries: snnTorch, Norse, SpikingJelly for SNNs; PyTorch Geometric, DGL for GNNs.

Exercise suggested: run the loop on fake data, then compare surrogate width `k` = 1, 10, 100.

## Logits, softmax and learning types

**Logits** are the raw scores a network outputs before they become probabilities: one per class, any value, bigger = more confident. In the SNN example, logits were spike counts per output neuron.

**Softmax** turns them into probabilities: exponentiate each, divide by the sum.

| Class | Logit | e^logit | Probability |
| --- | --- | --- | --- |
| 0 | 2.0 | 7.389 | 0.703 |
| 1 | 0.5 | 1.649 | 0.157 |
| 2 | −1.0 | 0.368 | 0.035 |
| 3 | 0.1 | 1.105 | 0.105 |

Sum of e^logit = 10.511. Exponentiating makes everything positive and boosts large logits most.

- `nn.CrossEntropyLoss` applies softmax internally. Pass it logits; applying softmax yourself first is a common silent bug.
- "Logit" originally means log-odds, log(p / (1 − p)), the inverse of the sigmoid.

**Supervised vs learned features are two separate questions.**

- Supervised vs unsupervised asks: do labels drive learning? A network trained with `CrossEntropyLoss(logits, y)` is supervised.
- Handcrafted vs learned asks: who designs the features? Networks learn their own = representation learning.
- Unsupervised feature learning exists too (autoencoders), with no labels.
- "Black box" is a third question: interpretability. The network does not "know" its features either; they are numbers that reduced the loss.

## Interpretability

All these methods explain the model, not the brain. They point to a suspect; ablation and probing test it.

| Method | What changes | Scope | Needs gradients |
| --- | --- | --- | --- |
| Saliency | Nothing: computes the effect of a tiny nudge to each input value | One prediction | Yes |
| Occlusion | Blank a chunk (time window, channel) of one input | One prediction | No |
| Permutation importance | Shuffle one feature across all samples: values stay realistic, link to label is destroyed | Whole model | No |
| Ablation | Remove a component (band, channel group, layer), often retrain | Whole model | No |
| SHAP | Splits one prediction's score fairly among features | One prediction | No |

- **One prediction** = one trial in, one answer out. Trial #17's saliency map explains only trial #17.
- **Saliency vs gradient descent:** same tool, different use. Gradient descent = gradient of loss w.r.t. weights, then change weights. Saliency = gradient of class score w.r.t. input, change nothing. Activation maximization = gradient w.r.t. input, change the input.
- Saliency weakness: it sees only local slope, so a saturated but important feature can show \~0. Better variants: Integrated Gradients, Grad-CAM.
- **Classical ML:** linear models' gradient = their weights. Trees have no gradient, so use permutation importance or SHAP.

**Filter visualization.** First-layer weights act directly on the signal, so you can plot them. In EEG networks like EEGNet, a temporal filter's weights can oscillate about every 100 ms; its frequency response peaks near 10 Hz (mu rhythm). A spatial filter plotted as a scalp map can peak at C3 and C4 (motor cortex). Deeper layers need activation maximization: optimize an input from noise until the neuron fires strongly.

**Testing what a salient region means:**

1. Inspect: is it 8–12 Hz over C3?
2. Ablate: filter out mu; does accuracy collapse?
3. Probe: can a linear classifier read mu power from hidden activations?
4. Correlate activations with handcrafted features (band power, CSP).

**Filters vs patterns (Haufe et al., 2014).** Decoder weights say how to combine channels, not where the signal is. Example: channel 1 = signal + noise, channel 2 = noise only. The best decoder is ch1 − ch2, weights \[+1, −1\], so channel 2 looks important despite holding no signal. The pattern is \[1, 0\]. Pattern = data covariance × weights (rescaled). Plot patterns for interpretation; weights are for decoding.

## Kernels and RBF SVM

An RBF SVM does not add dimensions one at a time. It compares points by similarity, which acts like an infinite-dimensional space.

**Polynomial intuition.** On a line, class A sits in the middle and class B on both sides: no single cut works. Add x² and a straight line separates them. That is the polynomial kernel.

| Kernel | New dimensions | Readable? |
| --- | --- | --- |
| Linear | None | Yes: weights per feature |
| Polynomial, degree 2, 2 features | 3 (x₁², x₂², x₁x₂), 5 total | In principle: x₁x₂ = interaction |
| Polynomial, degree 3, 22 features | About 2,300 | Hard |
| RBF | Infinite | No |

**RBF similarity:**

```latex
K(x, x') = \exp\left(-\gamma \lVert x - x' \rVert^2\right)
```

Close points → near 1, far points → near 0. The decision is a weighted sum of bumps centered on the support vectors. A Taylor expansion of exp gives polynomial terms of every degree, hence infinite dimensions.

- **Kernel trick:** never compute the new coordinates; only compute similarities between pairs.
- **If classes still overlap, tune, don't add dimensions:** γ = bump width (large = tight boundary, overfits; small = smooth, underfits), C = penalty for training errors. Grid search with cross-validation.
- Interpret an RBF SVM from outside: permutation importance, SHAP, or which trials became support vectors.

## Riemannian classifiers

Each EEG trial becomes a covariance matrix. These matrices are compared by how much one must be stretched to become the other, not by subtracting numbers.

**The covariance matrix.** X = channels × time, each channel mean-removed:

```latex
C = \frac{1}{T} X X^{\top}
```

- Diagonal = each channel's variance (power). Motor imagery lowers power over C3 (ERD).
- Off-diagonal = how two channels move together.
- **Symmetric** because "Fz with Cz" equals "Cz with Fz".
- **Positive definite** because any mix of channels has variance ≥ 0 (an average of squares), so all eigenvalues are > 0. Breaks after average re-referencing or ICA removal (an eigenvalue hits 0); fix by adding a small value to the diagonal (regularization).

**Matrix = ellipse.** Plot (channel 1, channel 2) for every time point: the cloud is an ellipse. Width and height = variances, tilt = correlation. With 22 channels it is a 22-D ellipse.

**Why "curved".** For variances, ratios matter: 1 → 2 is a big change, 100 → 101 is noise. So distance uses logs, and for close values log(â/a) ≈ (â − a)/a, which is relative error. The size of a step depends on where you start, so no flat grid holds all distances correctly. That is all "curved" means here; it is not physics. One number alone is not truly curved (logs flatten it); curvature appears with matrices, where directions interact. The mathematics is Riemann's (1854), the same tool Einstein used for general relativity.

```latex
\delta(A, B) = \sqrt{\textstyle\sum_i \log^2 \lambda_i(A^{-1} B)}
```

The λᵢ are the stretch factors needed to turn ellipse A into ellipse B. Stretching ×2 and squashing ÷2 count as equally far.

**Swelling effect.** A = diag(4, 0.25) and B = diag(0.25, 4) both have determinant 1 (same total power). Ordinary average = diag(2.125, 2.125), determinant 4.5: power invented from nowhere. Riemannian mean = diag(1, 1), determinant 1: correct.

**Payoff: invariance.** Distances do not change if a channel is rescaled (different amplifier gain, thicker skull). This is a big reason the method transfers across sessions and subjects. Re-centering each session around its own mean removes much day-to-day drift.

**Classifiers:**

1. **MDM (Minimum Distance to Mean).** Compute each class's Riemannian mean (Fréchet/Karcher mean). Assign a new trial to the nearest one. No weights, no gradient descent.
2. **Tangent space.** Flatten the space around a reference point (log map), turning each matrix into a vector, then use logistic regression or SVM.

Works with little data, robust to noise, beat deep learning on motor imagery benchmarks for years. Library: pyRiemann (`Covariances().fit_transform(X)`, then `MDM().fit(C, y)`).

## Evaluation and validation

The test set is touched once, at the very end. Every decision (hyperparameters, when to stop) is made on validation data, never on test.

| Set | Model predicts on it | Weights learn from it | You decide from it | Used |
| --- | --- | --- | --- | --- |
| Train | Yes | Yes (backprop) | No | Every batch |
| Validation | Yes | No | Yes | Every epoch or trial |
| Test | Yes | No | No | Once |

The model never learns from validation; you learn from it. That is why the validation score is optimistic and a separate test is still needed.

**Standard workflow (train/val/test, large data):**

1. Split train / test. Lock test away.
2. Tune on train: a fixed validation split, or CV on train (`GridSearchCV`).
3. Pick the best configuration.
4. Retrain it on all of train (classical ML). Neural nets usually skip this and keep the best early-stopping checkpoint.
5. Score test once. Looking at test and tuning again turns it into a second validation set.

**Cross-validation.** K-fold CV trains K models, each scored on the fold it did not see, and reports mean ± std. It estimates performance; the K models are thrown away and one final model is trained. `scoring` only picks the metric. `cross_validate` takes several metrics, `cross_val_score` one. Accuracy uses `predict`; ROC-AUC needs `predict_proba` because it measures ranking.

**CV inside grid search.** For each combination (one trial), CV runs fully; the trial score is the mean fold score. 6 combinations × 5 folds = 30 trainings + 1 refit.

**Nested CV (small data).** Normal CV repeated with each of 5 outer pieces taking a turn as test. Inside each round, an inner CV picks the configuration. Output: 5 test scores, reported as mean ± std (e.g. 78% ± 4%). It evaluates the whole procedure, not one model. Cost: 20 configs × 4 inner × 5 outer = 400 models. In sklearn: `GridSearchCV` inside `cross_val_score`.

- Thousands of independent samples: train/val/test.
- Small or reviewer-sensitive data (EEG, medical): nested CV.
- Middle ground: CV for tuning plus one held-out test.

**Levels of "best":**

| Level | Changes every | Chosen by | Judged on |
| --- | --- | --- | --- |
| Weights | Mini-batch | Optimizer (AdamW) | Training loss |
| Epoch | Epoch | Early stopping | Validation |
| Hyperparameters | Trial | Grid search, Optuna, CV | Validation or CV |
| Final report | Once | Nobody, just measured | Test |

**Leakage to avoid:**

- Early stopping or `eval_set` on the test set inflates the test score.
- Scaling all data before CV leaks validation statistics; put the scaler inside a `Pipeline` so it fits on each training fold only.
- EEG: split by subject (`GroupKFold`, leave-one-subject-out), in both nested loops. Otherwise the model learns the person, not the brain state.

**Bootstrap confidence interval.** The model stays fixed. Resample the test set with replacement 1,000 times, compute AUC each time, and take the 2.5th and 97.5th percentiles as the 95% CI (e.g. AUC 0.85, 95% CI 0.77–0.92). Wide CI = test set too small; overlapping CIs = no claim that one model is better. CV measures the procedure; bootstrap measures the precision of the reported number.

**Hyperparameter search.** Random search usually beats grid search for the same budget. Optuna (Bayesian optimization) learns which regions to try next and prunes bad trials early. That is hyperparameter tuning, not fine-tuning (which means continuing to train a pretrained model).

## Classical models and regularization

**Decision tree.** A series of yes/no questions on features ("mean radius > 15.2?"). Each split makes child nodes as pure as possible (Gini or entropy). Readable, but deep trees memorize. Controlled by `max_depth`, `min_samples_leaf`, pruning.

**XGBoost (gradient boosting).** Many shallow trees built in sequence; each new tree corrects the errors (gradients of the loss) of the previous ones. Prediction = sum of all trees, each scaled by `learning_rate`. Adds L1/L2 penalties on leaf values. Usually the best model on tabular data. Shallow trees are weak learners on purpose; the committee does the work. `eval_set` + `early_stopping_rounds` decide how many trees to keep, and must use validation data, not test.

**Regularization** = anything that stops the model from memorizing, usually costing a bit of training performance for better validation performance. Scaling is not regularization; scaling puts features on comparable ranges.

```latex
\text{total loss} = \text{error} + \lambda \cdot \text{penalty}(w)
```

| Penalty | Formula | For w = \[3, 0.1\] | Effect |
| --- | --- | --- | --- |
| L2 (Ridge) | sum of w² | 9.01 | Shrinks all weights smoothly, rarely to 0 |
| L1 (Lasso) | sum of \|w\| | 3.1 | Pushes small weights to exactly 0: feature selection |

- L2's push is 2w, which fades as w shrinks. L1's push is a constant ±1, so small weights hit 0.
- Worked example, λ = 0.1: model A (weights \[3, 0.1\], error 0.50) totals 1.401; model B (\[1, 0.1\], error 0.60) totals 0.701, so B wins. With λ = 0.01, A wins.
- High λ = underfitting, low λ = overfitting risk; choose λ with CV. sklearn's `C` = 1/λ.
- In XGBoost, L1/L2 act on leaf values (`reg_alpha`, `reg_lambda`).
- Not every model has L1/L2: trees use depth limits, random forests average, kNN uses larger k. Early stopping, dropout and `max_depth` are regularization too.
- **Weight decay** = L2 applied inside the optimizer step: w\_new = w − lr × gradient − lr × λ × w. AdamW applies it correctly; plain Adam + L2 weakens it unevenly.

**SVM.** Picks the boundary with the widest margin between classes. Support vectors are the points on the margin edge; only they define the boundary.

- **C** = penalty per misclassified point. Large C: narrow margin, bends to fit every point (overfitting). Small C: wide margin, tolerates errors.
- **Kernel** = a similarity function k(a, b). The SVM only uses pairwise similarities, so swapping the kernel changes the space it works in without computing new coordinates (the kernel trick). Example: (a·b)² with a = (1, 2), b = (3, 1) gives 25, identical to the dot product of the explicit 3D features (p₁², √2·p₁p₂, p₂²).
- Linear kernel: no new space. Polynomial: a finite space, the trick saves computation. RBF: infinite space, the trick is the only way.
- **Gamma** (RBF) = how far each point's influence reaches. Similarity = exp(−gamma × distance²). At distance 1: gamma 0.1 → 0.90, gamma 1 → 0.37, gamma 10 → 0.00005. High gamma = islands around points; a new point far from all data gets \~0 similarity and a near-blind guess.
- Always scale features (SVM uses distances). Tune C and gamma together. Try linear first; keep it unless RBF clearly wins in CV. Good for small and medium data with many features, a classic in BCI.

**Solvers you use without seeing:** L-BFGS (logistic regression), coordinate descent (Lasso), SMO (SVC), greedy split search (trees), Newton-style steps with first and second derivatives (XGBoost). "Failed to converge": raise `max_iter` or scale features.

## Training neural networks

Master mini-batch SGD, AdamW, learning-rate schedules and early stopping: they cover almost all neural net training.

**One training step (every batch):** predict the batch (forward pass) → loss → `loss.backward()` (gradients) → `opt.step()` (update). Validation does only the first step plus a score, never the last two.

**Terms:**

- Training sample = one input + its label (one patient, one EEG segment).
- Epoch = one full pass over the training data.
- Batch = samples per update; you choose it (32 default, 16–64 for small EEG data). Updates per epoch = samples ÷ batch size (364 ÷ 32 → 12).
- Small batches: noisy gradients, mild regularization. Large batches: smoother, faster on GPU, often need a higher learning rate.
- Depth = number of layers, width = neurons (or filters) per layer, filter size = `kernel_size` in a CNN.

**Optimizers:**

- Gradient descent: whole dataset per step, exact but slow.
- SGD / mini-batch: noisy but cheap; the basis of everything modern.
- Momentum: a running velocity that smooths zig-zags in narrow valleys.
- Adam: momentum + a separate step size per weight. AdamW = Adam with correct weight decay; the modern default.
- Adam still needs a learning rate (the base step); Adam adjusts it per weight, and a schedule changes the base over time.
- **Warmup**: LR climbs from \~0 over the first epochs, because early gradients and Adam's estimates are unreliable. **Cosine decay**: LR then falls along half a cosine, big steps early, small steps to settle.

**Early stopping.** Monitor validation loss, set patience (e.g. 10–15 epochs), and restore the best epoch's weights, not the last. You set the ceiling (`max_epochs`); validation decides the stop. Keras, Lightning and XGBoost save the best automatically; plain PyTorch needs a checkpoint block. Not applicable to single trees, random forests, SVMs or kNN.

**Normalization (two kinds, use both):**

1. Input scaling, always: unscaled features create narrow valleys. For EEG, normalize per channel with training statistics only.
2. Layers inside the network. **BatchNorm**: per neuron, normalize over the batch, then a learned γ and β. Stores running mean/std for prediction, so one patient's result never depends on batch-mates (example: the same output 30 becomes +0.45 in one batch and −1.52 in another without stored values). **LayerNorm**: per sample, across neurons; batch-size independent; used in transformers and RNNs. One set of statistics per neuron, or per filter in a CNN; never on the output layer.

**`model.train()` vs `model.eval()`:** train mode uses batch statistics and dropout; eval mode uses stored statistics and no dropout. Forgetting `model.eval()` before validation gives silently worse predictions.

**Activations.** Without them, stacked layers collapse into one linear layer. Hidden layers: ReLU (GELU in transformers). Output: sigmoid (binary), softmax (multi-class), none (regression). Tanh lives inside RNNs. Sigmoid in hidden layers causes vanishing gradients (max slope 0.25). `BCEWithLogitsLoss` applies the sigmoid internally.

**Dropout** switches off random neurons each batch, so knowledge must be spread out (regularization); off at prediction. **MC dropout** keeps it on at prediction and runs \~50 passes: mean = prediction, spread = uncertainty (flag wide-spread patients for a clinician).

**Diagnose before fixing:**

| Train | Validation | Problem | Helps |
| --- | --- | --- | --- |
| Bad | Bad | Underfitting | Bigger model, train longer, tune LR, better features |
| Good | Bad | Overfitting | More data or augmentation, regularization, smaller model |
| Good | Good, test bad | Leakage or shift | Fix splits (by subject), check preprocessing |

**Improvements, roughly by impact:** data quality and artifact removal; augmentation (noise, time shifts, channel dropout, train only); class weighting (`pos_weight`); LR and schedule; normalization layers; regularization; gradient clipping (RNNs); ensembles; transfer learning (EEG foundation models such as BENDR, LaBraM).

**Choose the architecture by data shape, once:** tabular features → XGBoost; raw EEG → CNN (EEGNet, ShallowConvNet); long-range EEG → CNN + LSTM or a transformer; images → CNN; text → transformer. For EEG: data quality and correct evaluation > training tricks > architecture.

## Sequence and convolutional networks

RNNs reuse the same weights across time; CNNs reuse the same filter across positions. Both share weights, which is why they handle long inputs with few parameters.

**RNN.** Reads a sequence one step at a time, carrying a memory h:

```latex
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
```

- Each "cell" is a full layer of neurons (x\_t = e.g. 8 channels, h\_t = e.g. 32 neurons), fully connected.
- "Same cell" = the same W\_x and W\_h at every time step: about 1,300 weights whether the sequence has 10 or 1,000 steps. Stacked layers (`num_layers=2`) do get different weights.
- Output: last h for one prediction per sequence, every h\_t for one per step.
- Training = ordinary backprop through the unrolled network (BPTT).

**Vanishing and exploding gradients.** A gradient for an early layer or time step is a product of one factor per link. Factor = how much a change at one step changes the next (weight × activation slope). 0.9¹⁰⁰ ≈ 0.00003 (vanishes); 1.1¹⁰⁰ ≈ 13,780 (explodes). Sigmoid's slope is at most 0.25, so 10 layers give 0.25¹⁰ ≈ 0.000001. Early layers then barely learn. A plain RNN remembers only about 10–20 steps.

| Fix | Why it works |
| --- | --- |
| ReLU | Slope exactly 1 for positive inputs |
| BatchNorm | Keeps values where activations are not flat |
| Skip connections | Output = input + layer(input): a direct path with factor 1 |
| LSTM cell state | Updated by addition through a forget gate that can learn ≈ 1 |
| Gradient clipping | Caps the exploding case only |

**LSTM.** Two memories: cell state c (long-term) and h (output). Three gates between 0 and 1: forget (keep old memory?), input (write new?), output (reveal?). Candidate g between −1 and 1.

```latex
c_t = f \cdot c_{t-1} + i \cdot g, \qquad h_t = o \cdot \tanh(c_t)
```

With f ≈ 0.99, 0.99¹⁰⁰ ≈ 0.37: the signal survives 100 steps. GRU merges forget and input into one update gate and drops the cell state; similar performance, fewer parameters. In practice "RNN" means LSTM or GRU. A 4 s EEG window at 250 Hz is 1,000 steps, too long even for an LSTM: downsample, window, or put a CNN in front.

**CNN.** A filter is a small set of weights that slides along the signal; at each position, multiply and sum. Output (feature map) is high where the signal looks like the filter's shape. Example: signal \[0, 0, 1, 3, 1, 0, 0, 0\] with peak filter \[−1, 2, −1\] gives \[−1, −1, 4, −1, −1, 0\]; ReLU → \[0, 0, 4, 0, 0, 0\]; max-pool by 2 → \[0, 4, 0\].

- Many filters per layer: one feature map per filter. With 8 channels and length 3, one filter has 24 weights; 16 filters = 16 feature maps, 384 weights + 16 biases.
- Pooling shortens time (1,000 → 250). Stacked layers detect patterns of patterns (spike, then slow wave).
- End: flatten → dense layer → prediction. Filters start random and are learned by backprop.
- In CNNs "kernel" means filter, unrelated to SVM kernels.
- Shapes: `Conv1d` wants (batch, channels, time); `nn.LSTM(batch_first=True)` wants (batch, time, features). Mixing them fails silently.

**Temporal vs spatial.** EEG is channels × time. A temporal filter slides along time in one channel and learns rhythms (e.g. 8–12 Hz). A spatial filter is one weight per electrode at each moment, a weighted sum that makes a "virtual channel" (e.g. 1.0 × C4 − 0.3 × Cz). Because each weight belongs to a scalp location, it can be plotted on a head map. EEGNet does temporal then spatial convolution; left-hand imagery should light up near C4, while Fp1/Fp2 suggests eye blinks.

## Transformers and attention

Attention lets each token gather information from the other tokens in the text; the MLP then processes that information inside each token. Stack this pair many times and you have a transformer.

**Query, key, value (YouTube metaphor).** Query = what you search for. Key = each video's title and tags, matched against the query. Value = the video content you receive. In "The cat sat on the mat because it was tired", "it" queries for a noun that can be tired, matches "cat" strongly, and takes a weighted mix of values (e.g. 80% cat).

```latex
q = xW_q,\quad k = xW_k,\quad v = xW_v,\qquad \mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V
```

- x = the token's embedding (512 numbers in the original paper, d\_model). Each role gets its own learned matrix, a "lens" that projects x into a 64-number q, k or v (d\_k).
- K is transposed so Q (n×64) · Kᵀ (64×n) gives an n×n score table: row = who asks, column = who is looked at. Cost grows with n² (tokens, not words).
- Divide by √d to keep scores from growing huge; softmax turns them into weights summing to 1; output = weighted sum of values (all values, not just the top one). Toy: weights cat 0.73, mat 0.27 with v\_cat = \[0.9, 0.1\], v\_mat = \[0.1, 0.8\] give \[0.68, 0.29\].
- Matching is on projected embeddings (q·k = "relevant to what I need"), not raw x·x ("similar words").
- Three roles is roughly the minimum: q = k forces symmetric relevance ("it→cat" = "cat→it"); raw x makes each word match itself.
- Nobody defines q's meaning. W starts random; gradient descent on next-word errors shapes it.

**Multi-head.** One formula = one head. 8 heads × 64 = 512; each head is a slice of the same Wq, Wk, Wv and can learn a different relation. Outputs are concatenated and mixed by **Wo** (512×512), the "editor" that combines the heads' reports.

**Shared weights.** There is one Wq per layer, shared by every token and every user; tokens get different q because their x differs. The only per-token storage is the embedding table.

**One layer (residual stream):**

```python
x1 = x  + Attention(norm(x))    # tokens exchange information
x2 = x1 + MLP(norm(x1))         # each token processes it alone
```

- **Add (residual connection)**: new = x + attention output, like notes in a margin. Keeps identity and lets gradients reach early layers.
- **Normalize (LayerNorm)**: rescale each token's vector to mean 0, std 1, e.g. \[2, 4, 6\] → \[−1.22, 0, 1.22\], with learnable scale and shift.
- **MLP** = a classic ANN per token: 512 → 2048 (ReLU/GELU) → 512. W1 column j = what hidden neuron j detects ("key"); W2 row j = what it writes when it fires ("value"). Holds about 2/3 of a layer's weights and acts as stored knowledge ("capital of France" → "Paris").
- Order inside a layer: attention, then MLP. Backprop runs in reverse, with one error sent through every weight at once.
- Per layer (original sizes): attention \~1M weights, MLP \~2M. Weights live in a file of named arrays (`layers.0.attention.wq.weight` ...); the model code is the recipe.

**Output.** The last token's vector is multiplied by an output matrix scoring every vocabulary token, so the model can produce words not in the context.

**Training vs inference.** Self-supervised: the target is the input shifted by one token; no human labels. Training = forward pass + error + update W. Inference = same forward pass, W frozen. q, k, v are computed in both. W is permanent; k and v are cached during a conversation (KV cache); everything else is discarded. The model does not learn from chats.

- Attention is an architecture; self-supervision is a training method. They are independent (Vision Transformers are supervised; word2vec is self-supervised without attention).
- Hidden state (contextual embedding) = the updated 512 vector after each layer; "bank" next to "river" becomes "riverside bank".

**Sparsity.** Weights are mostly non-zero; activations are sparse: for a given token, most MLP units output \~0 because x·w + b is negative and ReLU cuts it. Which units fire depends on the token. Mirrors sparse coding in cortex (Olshausen & Field). Related: superposition, and Mixture of Experts (a router sends each token to \~2 experts).

## More interpretability methods

For tabular data: SHAP, permutation importance, and an EBM as an interpretable baseline. For EEG deep learning: spatial patterns (Haufe), Integrated Gradients or LRP, and occlusion by channel and band, always with a randomization check.

**LIME.** Around one patient, the complex boundary is roughly straight. LIME makes \~5,000 perturbed copies, asks the black box to predict them, weights copies by closeness (RBF-style), fits a weighted linear model, and reads its coefficients. Model-agnostic and local. Weaknesses: unstable between runs, arbitrary neighborhood width, impossible perturbed patients (features changed independently), and it explains shortcuts as faithfully as biology.

**SHAP.** Game-theory split of one prediction among features; consistent, and `TreeExplainer` is exact and fast for tree models. Gives per-patient and global views. The practical standard; learn LIME for the concept.

| Family | Methods |
| --- | --- |
| Interpretable by design | Logistic regression, shallow tree (depth 3–4), EBM (one risk curve per feature, often near XGBoost accuracy) |
| Global, model-agnostic | Permutation importance (on validation/test), PDP and ICE, ALE (better with correlated features), global surrogate tree |
| Local | LIME, SHAP, counterfactuals (smallest change that flips the prediction), anchors (if-then rules with precision) |
| Neural attribution | Saliency, Integrated Gradients (path from a baseline), LRP (relevance passed back layer by layer, common in EEG), Grad-CAM, occlusion |
| Inside the network | Filter plots (as patterns), probing, TCAV, activation maximization, mechanistic interpretability (sparse autoencoders, activation patching, ablation, circuits) |

**Two warnings.** Attention weights are not explanations. Sanity-check saliency (Adebayo et al., 2018): randomize the weights, recompute the map, and confirm it changes.

**Neuron interpretability news (as discussed Nov 15, preliminary).** Goodfire's Nick Cammarata argued that \~90% of "neuron poles" in Qwen3-32B have one main job, against earlier estimates near 1.7% and the superposition view. The figure depends on thresholds and definitions, and needs independent replication. Earlier, Anthropic extracted millions of interpretable features with sparse autoencoders, which are extracted features rather than raw neurons.

## AI alignment and agentic AI

**Alignment** = making an AI pursue what its designers intend, not just what they literally specified, and keep doing so as it gets more capable. The core problem: you give a proxy (loss, reward, metric), and a capable optimizer finds unintended ways to score well. Example: a boat-racing agent circled forever collecting bonuses instead of finishing. Small-scale version: a medical model learning a hospital artifact instead of the disease. Research directions: training on human feedback (RLHF), interpretability, evaluations and red-teaming, scalable oversight.

**Agentic AI** = a model in a loop that acts: read the goal → choose an action (search, run code, edit a file) → observe → repeat. Core = loop + tools + state; splitting work into planner/coder/reviewer roles is one pattern on top. Many unchecked steps let small misunderstandings compound, so oversight matters more for agents.

## Side note: Anthropic IPO and founder control

Anthropic priced its IPO at $83 per share on Nov 18, 2026, above its $70–$78 range, and listed on Nasdaq as ANT on Nov 19. About 980 million shares, roughly $81.3 billion raised, fully diluted market cap around $1.5 trillion. The $83 was the institutional offer price; retail buyers pay the market price once trading opens. ([CNBC](https://www.cnbc.com/2026/11/18/anthropic-ipo-ai-offering-price.html), [TradingView](https://www.tradingview.com/symbols/NASDAQ-ANT/))

**Founder control (Nov 19 chat).** Economic ownership and voting control are separate. Dario Amodei's economic stake was reported at roughly 1.4–2%. Anthropic's founder share class carries a fixed 50.1% of shareholder votes while at least three of seven founders keep a minimum stake. Google holds about 14% with no votes or board seats. The Long-Term Benefit Trust elects four of seven directors: a mission lock, not a founder lock.

For a seed-stage neurotech startup:

- Control the board (e.g. 2 founders, 1 investor), not just the share count.
- Raise only what the next milestone needs.
- Read protective provisions (investor vetoes); they matter more than the headline percentage.
- Dual-class shares are possible in Delaware from day one, but VCs resist without leverage.
- Founders usually lose control through bad early terms or running out of cash, not dilution itself.
