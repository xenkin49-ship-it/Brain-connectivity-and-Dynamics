# Brain Networks, Connectivity and Neural Dynamics: Study Notes

Notes from a guided study session on how the brain is studied as a network, how we infer who drives whom (effective connectivity), and how neural population dynamics (manifolds, attractors, LFADS) explain why decoding brain activity is hard.

Written for someone with a medical background who knows neuroanatomy and basic Python/ML.

Convention used throughout:
- **Verified**: established, replicated, or directly checked.
- **Inference**: a reasonable interpretation that is not settled.

---

## Table of contents

1. [The brain as a network](#1-the-brain-as-a-network)
2. [What counts as a node](#2-what-counts-as-a-node)
3. [Three kinds of connectivity](#3-three-kinds-of-connectivity)
4. [Building a brain network: the pipeline](#4-building-a-brain-network-the-pipeline)
5. [Graph metrics](#5-graph-metrics)
6. [Modules](#6-modules)
7. [Effective connectivity: Granger, transfer entropy, DCM](#7-effective-connectivity-granger-transfer-entropy-dcm)
8. [Where chemistry fits](#8-where-chemistry-fits)
9. [Reading and writing the brain](#9-reading-and-writing-the-brain)
10. [State space, manifolds and attractors](#10-state-space-manifolds-and-attractors)
11. [Inside vs outside the manifold](#11-inside-vs-outside-the-manifold)
12. [Do we need to record every neuron?](#12-do-we-need-to-record-every-neuron)
13. [Neural drift](#13-neural-drift)
14. [LFADS](#14-lfads)
15. [Experiment: mini-LFADS and the hidden kick](#15-experiment-mini-lfads-and-the-hidden-kick)
16. [Statistics: permutation tests and cluster-based tests](#16-statistics-permutation-tests-and-cluster-based-tests)
17. [Confounds: when the result is not the brain](#17-confounds-when-the-result-is-not-the-brain)
18. [A map of brain research methods](#18-a-map-of-brain-research-methods)
19. [Limits: can we even understand networks?](#19-limits-can-we-even-understand-networks)
20. [US labs working on this](#20-us-labs-working-on-this)
21. [Resources](#21-resources)

---

## 1. The brain as a network

Network neuroscience models the brain as a **graph**:
- **Nodes**: brain regions (or neurons).
- **Edges**: connections between them.

Classical neurology asks **where** a function lives (Broca's area, V1). Network neuroscience asks **how the pattern of connections produces function**. Disconnection syndromes such as conduction aphasia already show the idea: a failure of an edge, not only a node.

### Scales

| Scale | Node | Edge | Method |
|---|---|---|---|
| Micro | Single neuron | Synapse | Electron microscopy (full fruit fly wiring diagram, ~140,000 neurons, 2024) |
| Meso | Neuron population, cortical column | Axonal projections | Tracer injections, mostly in animals |
| Macro | Brain region (mm to cm) | White matter tracts, or signal correlations | MRI, DTI, fMRI, EEG, MEG |

Human research is almost entirely macroscale, because that is what can be measured non-invasively.

---

## 2. What counts as a node

There is no single natural node. A node is a **choice**, and the right one depends on scale and question.

A good node should be:
1. **Internally uniform**: everything inside behaves roughly the same.
2. **Distinct from its neighbors**.
3. **Reproducible** across people.

| Node choice | Strength | Problem |
|---|---|---|
| Neuron | Closest to a natural unit | Cannot record billions in humans. Dendrites compute too, so even a neuron is not one simple unit |
| Anatomical region (e.g., Desikan, 68 regions) | Reproducible | Too coarse. One gyrus can hold several functional areas |
| Functional parcel (e.g., Schaefer, 100 to 1000) | More uniform inside | Boundaries depend on data and method |
| Multimodal parcel (Glasser, 360 areas) | Best current MRI compromise | Needs high-quality MRI |
| Voxel | Fine-grained | Arbitrary cube, follows no biology |
| EEG electrode | Easy | A scalp location, not a region. Mixes many sources |
| EEG source (reconstructed) | Closer to real regions | The reconstruction is an estimate with error |

- **Verified**: changing the parcellation (atlas) changes graph metrics.
- **Inference**: brain organization is partly continuous (gradients), so any node definition is an approximation.
- **Practical rule for EEG**: use source-reconstructed regions with a standard atlas when data quality allows. Electrodes as nodes are acceptable only for rough analyses.

---

## 3. Three kinds of connectivity

| | Structural | Functional | Effective |
|---|---|---|---|
| Question | Is there a physical wire? | Do signals change together? | Does A **cause** changes in B? |
| Type of claim | Anatomy | Description of the data | Claim about mechanism |
| Needs a model? | No | No, just compute a statistic | Yes (Granger, DCM), or a perturbation |
| Direction | Not from DTI | Usually none (A-B = B-A) | Yes (A→B ≠ B→A) |
| Time scale | Days to years | Seconds | Seconds |
| Measured with | DTI, tracers, electron microscopy | Correlation, coherence, wPLI | Granger, transfer entropy, DCM, stimulation |

**Analogy**
- Structural = the road map.
- Functional = cities whose traffic rises and falls together.
- Effective = which city's traffic is **causing** the other's.

**Party analogy for functional vs effective**: two people laugh at the same moment.
- Functional: "their laughter is correlated." True, and that is all you claim.
- Effective: "A told a joke and B laughed because of it." A causal claim. Maybe both were watching a funny video (a hidden driver). To claim causation you need a model or an experiment.

**Terminology note**: some researchers call Granger causality "directed functional connectivity" and reserve "effective" for biophysical models such as DCM. Both uses appear in papers.

**Medical example**: in a focal seizure, all channels light up together, so functional connectivity says "everything is connected." Effective connectivity asks which region **starts** it and drives the rest, which matters for localizing the seizure onset zone before surgery.

---

## 4. Building a brain network: the pipeline

Toy example with 4 regions A, B, C, D, each giving a signal over time.

**Step 1. Measure how much each pair moves together** (e.g., correlation, 0 to 1):

|   | A | B | C | D |
|---|---|---|---|---|
| **A** | 1 | 0.8 | 0.7 | 0.1 |
| **B** | 0.8 | 1 | 0.6 | 0.2 |
| **C** | 0.7 | 0.6 | 1 | 0.1 |
| **D** | 0.1 | 0.2 | 0.1 | 1 |

This table is the **N × N adjacency matrix**. Real studies use N = 68 to 400+. It is **symmetric** (A-B = B-A) because correlation has no direction.

**Step 2. Threshold**: keep only strong links (e.g., > 0.5):

```
    A ─── B
     \   /
      \ /
       C          D   (isolated)
```

The choice of threshold changes the network. At 0.65, the B-C link disappears.

**Step 3. Compute graph metrics** on the result (next section).

**Step 4. Compare** groups or conditions, with statistics that account for edges not being independent (e.g., permutation tests, Network-Based Statistic).

The whole pipeline in one line:

> signals → table of relationships → keep the strong ones → network → count its properties

---

## 5. Graph metrics

| Metric | Definition | Interpretation |
|---|---|---|
| Degree / strength | Number (or total weight) of a node's links | How connected a region is |
| Clustering coefficient | Fraction of a node's neighbors that are linked to each other | Local, specialized processing |
| Path length | Average number of steps between nodes | How easily information is integrated |
| Global efficiency | Inverse of path length | Same, more robust when the network is fragmented |
| Density | Fraction of all possible links that exist | How sparse the whole network is |
| Betweenness | How many shortest paths pass through a node | Bridge / bottleneck role |
| Modularity | Strength of division into dense groups | Functional subsystems |
| Participation coefficient | How spread a node's links are across modules | Connector vs local hub |

### Clustering coefficient

**Question**: are a node's neighbors also connected to each other?

**Analogy**: your friends. If they all know each other, you are in a tight group (high clustering). If none of them know each other, you connect separate worlds (low clustering).

**Formula**, for a node with k neighbors:
1. Possible links among the neighbors: k(k−1)/2.
2. Count the links that actually exist among them.
3. C = actual / possible (0 to 1).

**Worked example**: node X has neighbors A, B, C, D. Links among them: A-B and A-C.

```
      A ─── B
      │  ╲
      │    X
      │  ╱  ╲
      C      D
```

- Possible: 4 × 3 / 2 = 6.
- Actual: 2.
- C(X) = 2/6 = 0.33.

Each link between two neighbors closes a **triangle** with X, so the clustering coefficient counts triangles.

**Hub example**: a hub between the visual and motor networks has 10 neighbors and none of them are connected.
- Possible: 10 × 9 / 2 = 45. Actual: 0. **C = 0**.
- Meaning: it is a **bridge**, not part of a local group. Its role is integration. It has high betweenness, and damaging it disconnects the two networks. This is why hub damage in stroke or Alzheimer's causes outsized deficits. Real hubs are rarely this extreme.

### Clustering is not density

- **Density is global**: fraction of all possible links in the whole network that exist.
- **Clustering is local**: fraction of possible links among each node's neighbors that exist.

**Ring example**: people around a round table, each knowing only the 2 people on their left and the 2 on their right (4 neighbors).

```
   ... L2 ── L1 ── X ── R1 ── R2 ...
```

Do X's neighbors know each other?
- L2-L1 (side by side): yes
- L1-R1 (2 seats apart): yes
- R1-R2 (side by side): yes
- L2-R1, L1-R2, L2-R2 (3 or more seats apart): no

3 of 6 → **clustering = 0.5**.

With 100 people, each knows 4 of the 99 others → **density = 200 / 4,950 ≈ 0.04**.

Clustering stays 0.5 no matter how big the table is. Density shrinks as the table grows.

**Star example**: one center linked to 99 outer nodes. Density ≈ 0.02, clustering = 0. Same low density as the ring, opposite clustering.

**Link between them**: in a random network, expected clustering ≈ density. So always compare clustering against random networks with the same number of nodes and links. The ring's clustering is about 12× higher than chance. That gap is the meaningful result.

### Pitfall specific to functional networks (verified)

Correlation is partly transitive: if A correlates with B and B with C, A and C tend to correlate even with no direct link. This creates triangles automatically, so clustering in correlation-based networks (fMRI, EEG) is partly inflated by the math.

### Main replicated findings

- **Small-world organization**: high clustering **and** short path length. Local specialization plus global integration at low wiring cost.
- **Modules**: consistent large-scale networks (default mode, salience, frontoparietal control, sensorimotor, visual).
- **Rich club**: hubs (precuneus, superior frontal, superior parietal, insula) are densely interconnected, forming a costly backbone.
- **Hubs are vulnerable**: pathology concentrates in hubs in Alzheimer's, schizophrenia and other disorders (verified). That this reflects their high metabolic cost is plausible but not proven (inference).

---

## 6. Modules

**Definition**: groups of nodes with many links **inside** the group and few links **between** groups.

**Analogy**: a university. Students mostly talk within their own department (modules). A few people bridge departments (connector hubs).

### Modularity (Q)

> Q = (fraction of links inside modules) − (fraction expected by chance in a random network with the same degrees)

- Q ≈ 0: no better than random.
- Higher Q: clearer modular structure.

An algorithm (usually **Louvain**) tries many splits and keeps the one with the highest Q.

### Resolution parameter (γ)

A **zoom knob** for module detection. It sets how strict the comparison against chance is.
- Low γ (zoomed out): a few big modules.
- High γ (zoomed in): many small modules.

**Analogy**: a map. Zoomed out you see continents, zoomed in you see countries, then cities. All are real. None is "the correct" level.

In the brain:
- Low resolution: ~2 modules (sensory/motor vs higher-order association).
- Medium: the 7 Yeo networks (2011): visual, somatomotor, dorsal attention, ventral attention/salience, limbic, frontoparietal control, default mode.
- High: 17 or more subnetworks.

If a paper reports "patients have 5 modules, controls have 7," this may only reflect the chosen γ. Good practice: test a range of γ and report what holds across it.

### Hub types

- **Provincial hubs**: many links, mostly within their own module.
- **Connector hubs**: links spread across many modules. Measured by the **participation coefficient** (0 = all links in one module, near 1 = spread evenly).

### Caveats

- Louvain is partly random. Runs differ. Use many runs and take a consensus.
- The brain is hierarchical (modules within modules), so there is no single true number of modules.

---

## 7. Effective connectivity: Granger, transfer entropy, DCM

The core idea for detecting direction is **time**: a cause comes before its effect.

### 7.1 Granger causality

**One sentence**: if A's past helps predict B's future beyond what B's own past already tells you, A "Granger-causes" B.

**Step 1.** Predict B from its own past only (a regression):
> B(now) ≈ a₁·B(1 step ago) + a₂·B(2 steps ago) + ... + error₁

**Step 2.** Predict B from its own past plus A's past:
> B(now) ≈ a₁·B(1 ago) + a₂·B(2 ago) + **c₁·A(1 ago) + c₂·A(2 ago)** + error₂

**Step 3.** Compare errors.
- error₂ clearly smaller → A's past carries information about B's future → **A → B**.
- About the same → no arrow.

Effect size: **GC(A→B) = log(var(error₁) / var(error₂))**. 0 means no effect. An F-test or permutation test decides significance.

**Step 4.** Test the reverse direction separately. Possible outcomes: A→B, B→A, both, neither.

**Example**: a seizure starts in the left hippocampus and reaches temporal cortex ~20 ms later. The hippocampus's past improves prediction of the cortex, not the reverse.

**Variants**
- **Conditional Granger**: tests A→B while including C's past, removing arrows caused by a recorded common driver.
- **Spectral Granger**: split by frequency ("A drives B in alpha, not gamma"). Common in EEG/LFP.

**Where it breaks (verified)**
1. An **unrecorded driver** creates fake arrows.
2. It is **linear** and misses nonlinear influence.
3. **Unequal noise** can flip direction: if B is recorded more cleanly than A, you can get a false B→A.
4. **Filtering and slow sampling** can distort or reverse time order.
5. **EEG volume conduction** spreads one source to many electrodes instantly, creating fake connectivity. Work in source space.

Granger measures **prediction from the past**, not true causation. The name is misleading.

### 7.2 Transfer entropy

Same question as Granger, but measured as **information** (bits) instead of prediction error.

**Entropy** = uncertainty. A fair coin has 1 bit. A coin that always lands heads has 0 bits.

> TE(A→B) = how much knowing A's past reduces uncertainty about B's next value, beyond B's own past.

| | Granger | Transfer entropy |
|---|---|---|
| Baseline | Predict B from B's past | Uncertainty about B given B's past |
| Add A's past | Does error drop? | Does uncertainty drop? |
| Output | Ratio of errors | Bits |
| Assumes | Linear relationship | Nothing about the shape |

**Example Granger misses**: B(now) = A(1 step ago)². When A was +3 or −3, B is 9 either way. A linear regression finds nothing (positive and negative cancel). TE detects strong dependence.

**Verified facts**
- For linear systems with Gaussian noise, TE and Granger are equivalent (TE = half the Granger value). Granger is TE's linear special case.
- TE needs much more data and its estimation is hard (depends on binning or estimator).
- It inherits all Granger pitfalls.

**Practical rule**: start with Granger. Use TE with lots of data and a reason to expect nonlinear coupling.

### 7.3 DCM (Dynamic Causal Modeling)

Instead of scanning data for arrows, you **write candidate wiring diagrams in advance**, simulate the data each would produce, and let the data choose. Granger and TE are data-driven. DCM is hypothesis-driven.

**Part 1: neural model** (how hidden activity evolves), for a few regions:
- **Fixed connections**: A always drives B with some strength.
- **Modulatory connections**: a task condition changes a connection (e.g., attention strengthens V1 → V5).
- **Inputs**: where stimuli enter (e.g., V1).

**Part 2: observation model** (how hidden activity becomes measurements):
- fMRI: hemodynamic model (neural activity → slow blood signal).
- EEG/MEG: neural mass model (excitatory and inhibitory populations with synaptic time constants) plus a forward model to the sensors.

**Procedure**
1. Choose 3 to 8 regions.
2. Write candidate models, e.g.:
   - Model 1: attention boosts V1 → V5.
   - Model 2: attention boosts parietal → V5 (top-down).
   - Model 3: both.
3. Fit each model.
4. Compare by **model evidence**: fit minus a complexity penalty.
5. Read the winner's parameters.

**Strengths**: parameters have biological meaning. In EEG DCM, synaptic gains and receptor time constants can be estimated (some studies infer NMDA receptor function from scalp EEG). Works with fMRI, where Granger is weak.

**Limits**: only chooses among models you wrote, so a winner is crowned even if all are wrong. Few regions only. Results depend on built-in assumptions. Main software: SPM (MATLAB).

### 7.4 Comparison

| | Granger | Transfer entropy | DCM |
|---|---|---|---|
| Approach | Data-driven | Data-driven | Hypothesis-driven |
| Relationship | Linear | Any | Biophysical model |
| Regions | Many | Many (data hungry) | Few |
| Output | Arrows and strengths | Arrows in bits | Best model + biological parameters |
| Main risk | Hidden drivers, noise | Data hunger, estimation | Wrong candidate models |

---

## 8. Where chemistry fits

Standard graph models mostly **ignore chemistry**. This is a real limitation.

Two kinds of chemical signaling:
1. **Fast transmission (glutamate, GABA)**: the signal travelling along edges. At the macro scale it is averaged into connection strength.
2. **Neuromodulation (dopamine, noradrenaline, acetylcholine, serotonin)**: changes **how nodes respond**, like a gain knob.

> Same wiring, different chemistry, different functional network.

**Analogy**: the road map (structure) stays fixed. Neuromodulators are traffic control: which roads are open, which lights are green. Traffic patterns (functional and effective connectivity) can reorganize in seconds with no wire changing.

**How researchers include chemistry**
1. **Receptor maps as node properties**: PET gives receptor/transporter density per region. Public multi-receptor atlases exist.
2. **Pharmacological studies**: record before and after a drug. Example: under propofol anesthesia, frontal → parietal effective connectivity decreases, one of the better-replicated findings in consciousness research.
3. **Biophysical models**: DCM and neural mass models (e.g., Jansen-Rit for EEG) have parameters for synaptic gain and receptor function. The inference is indirect and depends on the model being right.

Also: glia, hormones and blood flow shape signals. fMRI measures blood oxygenation, an indirect proxy for neural activity.

---

## 9. Reading and writing the brain

**Read (decode)**: translate activity into something useful.
- Implanted BCIs: decoding intended movement and speech in paralyzed patients. Useful speech rates have been reached (verified).
- EEG BCIs: spellers, motor imagery. Non-invasive but low bandwidth.

**Write (modulate)**:
- Electrical: DBS (Parkinson's, tremor), TMS (depression), tDCS, vagus nerve stimulation.
- Chemical: drugs.
- Optogenetics: cell-type-specific control with light, animals only.

**Closed loop**: sense and stimulate together. Example: responsive neurostimulation detects seizure onset and stimulates to stop it.

### How good is decoding?

- **Verified**: with implanted arrays, decoding of movement and speech intent works well in narrow tasks.
- **Verified**: with EEG, decoding is poor: few commands, slow, unstable across sessions.
- **For both**: abstract content (general intentions, emotions, thoughts) is decoded badly, and decoders drift over days.

### Why effective connectivity matters for intervention

- Stimulate a **source** (a region that drives others): the effect spreads.
- Stimulate a **sink** (a region that only receives): the effect may stay local.

**Network control theory** formalizes this: given the wiring, which nodes can steer the system into a target state? The idea is solid. Clinical evidence that connectivity-guided targeting beats standard targeting is still limited. The best case is TMS for depression, targeting a prefrontal spot chosen by its connectivity with the subgenual cingulate.

---

## 10. State space, manifolds and attractors

### State space

Record 3 neurons. At each instant, their 3 firing rates are coordinates of a point:

```
moment 1: (5, 20, 3)   → a point in 3D space
moment 2: (6, 18, 4)   → a nearby point
moment 3: (8, 15, 6)   → ...
```

Activity becomes a **point moving through space**. Its path is a **trajectory**. With 100 neurons the space has 100 dimensions. The math is the same.

### The manifold: where activity can be

**Key finding**: activity of 100 neurons does not wander through all 100 dimensions. It stays on a much smaller surface, often ~5 to 15 dimensions. That surface is the **neural manifold**.

**Why**: neurons are connected. If A excites B, B cannot do anything it likes. Wiring forces neurons to move together.

**Puppet analogy**: a puppet with 20 joints controlled by 3 strings. Joints have 20 coordinates, but every pose is determined by 3 string positions. The puppet "lives" on a 3D manifold inside a 20D space. Neurons = joints. Strings = **latent variables**.

**What the manifold is built from**: in practice, smoothed firing rates (also calcium signals or LFP). It leaves out exact spike timing, subthreshold voltage and chemistry. What matters is not each rate alone but which **combinations** of rates occur together.

**Verified**: in motor cortex, the same low-dimensional structure appears across days and across animals, even though the specific neurons recorded change.

### Manifold vs dynamics: two different objects

- **Manifold**: **where** activity can be. A shape.
- **Dynamics**: the **rule of movement** on that shape. From the current state, where does activity go next?

**Analogy**: a mountain road network. The manifold is the roads (cars never drive through the mountain). The dynamics are the traffic flow on those roads. The manifold is a curved surface, not a grid.

### How to build the manifold

1. Record spikes from N neurons over many trials.
2. Smooth spikes into firing rates.
3. Stack into a matrix: rows = time points, columns = neurons. Each row is a point in N-dimensional space.
4. Find the shape the points lie on (dimensionality reduction):
   - **PCA**: best flat surface. The usual first step.
   - **Nonlinear methods** (Isomap, UMAP): curved surfaces.
   - **Autoencoders**: the bottleneck coordinates are the manifold.
5. Count dimensions needed to explain most of the variance.

**Tiny example**: if neuron C always equals A + B, every recorded point (A, B, C) lies on a flat sheet in 3D. PCA finds 2 dimensions, not 3.

These steps give only the manifold. Getting the **dynamics** requires a model of how points move over time (e.g., LFADS).

### Attractors: where activity tends to go

Picture the dynamics as a **landscape of hills and valleys**, with activity as a ball rolling on it. An **attractor** is where the ball settles or keeps returning.

| Type | Picture | Brain example |
|---|---|---|
| Fixed point | A valley. The ball rolls in and stops | Memory retrieval: a partial cue rolls into the stored pattern (Hopfield networks) |
| Line / ring attractor | A circular trough. The ball stays wherever placed along it | Head direction: in the fruit fly, a literal ring of neurons holds heading (observed directly) |
| Limit cycle | A racetrack. The ball keeps circling | Rhythms, walking, breathing. Rotational dynamics in motor cortex during reaching |

**Unstable points**: hilltops and saddles (ridges between valleys). Nearby starting points can end in different valleys, which works as a decision boundary.

**Clinical intuition (inference)**: a seizure can be modeled as the brain falling into a pathological attractor: self-sustaining, and sometimes escapable with a well-timed perturbation. A working model in epilepsy modeling, not a proven mechanism for every seizure type.

### How we know which type an attractor is

Two separate questions: (a) what type, (b) what it represents.

**(a) Finding the type.** A fixed point is where the flow stops (speed zero). A cycle is a closed loop the flow keeps going around.
1. **Watch trajectories**: settles and stays → fixed point. Keeps looping → cycle. Can stay anywhere along a line/ring → line or ring attractor.
2. **Perturb and watch recovery**: returns to the same spot → stable fixed point. Returns to the loop → cycle. Stays where pushed along the ring → ring attractor. The most convincing evidence.
3. **Fit a model, then search it** (Sussillo & Barak, 2013):
   1. Train an RNN or LFADS on the data.
   2. Search for states where speed ≈ 0 (candidate fixed points).
   3. Nudge each in every direction: all nudges fade → stable (valley); some grow → unstable (ridge or hilltop); nudges spiral → rotation. Mathematically: eigenvalues of the linearized system.

**(b) Finding the meaning**: align attractors with behavior.
- Fly head direction: the bump's position on the ring matches the fly's heading. Turn the fly, the bump moves by the same angle (verified).
- Working memory: a different stable state per remembered item during the delay.
- Decisions: two valleys, one per choice. The valley reached predicts the choice.

**Caveat**: real data usually reveals **slow points** (very slow flow), not exact fixed points. Over a task's time scale they behave the same. Model attractors must be checked with perturbation to rule out training artifacts.

### How everything connects

```
Wiring (structural connectivity)
   ↓ sets
Who drives whom (effective connectivity)
   ↓ defines
The landscape (flow field with attractors)
   ↓ restricts activity to
The manifold (where activity lives)
   ↓ is what you should
Decode
```

Effective connectivity and attractors are two views of the same thing: the forces between nodes, versus the flow those forces produce. Neuromodulation reshapes the landscape (deepens, flattens, or moves valleys).

### Why this explains poor decoding

Classic decoders treat each neuron/channel as an independent feature. Problems:
1. You record a tiny, changing sample of neurons. The decoder breaks.
2. Single neurons are noisy and drift. The manifold is more stable.
3. Ignoring dynamics throws away information: the state at t largely predicts t+1.

Manifold-based fixes:
- Decode from latent variables, not individual neurons.
- **Realign the manifold** across days instead of retraining (verified to restore performance).
- Use learned dynamics as a prior (LFADS).

### Warning: "manifold" means two things in EEG papers

- **Neural manifold** (above): where brain activity lives. A claim about the brain.
- **Riemannian manifold**: covariance matrices have curved geometry, so distances between them should use curved-space math. A claim about the data's math. Riemannian classifiers are among the strongest standard EEG BCI methods.

### The physical limit for EEG

- **Verified**: manifold and attractor results come mostly from invasive recordings of neuron populations.
- **Inference**: EEG sums millions of neurons through the skull. Low-dimensional brain states remain visible (microstates, sleep stages), but the fine manifold of a cortical area cannot be recovered. This is the main physical reason EEG decoding stays weak.

---

## 11. Inside vs outside the manifold

### What it means

Use the 3-neuron example where wiring forces **C = A + B**. All activity lies on that sheet.
- **Inside**: A = 5, B = 3, C = 8. Obeys the coupling. Easy to produce.
- **Outside**: A = 5, B = 3, C = 0. Breaks the coupling. The wiring does not allow it.

### How to put a decoder outside the manifold

A decoder is a rule mapping rates to cursor movement:

> cursor velocity = w_A·rate_A + w_B·rate_B + w_C·rate_C

The weights define a **direction in neuron space**. The cursor moves only if activity has a component along that direction.

- **Inside-manifold decoder**: weights (1, 0, 1), reads A + C. When A rises, C rises too, so the cursor moves using patterns the subject already makes.
- **Outside-manifold decoder**: weights (1, 1, −1), reads A + B − C. On the sheet, C = A + B, so this sum is **always zero**. The decoder points straight out of the sheet. The cursor stays still unless the subject produces a forbidden pattern.

### The 2014 experiment (Sadtler et al., Nature; verified)

Monkeys controlled a cursor through a BCI recording ~90 neurons.
1. Find the manifold (~10 dimensions) with factor analysis, a relative of PCA.
2. **Inside perturbation**: shuffle which manifold dimension drives which cursor direction. Learned within a session.
3. **Outside perturbation**: shuffle which neuron feeds which weight, breaking the natural coupling. Mostly not learned in one session.

**Conclusion**: the manifold is a real constraint on what the brain can produce in the short term, not just a description of data.

---

## 12. Do we need to record every neuron?

The concern: we measure too little (hundreds to a few thousand neurons in humans, out of ~86 billion; EEG blurs millions together). Compared with an LLM, where every weight and activation is known and interpretation is still hard, the brain is worse: no weights, few activations, plus noise.

### Correction: a sample is enough if the system is low-dimensional

> If a system is truly low-dimensional, a random sample of its parts is enough to recover its structure.

Puppet: you do not need all 20 joints to discover 3 strings. Any 6 joints reveal that motion is explained by 3 numbers, because every joint reflects all strings. Similarly, 200 recorded neurons driven by 10 latents each reflect those 10 latents.

**Verified theory**: related to random projections and compressed sensing. The number of neurons needed depends on the manifold's dimension and smoothness, not on total neuron count.

### The real problems

1. **Low dimensionality may be partly a task artifact.** A monkey doing one reaching task for months uses a small slice of its repertoire. **Verified**: in mouse visual cortex, ~10,000 neurons recorded during natural images showed **high-dimensional** activity (Stringer et al., 2019). Refined statement: **connectivity sets the space available, the task decides how much of it you see.**
2. **Unobserved inputs.** Each area's "strings" may be pulled by another area you do not record. A model of the recorded area mistakes external input for internal dynamics. Partial fixes: record multiple areas simultaneously (Neuropixels), models with an explicit input term (LFADS), and perturbation instead of observation.

### How close are we to recording everything (verified)

- **C. elegans** (302 neurons): full wiring, near whole-brain activity.
- **Larval zebrafish** (~100,000 neurons): near whole-brain imaging.
- **Fruit fly**: full wiring diagram (~140,000 neurons, 2024). Synapse counts are not the same as weights: synapse strength, chemistry and neuromodulation are missing.
- **Mouse**: around a million neurons imaged at once, but slowly and only near the surface.
- **Human**: nowhere close.

---

## 13. Neural drift

Two different things:

1. **Recording drift (technical)**: electrodes shift, scar tissue forms, impedance changes. Neuron #12 on Monday is not neuron #12 on Friday.
2. **Representational drift (biological)**: **verified** in mouse hippocampus and olfactory cortex. A single neuron's response to the same stimulus changes over days and weeks while behavior stays stable. **Leading interpretation (inference)**: the code lives at the population level. Individual neurons change, the manifold's shape stays stable.

Both break neuron-by-neuron decoders. Both are partly fixed by decoding from the manifold and realigning it daily.

---

## 14. LFADS

**LFADS** = Latent Factor Analysis via Dynamical Systems (Pandarinath et al., Nature Methods, 2018). A sequential variational autoencoder.

A normal autoencoder compresses **one snapshot** into a few numbers. LFADS compresses **a whole trial** into a starting point plus a rule for how it evolves.

### Simple version: the pendulum behind a dirty window

A pendulum filmed through a dirty, flickering window. Each frame is noisy.
- Option 1: clean each frame separately. Hard, frames are mostly noise.
- Option 2 (LFADS): pendulums follow rules. Know where it started and the rules, and you can reconstruct the whole movement.

Four characters:
1. **The watcher (encoder)**: watches the whole trial once and writes down the starting point.
2. **The simulator (generator, an RNN)**: plays the movement forward from the starting point using learned rules. Never looks at the data again. **Its rules are the dynamics (the landscape).**
3. **The judge (loss function)**: compares simulation with data (Poisson likelihood, correct for spike counts). Watcher and simulator adjust. Repeat over thousands of trials.
4. **The helper (controller)**: can add **pushes** (inputs) to the simulation at any moment. Every push **costs** something, so it pushes only when the simulation is clearly wrong without one.

Mapping back: dirty video = noisy spikes. True pendulum motion = latent activity on the manifold. Swing rules = dynamics. A kid hitting the pendulum = another brain area you did not record.

### Technical architecture

```
spikes (whole trial)
      │
      ▼
 [1] ENCODER ──→ g0 (initial state)
                  │
                  ▼
 [2] GENERATOR (RNN), runs on its own: g0 → g1 → g2 → ...
        ▲             │
        │             ▼
 [5] CONTROLLER    [3] FACTORS (low-dim manifold coordinates)
   inputs u(t)        │
                      ▼
                  [4] RATES for every neuron → compared with real spikes
```

- **Generator = dynamics. Factors = manifold.** Both objects separated.
- **Loss = reconstruction + cost of inputs.** The model prefers to explain everything with internal dynamics and uses inputs only when needed. The pattern of inferred inputs estimates hidden inputs.

**Verified results**: on monkey reaching data, cleaner single-trial trajectories than standard smoothing and better hand-movement decoding. When the target changed unexpectedly mid-trial, inferred inputs appeared around the change.

### Limits

1. **The split between dynamics and inputs is a choice**, set by the input cost. Data alone cannot fully settle it.
2. **"Input" means "unpredictable by the model"**, not proof it came from another area.
3. Needs many trials of a repeated task.
4. Built for spikes. For EEG, the blurring problem remains.

### What if pushes were free?

The helper would explain every wiggle as a push. The simulator would have no reason to learn rules. You would get a perfect fit to the data, noise included, and no understanding.

---

## 15. Experiment: mini-LFADS and the hidden kick

**Goal**: test whether a minimal LFADS-style model can locate a hidden input it was never told about, and what the input cost does.

**Setup**
- 2 hidden latents moving on a two-valley landscape (valleys at x = ±1, ridge at x = 0).
- On half the trials, at time step 50, a **hidden kick** shoves x across the ridge.
- 30 neurons each read a different mix of the latents and fire Poisson spikes.
- Model: encoder (starting point), controller (pushes), GRU generator (dynamics), factors (manifold), rates.
- Loss: Poisson reconstruction + push_cost × mean |push|.

Real LFADS makes the starting point and pushes probabilistic and uses a KL penalty. The logic here is the same, simplified.

### Code

```python
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

# ---------- 1. Fake brain with known ground truth ----------
T, DT, N_NEURONS, N_TRIALS = 100, 0.1, 30, 600
KICK_TIME = 50

def flow(z):
    # two valleys at x=-1 and x=+1, ridge at x=0
    x, y = z[:, 0], z[:, 1]
    return np.stack([x - x**3, -y], axis=1)

z = np.zeros((N_TRIALS, T, 2))
z[:, 0, 0] = np.random.choice([-1.0, 1.0], N_TRIALS) + 0.1 * np.random.randn(N_TRIALS)
z[:, 0, 1] = np.random.randn(N_TRIALS)
kicked = np.random.rand(N_TRIALS) < 0.5          # trials that get the hidden input

for t in range(1, T):
    # next state = rule(current state) + small noise
    z[:, t] = z[:, t - 1] + DT * flow(z[:, t - 1]) + 0.02 * np.random.randn(N_TRIALS, 2)
    if t == KICK_TIME:
        z[kicked, t, 0] += -2.0 * np.sign(z[kicked, t, 0])   # shove to the other valley

readout = np.random.randn(2, N_NEURONS) * 0.8     # how each neuron mixes the latents
log_rates = z @ readout + np.log(0.5)
spikes = np.random.poisson(np.exp(log_rates))     # noisy spike counts

X = torch.tensor(spikes, dtype=torch.float32)

# ---------- 2. Mini-LFADS ----------
class MiniLFADS(nn.Module):
    def __init__(self, n_neurons, hidden=64, n_factors=4):
        super().__init__()
        self.encoder = nn.GRU(n_neurons, hidden, batch_first=True, bidirectional=True)
        self.to_g0 = nn.Linear(2 * hidden, hidden)                 # watcher
        self.controller = nn.GRU(n_neurons, hidden, batch_first=True, bidirectional=True)
        self.to_u = nn.Linear(2 * hidden, 1)                       # helper
        self.generator = nn.GRUCell(1, hidden)                     # simulator
        self.to_factors = nn.Linear(hidden, n_factors)             # manifold coordinates
        self.to_logrates = nn.Linear(n_factors, n_neurons)         # back to neurons

    def forward(self, x):
        _, h = self.encoder(x)
        g = self.to_g0(torch.cat([h[0], h[1]], dim=1))   # starting point
        c, _ = self.controller(x)
        u = self.to_u(c)                                  # one push per time step
        log_rates, factors = [], []
        for t in range(x.shape[1]):
            g = self.generator(u[:, t], g)                # simulator steps forward
            f = self.to_factors(g)
            factors.append(f)
            log_rates.append(self.to_logrates(f))
        return torch.stack(log_rates, 1), torch.stack(factors, 1), u.squeeze(-1)

# ---------- 3. Training with a given push cost ----------
def train_model(push_cost, epochs=400):
    model = MiniLFADS(N_NEURONS)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    poisson = nn.PoissonNLLLoss(log_input=True)          # the judge
    for ep in range(epochs):
        idx = torch.randint(0, 500, (64,))
        log_r, _, u = model(X[idx])
        loss = poisson(log_r, X[idx]) + push_cost * u.abs().mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model

# ---------- 4. Did it find the kick? ----------
def report(name, model):
    test = slice(500, 600)
    with torch.no_grad():
        log_r, factors, u = model(X[test])
    u = u.abs().numpy()
    k = kicked[test]
    window = slice(KICK_TIME - 2, KICK_TIME + 3)
    fit = np.corrcoef(np.exp(log_r.numpy()).ravel(), np.exp(log_rates[test]).ravel())[0, 1]
    print(f"\n{name}")
    print(f"  match to TRUE rates: {fit:.2f}")
    print(f"  push at kick time, kicked trials:   {u[k][:, window].mean():.3f}")
    print(f"  push at kick time, unkicked trials: {u[~k][:, window].mean():.3f}")
    print(f"  push elsewhere:                     {np.delete(u, range(48, 53), 1).mean():.3f}")

for c in [0.0, 0.03, 0.1, 0.3, 1.0]:
    report(f"push_cost={c}", train_model(push_cost=c))
```

Runs in about 1 minute per model on a CPU.

### Results

| Push cost | Match to true rates | Push at kick (kicked) | Push at kick (unkicked) | Push elsewhere |
|---|---|---|---|---|
| 0 (free) | 0.92 | 4.45 | 4.03 | 4.04 |
| 0.03 | 0.90 | **0.36** | 0.07 | 0.08 |
| 0.1 | 0.94 | 0.007 | 0.006 | 0.006 |
| 0.3 | 0.95 | 0.010 | 0.009 | 0.010 |
| 1.0 | 0.96 | 0.011 | 0.011 | 0.010 |

### Interpretation

1. **Free pushes**: pushes everywhere, equally on kicked and unkicked trials. They carry no information about the kick. Prediction confirmed.
2. **Small cost (0.03)**: works. Pushes at step 50 on kicked trials are ~5× larger than anywhere else. The model found the hidden input.
3. **Higher costs**: pushes vanish, yet the fit is the best (0.96). The model **cheats through the starting point**:
   - The watcher sees the whole trial, including the kick at step 50.
   - It writes "this trial flips at step 50" into the starting point.
   - The simulator learns an **internal timer**: part of its state counts down and flips the state at zero.
   - The kick is disguised as a rule.

This is the hidden-input problem made concrete: **external input and internal dynamics are interchangeable from data alone**, and the model chooses whichever is cheaper.

### Why restricting the watcher would fix it

If the watcher only sees the first 10 steps, it cannot know whether a trial will be kicked at step 50 (the kick is random and happens later). It cannot write the plan into the starting point, so the timer cheat becomes impossible. The only remaining way to explain the flip is a push at step 50.

**Analogy**: a fortune teller who has read the last page writes the ending into the first chapter. Tear out every page after chapter 1 and they cannot spoil it.

**How real LFADS handles it**: the encoder sees the whole trial, but information placed in the starting point also has a cost (KL penalty), so packing "flip at step 50" into the start becomes expensive. The simplified model had no such cost.

**Next experiment**: restrict the encoder to the first 10 steps and check whether pushes at step 50 return at high costs.

---

## 16. Statistics: permutation tests and cluster-based tests

### Permutation vs perturbation

- **Perturbation**: you interfere with the system (stimulate, lesion) and observe. An experiment.
- **Permutation**: you **shuffle labels in your data** to see what chance alone produces. A statistical test.

### Permutation test

**Question**: is the difference I found real or chance?

**Idea**: if there is truly no difference between groups, the group labels are meaningless. Shuffling them should not change anything. So shuffle many times and see how large differences get by chance.

**Example**: 20 patients and 20 healthy controls. Patients' mean clustering coefficient is 0.05 lower.
1. Real difference: 0.05.
2. Randomly shuffle the 40 "patient/control" labels. Recompute the difference (e.g., 0.01).
3. Repeat 5,000 times → the **null distribution** (what chance produces).
4. Count how often shuffled differences are ≥ 0.05:
   - 25 of 5,000 → p = 0.005. Unlikely to be chance.
   - 1,500 of 5,000 → p = 0.3. Chance produces this often.

**Why it is used in network science**: graph metrics follow no known distribution and edges are not independent. The test builds its null distribution from your own data, with no distribution assumptions.

### Cluster-based permutation test (as in MNE)

**What is measured**: an EEG experiment. 20 subjects see two kinds of pictures, e.g., familiar faces and unfamiliar faces, many trials each. Trials are averaged per subject and condition, giving one voltage curve (an ERP) per condition per subject: voltage in microvolts at each electrode, at each time point from 0 to 500 ms after the picture.

**The question**: at which electrodes and times do the two conditions differ?

**At each time point**: compare the 20 subjects' voltages between conditions with a t-test. The t value = size of the difference relative to its variability.

**The problem**: one electrode × 100 time points = 100 tests. At p < 0.05, ~5 are significant by pure chance. With 64 electrodes × 500 time points = 32,000 tests, ~1,600 false positives.

**The key observation**:
- Real brain effects **last**: tens of milliseconds, many consecutive time points, neighboring electrodes.
- Chance false positives are **scattered and short**.

So judge **stretches**, not single points.

**Step 1.** Mark time points with t above a threshold (e.g., t > 2):

```
time:  0 ──────── 110 ████████████ 160 ──── 300 █ 305 ──── 500 ms
                      stretch A               stretch B
```

**Step 2.** Score each stretch by summing its t values:
- Stretch A (110-160 ms): score 120.
- Stretch B (300-305 ms): score 12.

**Step 3.** Build the chance distribution. Shuffle the condition labels (for each subject, randomly swap which curve is called "familiar" and which "unfamiliar"; if conditions do not matter, this changes nothing). Redo steps 1 and 2, and record **only the biggest stretch score**. Repeat 1,000 times. Say 95% of these maximum scores are below 45.

**Step 4.** Compare:
- Stretch A: 120 > 45 → **significant**.
- Stretch B: 12 < 45 → not significant.

**With many electrodes**: a stretch can also spread to neighboring electrodes, giving a blob in space and time (e.g., "occipital electrodes, 110-160 ms").

**Why it solves the problem**: you compare only against the biggest chance cluster each time, so there is no need to correct for thousands of separate tests.

**Verified caveat (Sassenhagen & Draschkow, 2019)**: a significant cluster means "there is a real effect somewhere in it." It does **not** prove the effect starts exactly at 110 ms. Cluster edges depend on the threshold in step 1.

**Network-Based Statistic (NBS)**: the same idea applied to edges: clusters of connected significant edges in a network.

---

## 17. Confounds: when the result is not the brain

**Scenario**: patients (people with the disease) show lower modularity than controls (a healthy comparison group, ideally matched in age and sex), with p = 0.01. Is it their brains?

Not necessarily. **Data quality** can create the difference:
- **fMRI**: head motion changes many regions' signals at once, adding fake correlations unevenly. Verified: motion inflates short-range correlations and weakens long-range ones, which shifts network metrics.
- **EEG**: muscle and eye artifacts spread across many channels, adding shared signal everywhere and blurring module boundaries → lower modularity.

Patients usually move more and produce more artifacts than controls. "Lower modularity in patients" may mean "patients moved more."

**Fix**: measure motion/artifacts in every subject, check whether groups differ, then match, correct for it, or exclude bad recordings.

---

## 18. A map of brain research methods

Every study makes one choice on each of three axes.

### Axis 1: What is measured (recording)

| Method | Spatial resolution | Temporal resolution | Invasive? | Measures |
|---|---|---|---|---|
| Electron microscopy connectomics | Synapse | None (dead tissue) | Yes | Wiring only |
| Single units / Neuropixels / Utah array | Single neuron | ms | Yes | Spikes |
| Calcium imaging | Single neuron | ~100 ms | Yes (animals) | Activity, indirect |
| LFP | ~mm | ms | Yes | Summed local currents |
| ECoG | mm to cm | ms | Yes (on cortex) | Cortical surface activity |
| EEG / MEG | cm | ms | No | Summed activity through the skull |
| fMRI | mm | Seconds | No | Blood oxygenation (proxy) |
| fNIRS | cm | Seconds | No | Blood oxygenation, surface only |
| PET | cm | Minutes | Tracer injection | Metabolism, receptors |
| DTI | mm | None (static) | No | White matter tracts |

### Axis 2: Observe or interfere (manipulation)

- **Observe**: everything above.
- **Perturb**: lesions (including clinical stroke cases), TMS, tDCS, DBS, drugs, optogenetics, chemogenetics.

Only perturbation gives direct causal evidence. Observation gives correlation, or statistical guesses at causation (Granger).

### Axis 3: What question the analysis asks

| Question | Method family | Examples |
|---|---|---|
| Where? | Localization / activation | fMRI GLM, ERPs, lesion mapping |
| What information is there? | Representation / decoding | MVPA, decoders, BCI |
| How do regions relate? | **Connectivity** | Structural, functional, effective |
| How does activity evolve? | Dynamics | Manifolds, attractors, LFADS |
| What mechanism explains it? | Modeling / theory | Neural mass models, RNNs, biophysical models |

Examples:
- "EEG + resting state + wPLI network" = measure / observe / connectivity.
- "Utah array + decoder perturbation + manifold analysis" = the 2014 inside/outside-manifold experiment.

### Where connectivity lands

Connectivity is an **analysis-level** question (axis 3). Its three types need different choices on axes 1 and 2:

| Type | Comes from | Main methods |
|---|---|---|
| Structural | DTI, tracers, electron microscopy | Tractography, connectomes |
| Functional | Any time series (EEG, fMRI, LFP) | Correlation, coherence, wPLI |
| Effective | Time series **plus** a model, or a perturbation | Granger, transfer entropy, DCM, stimulate-and-record (gold standard) |

Within the analysis layer:

```
Localization   →   Connectivity   →   Dynamics    →   Mechanism
(which nodes)      (which edges)      (the flow)      (why that flow)
```

---

## 19. Limits: can we even understand networks?

**The concern**: AI researchers have every weight of a neural network and can perturb anything, yet full interpretability is not solved. The brain gives us far less.

**Where the concern is right (verified)**
- **"Could a neuroscientist understand a microprocessor?"** (Jonas & Kording, 2017): standard neuroscience methods (lesions, connectivity, tuning curves, dimensionality reduction) applied to a simple 1970s chip running video games produced convincing but **wrong** explanations. Removing certain transistors "abolished Donkey Kong," which sounds like "a Donkey Kong area" but is not.
- Full understanding of large AI models does not exist.

**Where it is outdated (verified)**
- Mechanistic interpretability has found specific circuits in transformers (e.g., "induction heads" that copy patterns from earlier text).
- **Sparse autoencoders** have extracted millions of interpretable features from production language models, and perturbing those features changes behavior predictably.
- Tools move both ways: the fixed-point method of Sussillo & Barak was developed on artificial RNNs and later applied to brains.

**Adjusted conclusion**
1. Graph theory alone will not explain the brain. It is a **description** of structure, one level among several. Dynamics, perturbation and mechanistic models are also needed.
2. **Useful does not require full understanding.** Speech BCIs decode words without a full theory of speech cortex. DBS treats Parkinson's though its mechanism is still debated.
3. Science asks "how does it work?" Engineering asks "can I reliably use it?" The second is achievable now. The first is open research.

---

## 20. US labs working on this

Affiliations marked (checked Nov 2026) were confirmed online. The rest are long-established groups listed from prior knowledge. Check lab pages before contacting.

### Latent dynamics, LFADS, decoding
- **Chethan Pandarinath**, Emory + Georgia Tech, Systems Neural Engineering Lab. Creator of LFADS, BrainGate member. (Checked Nov 2026: lab page lists Associate Professor at Emory/Georgia Tech with Emory Neurosurgery appointment. A Caltech neuroscience page under his name also exists; possible new affiliation, confirm directly.)
- **Eva Dyer**, UPenn, Computer and Information Science, joint with Bioengineering (checked Nov 2026: joined 2025 from Georgia Tech). Foundation models for neural population data across animals, tasks and species.
- **Scott Linderman**, Stanford, Statistics. Probabilistic models of neural dynamics (switching dynamical systems).
- **Liam Paninski**, Columbia. Statistical methods for neural data.

### Manifolds and BCI learning
- **Byron Yu and Steven Chase** (Carnegie Mellon) and **Aaron Batista** (University of Pittsburgh). The 2014 inside/outside-manifold experiment, later manifold alignment for stable BCIs.
- **Matthew Golub**, University of Washington. BCI learning and dynamics.
- **Lee Miller**, Northwestern. Long-term manifold stability, drift-robust decoders.

### Attractors and dynamics theory
- **Ila Fiete**, MIT. Attractor networks, grid cells, ring attractors. Theory-heavy.
- **Mehrdad Jazayeri**, MIT. Dynamics underlying timing and cognition, experiments plus RNN models.
- **Mark Churchland and Larry Abbott**, Columbia. Rotational dynamics in motor cortex, theory.
- **Kanaka Rajan**, Harvard Medical School, Neurobiology (checked Nov 2026). RNNs trained on neural data, attractor analysis.
- **Carlos Brody and David Tank**, Princeton. Attractors in decisions and memory.
- **SueYeon Chung**, NYU / Flatiron Institute. Geometry of neural manifolds.

### Attractors seen directly in small brains
- **Vivek Jayaraman and Ann Hermundstad**, Janelia. Fruit fly head-direction ring attractor.
- **Carsen Stringer and Marius Pachitariu**, Janelia. ~10,000-neuron recordings, high-dimensional cortical activity.

### Human BCI (clinical)
- **Frank Willett and Jaimie Henderson**, Stanford. Speech and handwriting BCIs (checked Nov 2026: 2025 inner-speech decoding in people with paralysis).
- **Sergey Stavisky and David Brandman**, UC Davis Neuroprosthetics Lab (checked Nov 2026: 2025 real-time speech BCI for a man with ALS).
- **Edward Chang**, UCSF. Speech decoding from ECoG.
- **BrainGate** consortium (Brown, MGH, Stanford, Emory). Leigh Hochberg leads the clinical trial.

Note: Krishna Shenoy (Stanford), a central figure in this field, died in 2023. Many of the people above trained with him.

### Fit notes for an MD with EEG and self-taught ML background
- Strongest fit: human BCI and latent-dynamics groups inside neurosurgery departments, where clinical knowledge is an asset.
- Weakest fit right now: theory-heavy attractor labs, which expect strong linear algebra, dynamical systems and probability.
- Gap to close: these labs mostly use spikes from implants. Useful bridging projects: latent-dynamics models on public spike datasets (Neural Latents Benchmark), or on ECoG/EEG with honest limits.
- Cold emails get answers when they show the lab's methods already run on the lab's data.

---

## 21. Resources

### Network neuroscience
- Fornito, Zalesky & Bullmore, *Fundamentals of Brain Network Analysis* (2016). The textbook.
- Bullmore & Sporns, "Complex brain networks," *Nature Reviews Neuroscience* (2009).
- Bassett & Sporns, "Network neuroscience," *Nature Neuroscience* (2017).

### Population dynamics
- Vyas et al., "Computation Through Neural Population Dynamics," *Annual Review of Neuroscience* (2020).
- Pandarinath et al., "Inferring single-trial neural population dynamics using sequential auto-encoders," *Nature Methods* (2018). [LFADS paper](https://www.nature.com/articles/s41592-018-0109-9.pdf)
- Sadtler et al., "Neural constraints on learning," *Nature* (2014).
- Sussillo & Barak, "Opening the black box," *Neural Computation* (2013).

### Statistics and limits
- Sassenhagen & Draschkow, "Cluster-based permutation tests of MEG/EEG data do not establish significance of effect latency or location," *Psychophysiology* (2019).
- Jonas & Kording, "Could a neuroscientist understand a microprocessor?" *PLoS Computational Biology* (2017).

### Courses and tools
- Neuromatch Academy, Computational Neuroscience (free, Python notebooks).
- `networkx` (general graphs), `bctpy` (Brain Connectivity Toolbox in Python), `mne` and `mne-connectivity` (EEG/MEG), SPM (DCM).

### Lab and news links
- [SNEL lab, Chethan Pandarinath](https://snel.gatech.edu/pi)
- [BrainGate: Chethan Pandarinath](https://www.braingate.org/team/chethan-pandarinath-phd/)
- [Penn CIS new faculty 2025 (Eva Dyer)](https://highlights.cis.upenn.edu/cis-welcomes-new-faculty-for-2025/)
- [Kanaka Rajan, Harvard Medical School](https://neuro.hms.harvard.edu/faculty-staff/kanaka-rajan)
- [Stanford Medicine: inner speech BCI (2025)](https://med.stanford.edu/news/all-news/2025/08/brain-computer-interface.html)
- [UC Davis Neuroprosthetics Lab](https://health.ucdavis.edu/neurological-surgery/research/labs/neuroprosthetics)
- [UC Davis: man with ALS speaks in real time (2025)](https://health.ucdavis.edu/news/headlines/first-of-its-kind-technology-helps-man-with-als-speak-in-real-time/2025/06)
- [CMU: Stabilizing brain-computer interfaces](https://www.cmu.edu/bme/News_Events/story_archive/bci-recalibration.html)

### Minimal clustering example

```python
import networkx as nx

G = nx.Graph()
G.add_edges_from([("X","A"), ("X","B"), ("X","C"), ("X","D"),
                  ("A","B"), ("A","C")])

print(nx.clustering(G, "X"))      # 0.333...
print(nx.average_clustering(G))   # whole-network average
```
