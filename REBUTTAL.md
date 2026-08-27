# Author Responses to the Reviews of Submission #57

We thank all three reviewers for their careful reads. This document gives the
full point-by-point responses; the OpenReview rebuttal note summarizes it
within the character limit. All numbers below come either from the released
CSVs (`results.csv`, `realdata.csv`, `fairness.csv`) or from two new
experiment CSVs added to this repository (`rebuttal57.csv`, 3,538 runs,
and `rebuttal_newdata.csv`, 42 runs; 3,580 new runs in total, produced by
`rebuttal_runner.py` and `rebuttal_newdata.py` and summarized by
`rebuttal_analysis.py`). Math is
written in plain text because this viewer does not render TeX.

While preparing this rebuttal we re-verified every numeric claim in the
submission against the released CSVs. That audit produced four corrections,
listed first; we prefer to report them ourselves. None changes any
conclusion.

## Corrections to the submitted PDF (self-reported)

**C1. "up to 0.94" should be "up to 0.93".** The largest GNN deficit
(Figure 2a minimum, best GNN minus best non-GNN) is -0.9346, at
(s, f) = (0.65, 0), which rounds to -0.93, not -0.94. The abstract,
Contribution 2, and Section 3 all say 0.94; the correct top-three deficits
are -0.93, -0.92, -0.89. (Reviewer 2Bv6's summary understandably echoes the
0.94; the correct value is 0.93.)

**C2. The win-band sentence is too tight.** Section 3 says the entire GNN
win band lies at s in [0.35, 0.65], f in [0.2, 0.65]. In fact the 18
GNN-win cells span s in [0.2, 0.65] and f in [0.2, 1.0]: eight of the 18
fall outside the printed band, all with wins of at most +0.0253. The
correct statement, which the camera-ready will carry: all wins exceeding
+0.03 NMI lie in s in [0.35, 0.55], f in [0.2, 0.65]; the remaining wins
are at most +0.029, seven of them at most +0.022 at f >= 0.8, where the
best GNN and the best non-GNN method are both within 0.023 of perfect
NMI (a related sub-0.005 rounding effect at the s = 0.65 edge of this region
is discussed in the response to Reviewer jYQg). The abstract's and
Contribution 2's "confined to a narrow band"
phrasing will carry the same "exceeding +0.03" qualifier.

**C3. "k-means alone achieves ~0.05" should be "~0.09".** Mean k-means NMI
in the f = 0.2 column is 0.091. The sentence's +0.89 lift and
"near-perfect recovery" are correct as printed; for the record, the peak
is at s = 0.95, where DMoN reaches 0.986.

**C4. Photo margin is +0.12, not +0.13.** Best classical (Leiden, 0.6570)
minus DMoN(ref) (0.5321) is 0.125; the +0.13 in Section 3 came from
subtracting already-rounded Table 1 entries. A second precision slip:
"Louvain and Leiden
recover the planted partition perfectly" holds exactly (mean NMI 1.000)
only at (s, f) = (0.8, 0) among the three deficit cells: at s = 0.65 they
score 0.995/0.998 and at s = 0.55 they score 0.956/0.955 (at s = 0.95,
f = 0, not a top-three deficit cell, both are also 1.000). The
camera-ready will say "score 0.95-1.00" there.

---

## Response to Reviewer 2Bv6 (rating 7)

We thank the reviewer for the accurate summary and positive assessment. One
number the review echoes from our text needed correction: the largest
classical-over-GNN margin is 0.93 NMI, not 0.94 (correction C1 above; our
error, not the reviewer's). The counts (classical 26 of 64, GNN 18) and the
+0.29 cap on GNN wins are confirmed exactly by the released CSVs. If any
point in the other responses or the new depth/tuning/dataset experiments
below raises questions, we are happy to follow up during the discussion
phase.

---

## Response to Reviewer a4zg (rating 4)

We thank the reviewer for the detailed engagement. We respond to the three
weaknesses and four question groups in order, and we ran new experiments
for W2/Q2 (encoder depth), Q3 (per-regime tuning), and W3 (more datasets).
Q1 asks for our perspective on the weaknesses; the three weakness
responses below are that answer.

**W1 (novelty).** We agree the qualitative direction is anticipated, and
the submission does not claim otherwise. The DMoN paper's own real-data
tables show non-neural pipelines winning cells (DGI + k-means beats DMoN on
Cora NMI 52.7 vs 48.8 and on Citeseer; Tsitsulin et al., JMLR 2023, Table
4), and for supervised settings Baranwal et al. (ICML 2021; ICLR 2023)
prove that graph convolution improves classification when edges are
class-informative: their guarantees require the intra/inter edge-density
gap (p - q)/(p + q) to be bounded away from zero, and the class-mean
separation contracts by exactly that factor, so the benefit vanishes, and
can reverse, as edges lose class information, which predicts the
direction of the random-graph failure. What we could not
find anywhere, and what the abstract contributes, is the quantified joint
map for unsupervised clustering: prior synthetic evaluations vary one axis
at a time (the DMoN paper's six synthetic scenarios are 1-D sweeps, without
Louvain/Leiden baselines, and conclude uniform DMoN superiority), while the
works closest in spirit measure different things (Shchur et al. 2018:
semi-supervised node classification; Leeney & McConville 2024: protocol
randomness; Goel et al. 2026: perturbation robustness; Deshpande et al.
2018 and Duranthon & Zdeborova 2024: asymptotic thresholds and
Bayes-optimality gaps without practical clustering pipelines). The
specific findings, that the win region is a narrow band at the classical
detectability cliff, that the best-case win (+0.29) is 3x smaller than the
worst-case loss (-0.93), and that the graph channel is an amplifier rather
than a source, have to our knowledge no prior quantitative statement. We
will sharpen the contribution wording to "the first controlled
two-knob decomposition with margins" rather than any claim of qualitative
surprise; if prior work contains such a map, we would welcome the pointer
and will cite it.

**W2 + Q2 (one GCN layer; deeper GNNs).** Two answers, one on fidelity and
one empirical, plus a scoping commitment.

*Fidelity.* The 1-layer GCN encoder is not our simplification; it is the
published configuration of both systems studied. The official DMoN
implementation defaults to a single 64-unit GCN layer (architecture flag
[64]), and the DMoN paper fixes one hidden layer for all GNNs (64 units
on synthetic graphs, 512 on real data; JMLR 2023, Parameter settings).
MinCutPool's own node-clustering experiments use "a one-layer
GNN followed by a single-layer MLP" (Bianchi et al., ICML 2020, Sec. 5.1).
Our study asks whether the published reference systems beat cheap
baselines, so we match their published configurations, including the skip
connection that lets even the 1-layer model exploit features when the graph
is noise.

*Empirical.* For this rebuttal we ran the full 8x8 grid x 5 seeds with 2-
and 3-layer encoders for BOTH DMoN (reference config) and MinCut, plus
depth 2/3 on all three real datasets (1,298 new runs; `rebuttal57.csv`,
summarized by `rebuttal_analysis.py`). Four results:

(i) The map's geometry is robust to depth. Family win counts: depth 2
gives classical 24 / k-means 19 / GNN 17 / none 4 (vs 26/17/18/3 at depth
1); depth 3 gives 24/22/15/3. At every depth, every GNN win exceeding
+0.03 NMI stays inside the same band, s in [0.35, 0.55] and f in
[0.2, 0.65]; the largest win grows modestly, +0.29 (depth 1) → +0.32
(depth 2) → +0.34 (depth 3). Even granting the GNN family the best depth
per cell (an oracle over all six GNN configurations) yields 20 GNN wins
vs 24 classical, max win +0.34.

(ii) The failure modes persist or deepen. The structureless-graph poison
is depth-independent for DMoN: at s = 0.05 the deeper models sit up to
0.74 below their own graph-blind control (vs 0.71 at depth 1). MinCut
collapses outright in that row at every depth, including depth 1: mean
NMI 0.000 vs 1.000 for k-means at (s, f) = (0.05, 1.5), a -1.00 gap for
MinCut specifically; depth does not rescue it.
In the f = 0 column the worst deficit is -0.83 at depth 2 (s = 0.45) vs
-0.93 at depth 1.

(iii) One finding is depth-sensitive, and we will scope it. With f = 0
and strong structure, deeper DMoN extracts almost everything: at
(s, f) = (0.95, 0), mean NMI 0.999 (depth 2) and 0.994 (depth 3) vs
0.293 at depth 1. This creates no new wins, because Louvain already reads
those graphs at 1.000 (the deficit there just shrinks to about zero), and
the deficits stay catastrophic at mid structure, but the "weak source
(<= 0.30 alone)" statement in Contribution 3 is a property of the
reference 1-layer configuration; the camera-ready will say exactly that.
The amplifier and poison halves of the mechanism are unchanged by depth.

(iv) Real data: Cora 0.399 ± .011 (depth 2) and 0.403 ± .032 (depth 3) vs
0.459 ± .021 (depth 1, `fairness.csv`); Citeseer 0.237 ± .008 and 0.213 ± .019 vs
0.292 ± .006; Photo improves to 0.582 ± .007 (depth 2) and 0.552 ± .024
(depth 3) vs 0.532 ± .031 but stays below the best classical (Leiden,
0.657). Classical still wins all three at every depth at the level of
seed-averaged NMI (no depth-2/3 seed beats the classical mean; one depth-1
Cora seed reaches 0.489, above Cora's classical mean of 0.466). So the
conclusions the
reviewer worried were 1-layer artifacts survive depths 2 and 3
unchanged, with one scoped exception, (iii), that we will state.

*Scope of conclusions.* We agree the Conclusion should not say "GNN
clustering" unqualified. The camera-ready will scope every conclusion to
"the studied GNN pooling methods (DMoN and MinCutPool at their reference
and deeper configurations)".

**W3 (only three small real datasets).** We agree this is a limitation of a
4-page extended abstract, and we ran the full Table 1 battery (Louvain,
Leiden, k-means, MinCut, DMoN reference, noise-X and rewired ablations, 3
seeds) on two additional datasets, Amazon Computers (n = 13,752) and
Coauthor CS (n = 18,333) (`rebuttal_newdata.csv`). The two land on
opposite sides of the map, and both match its prediction.

Amazon Computers is another structure-dominant graph: Louvain
0.529 ± .003 vs k-means 0.174 ± .014. Classical wins again: best GNN is
MinCut at 0.403 ± .012 (DMoN reference 0.331 ± .009), 0.13 below Louvain.
The channel ablations behave as before (noise-X 0.277 ± .019, rewired
0.011 ± .001). Its nearest map cell by the two cheap measurements is
(s, f) = (0.45, 0.2), a classical cell; we note that Computers
sits near the map's steepest edge (Louvain 0.53 falls between the
s = 0.35 and s = 0.45 rows, which straddle the detectability cliff), so
the placement is coarse, and the observed outcome matches the nearest
cell.

Coauthor CS is the informative one: its channel measurements are
mid-strength and complementary (Louvain 0.578 ± .013, k-means
0.661 ± .005, neither near ceiling), which places it INSIDE the synthetic
win band. All four neighboring map cells (s in {0.35, 0.45} x f in
{0.35, 0.5}) are GNN-win cells (+0.01 to +0.29; nearest cell (0.45, 0.5)
predicts +0.10), and on the real graph the GNNs indeed win: DMoN
reference 0.756 ± .015 and MinCut 0.723 ± .017 vs best cheap baseline
0.661 (k-means), a +0.095 margin, positive in every seed (min DMoN 0.734
vs max k-means 0.668). This is the first real-data instance of the win
band in our study, appearing where the map says it should with close to
the predicted margin, and it sharpens rather than weakens the thesis:
the GNN's value is conditional and measurable, in both directions. The
ablations also replicate the poison result on real data: rewired CS is
0.351 ± .027, i.e. 0.31 below simply ignoring the graph.

The full battery (mean NMI ± std over 3 seeds, `rebuttal_newdata.csv`):

| method | Computers | CS |
|---|---|---|
| Louvain | 0.529 ± .003 | 0.578 ± .013 |
| Leiden | 0.523 ± .014 | 0.591 ± .007 |
| k-means (features) | 0.174 ± .014 | 0.661 ± .005 |
| MinCut (fixed) | 0.403 ± .012 | 0.723 ± .017 |
| DMoN (ref) | 0.331 ± .009 | 0.756 ± .015 |
| DMoN noise-X | 0.277 ± .019 | 0.204 ± .005 |
| DMoN rewired | 0.011 ± .001 | 0.351 ± .027 |

Three camera-ready consequences. (1) Table 1 gains both columns, and the
"real benchmarks live where classical wins" heading, the abstract's
"classical methods never lose", and Contribution 4 are all rescoped to
the four structure-dominant benchmarks, with CS reported as the win-band
case. (2) The CS result exposes a wording gap in our Discussion protocol
("if either channel alone is strong, the GNN will at best match it"): CS's
k-means reads 0.661, which one could call strong, yet DMoN beats it by
+0.095. The intended meaning of "strong" is "near ceiling", exactly as in
the synthetic band, where k-means reads as high as 0.93 (the f = 0.65
column) in cells the GNN still wins; the camera-ready will make the
protocol quantitative ("if either channel alone is near ceiling, the GNN
will at best match it"). We thank the reviewer for W3, which prompted this
test. (3) On Computers the noise-X
channel reaches 0.277 vs Louvain's 0.529 (52 percent), so "never exceeds
half of what Louvain extracts" will become the per-dataset ratios (20, 7,
50, 52, 35 percent on Cora, Citeseer, Photo, Computers, CS).

**Q3 (tuning per regime).** We ran a per-regime tuning study: all
combinations of learning rate {1e-3, 1e-2} x dropout {0, 0.5} x hidden
width {64, 128} for the reference DMoN, on every one of the 64 cells, 5
seeds each (2,240 new runs, `rebuttal57.csv`; the eighth combination is
the reference configuration itself, reused from `results.csv`). The
tuning grid covers DMoN, the paper's headline GNN; MinCut received the
depth study of W2/Q2 rather than the tuning grid. We then gave
DMoN the per-cell ORACLE configuration (the config with the best mean NMI in
each cell,
selected using the ground-truth labels). Note this oracle is an upper
bound no practitioner can realize twice over: unsupervised model
selection cannot use NMI, and picking the max of eight noisy means also
collects selection noise.

Results: the oracle changes the numbers but not the conclusions. Mean
oracle gain over the reference config is +0.023 NMI (above +0.01 in 25
of 64 cells, above +0.05 in 11); the largest gains sit in the
feature-free column (+0.22 at s = 0.8, f = 0, via lr 1e-2) and at the
band edge (+0.11 at s in {0.35, 0.45}, f = 0.2). The oracle-tuned map:
classical 25 / k-means 15 / GNN 21 / none 3 (vs 26/17/18/3); the largest GNN
win grows from +0.29 (at s = 0.35, f = 0.35) to +0.33 (at s = 0.35, f =
0.2), and wins above
+0.03 now include two cells at s = 0.2, so the band's lower edge moves
down one row. Everything else is stable: the worst GNN deficit stays
catastrophic (-0.86 at s = 0.55, f = 0), the structureless-graph poison
stays (-0.68 at s = 0.05, vs -0.71 untuned; no configuration in the grid
closes it), and the 1-layer f = 0
weak-source column rises only to 0.46. So under per-regime oracle tuning
the GNN's best case improves by about +0.04 and every failure mode
persists; the asymmetry conclusion is unchanged.

**Q4.1 (state the Gaussian std).** Agreed. Exact feature model: community
centers mu_1..mu_k are drawn iid from N(0, f^2 I_32); node i's features are
x_i = mu_{y_i} + eps_i with eps_i ~ N(0, I_32). So the within-community std
is 1 in every coordinate and f is the ratio of center spread to
within-community noise. The camera-ready states this in Section 2.

**Q4.2 (k-means initialization).** scikit-learn KMeans with k-means++
initialization, n_init = 10 restarts (best inertia kept), and a fixed
random_state per run (the run's seed). Will be stated.

**Q4.3 (Table 1 caption).** Agreed on all three points. noise-X replaces
the feature matrix by iid N(0, 1) entries, the same distribution for every
node regardless of class (so features carry zero class information);
rewired replaces the graph by a uniform random graph with the same node
and edge counts (networkx gnm_random_graph, G(n, m)); the synthetic
generator parameters will be restated in the caption.

**Q4.4 (wording).** Agreed; "the official objective trusts the graph
harder and pays for it" will become: "the reference objective weights the
structure channel more heavily, and when the graph is uninformative this
weighting becomes a cost."

We hope the fidelity point, the new depth, tuning, and dataset results, and
the agreed scoping address the substance of the concerns, and we are happy
to run further variations the reviewer considers decisive during the
discussion phase.

---

## Response to Reviewer jYQg (rating 5)

We thank the reviewer for the constructive read and the sharp catch on the
figures.

**W1 + Q1 (exact synthetic specification).** The generator, in full. Graph:
planted-partition SBM on n = 1000 nodes with k = 4 equal communities and
average degree 16; the structure knob s in [0, 1] sets the cross-community
edge fraction mu = (1 - s)(1 - 1/k), giving within/between edge
probabilities p_in = (1 - mu) * 16 / (n/k - 1) and
p_out = mu * 16 / (n - n/k); s = 1 is perfect blocks, s = 0 is statistically
indistinguishable from Erdos-Renyi. Features: community centers mu_1..mu_4
drawn iid from N(0, f^2 I_32); x_i = mu_{y_i} + eps_i, eps_i ~ N(0, I_32),
so f is the ratio of between-community center spread to within-community
noise (std 1). The camera-ready adds both formulas to Section 2 (Reviewer
a4zg asked for the same).

**W2 (method descriptions).** Agreed; the camera-ready adds one sentence
each: Louvain and Leiden greedily maximize modularity over graph partitions
(structure only; Leiden adds a refinement step guaranteeing well-connected
communities); k-means clusters raw features (graph-blind); DMoN trains a
GCN with a soft assignment head by maximizing a spectral relaxation of
modularity plus a collapse regularizer; MinCutPool trains the same encoder
with a relaxed normalized-cut objective plus an orthogonality term.

**W3 + Q3 (Figure 1 vs Figure 2 at (f, s) = (1, 0.65); ties).** The two
panels are consistent; the apparent conflict is two-decimal rounding of a
genuinely positive but tiny delta. At that cell, (s, f) = (0.65, 1) in our
axis order, both GNNs reach mean NMI
1.0000 (5/5 seeds each) while the best non-GNN methods (Leiden and k-means)
reach 0.9981 (each misses perfect recovery on two seeds, scoring 0.9953
there), so Figure 2a's
delta is +0.0019: strictly positive, and rendered "+0.00" by the
two-decimal annotation format. Figure 1b does not round and therefore
correctly colors the cell as a GNN win. Ties in Figure 1 are resolved in
the order [Louvain, Leiden, k-means, DMoN, MinCut], i.e., always AGAINST
the GNN family, so a GNN-colored cell is always a strict win; at this cell
the only exact tie involving the winner is DMoN vs MinCut, within the GNN
family (Leiden and k-means are also exactly tied with each other on the
non-GNN side). Six cells on
the near-saturation edge of the band show the same sub-0.005 effect (five
at +0.0019, one at +0.0045), and no cell shows the converse (non-GNN color
with a positive delta). The camera-ready will (i) use three-decimal
annotations in Figure 2, (ii) state the tie rule in the Figure 1 caption,
and (iii) note explicitly that along the s = 0.65 edge the best GNN and
the best non-GNN method are within 0.005 of each other and of perfect
NMI, so these wins, while real, are practically
negligible, which is consistent with the paper's own "when GNNs win, they
win by cents".

**Q2 (relation to the contextual SBM of Shi et al. 2024).** Our generator
is a k = 4, unsupervised, fixed-dimension analogue of the two-class CSBM
(Deshpande et al., NeurIPS 2018), and the correspondence on the graph side
is exact: writing the CSBM in Shi et al.'s homophily convention,
c_in = d(1 + (k-1)*lambda), c_out = d(1 - lambda) (Deshpande et al. scale
the same signal by sqrt(d): c_in = d + lambda*sqrt(d)), the expected
cross-community edge fraction is (1 - lambda)(1 - 1/k), which is precisely
our mu = (1 - s)(1 - 1/k), so our s IS Shi et al.'s homophily parameter
lambda (restricted to the assortative half lambda >= 0). Our f plays the
role of the CSBM feature
SNR mu, with one modeling difference: our four centers are drawn at random in fixed
dimension 32 (mean structure of rank up to k - 1 = 3) rather than the
rank-one spike sqrt(mu/n) y_i u in the proportional high-dimensional
regime, plus a notational clash (our mu is the cross-edge fraction; theirs
is feature SNR). The task differs in the same way: Shi et al. characterize
SEMI-SUPERVISED GCN generalization (double descent in model complexity,
label fraction, self-loop effects) on that (lambda, mu) plane, while we
study unsupervised clustering, whose CSBM-side counterparts are the
detection-threshold results (Deshpande et al. 2018; Lu & Sen, JMLR 2023).
The camera-ready will add this paragraph and the citations, and we thank
the reviewer for the pointer; the heterophilic half (lambda < 0), which
Shi et al. cover and our s in [0, 1] does not, is a natural extension we
will name in the limitations.

If the clarified figures, the exact generator specification, or the CSBM
positioning leave any remaining concern, we would be glad to address it in
the discussion phase.

---

## Camera-ready plan

The commitments above are individually small but numerous, so for
transparency about the 4-page budget: the main changes entering the pages include the
corrected numbers (C1-C4), the generator formulas, the one-sentence
method descriptions, the two new Table 1 columns with the expanded
caption, the three-decimal Figure 2 annotations with the tie rule in the
caption, the scoped conclusions and the quantitative protocol wording,
and a two-to-three-sentence CSBM correspondence with its citations. The
depth and tuning studies will be summarized in one or two sentences each,
with this repository's scripts and CSVs as the complete record.

We again welcome follow-up questions from all reviewers during the
discussion phase.
