# ASCC-v3 choice-construct fallback (pre-outcome contract)

This fallback was designed without reading ASCC-v2.1 outcomes. It repairs the
semantic ambiguity of a bare `uncertain` continuation and the text-image
conflict induced by an asserted parent finding. No GPU run is authorized until
the two physician admissions below pass.

## Radiograph-evidence options

For each edge, freeze three complete options. Pneumonia example:

- NEG: `Based only on this chest radiograph, the supported conclusion is that pneumonia is absent.`
- UND: `Based only on this chest radiograph, it is indeterminate whether pneumonia is present or absent.`
- POS: `Based only on this chest radiograph, the supported conclusion is that pneumonia is present.`

The model sees all three and returns only `A`, `B`, or `C`. These response codes
carry no clinical semantics. Use the three cyclic Latin mappings:

- L0: A=NEG, B=UND, C=POS
- L1: A=UND, B=POS, C=NEG
- L2: A=POS, B=NEG, C=UND

Within every `(edge, child_vote, parent_vote, aspect)` stratum, hash-sort images
and assign L0/L1/L2 round-robin. Mapping counts may differ by at most one. The
same image keeps the same mapping in all four conditions.

## True 2x2 intervention

Fix speech act to `Assess` and cross:

- noun: `findings` versus `abnormalities`;
- context: `no_parent` versus `fixed_parent`.

Prompt:

`Assess the image-grounded <noun> on this chest X-ray. Do not assume unavailable history, laboratory tests, pathology, or prior images. Choose exactly one option below and reply only with A, B, or C. <mapped options>`

Prefixes:

- no-parent: `Choice:`
- fixed-parent: `Observation: Lung opacity is present.\nChoice:`

`Observation` prevents the context manipulation from reusing either noun.
Child identity remains fixed in the options. The no-parent condition is primary;
fixed-parent is diagnostic-completion replication.

## Estimands

Invert the Latin mapping to `(q_N,q_U,q_P)` and define:

`C = log((q_N + q_P) / q_U)`, `P = log(q_P / q_N)`.

For context `g` and vote bin `s`:

`Delta[g,s] = E[C_abnormalities - C_findings | g,s]`.

- negative local: `lambda-[g] = Delta[g,1] - Delta[g,0]`;
- positive local: `lambda+[g] = Delta[g,2] - Delta[g,3]`;
- DID: `theta[g] = 0.5 * (lambda-[g] + lambda+[g])`.

Primary is `theta[no_parent]`. The parent amplification
`theta[fixed_parent]-theta[no_parent]` is descriptive. Bootstrap images within
fixed `(parent_vote, aspect, mapping)` overlap strata.

Panel-state proxy targets remain explicitly constructed:
0/3→NEG, 1/3 and 2/3→UND, 3/3→POS. Report NLL and Brier; never call them patient
truth.

## Pre-GPU admissions

1. At least three chest radiologists review the complete sentences. Each state
   needs >=90% polarity agreement; UND needs >=80% unique radiograph-only
   indeterminacy interpretation; median naturalness >=4/5; kappa >=0.6.
2. On at least 120 independent images (30 per vote bin), three to five chest
   radiologists provide image-only NEG/UND/POS plus reason. Both
   `Pr(UND|1)>Pr(UND|0)` and `Pr(UND|2)>Pr(UND|3)` need image-bootstrap 95% CIs
   above zero, and positive support must rise ordinally. Missing clinical data
   cannot be the dominant UND reason.

Failure of either admission kills the wording or edge before GPU use.

## Computational gates

- Latin balance and single contextual `A/B/C` tokens;
- >=90% full-vocabulary restricted top-1 in every
  `(noun, context, mapping, vote)` cell;
- neutral findings/no-parent admission at both local boundaries;
- both no-parent noun locals positive; `theta_no_parent >= log(1.5)` with CI>0;
- at least two Latin subgroups same-direction and no qualitative reversal;
- absolute ambiguous-bin polarity shift CI90 within +/-0.2 for both contexts;
- gauge-invariant clear-bin cross-fit calibration and >=99% computable nested
  bootstraps;
- fixed-parent local directions agree with no-parent;
- text-only and cross-support image swaps cannot reproduce the own-image reader
  interaction.

## Claim ceiling and budget

Even after all gates, the maximum claim is a fixed-candidate,
option-randomized radiograph-evidence framing interaction. It is not a patient
truth hallucination, latent epistemic state, natural OE effect, or mitigation.

The 509-image primary census requires four forwards per image: 2,036 jobs,
matching v2. A 12-image engineering canary costs 48 forwards. On the current
RTX 4090, expected full Huatuo scoring is roughly 8–12 minutes. Second models,
replication edges, and hidden-state interventions remain gated.

## Minimal patch map

- `prepare_ascc_factorial_v3.py`: options, Latin assignment, 2x2 conditions;
- `run_huatuo_ascc_choice_v3.py`: final-logit `A/B/C` scoring only;
- `analyze_ascc_choice_v3.py`: inverse mapping, stratified nested analysis;
- blinded semantic/reference admission compiler;
- synthetic tests for mapping balance/inversion, prompt identity, estimands,
  bootstrap, and fail-closed gates.
