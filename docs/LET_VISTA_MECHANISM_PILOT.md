# LET–VISTA Mechanism Pilot

## Protocol

- Dataset/model: first 128 RULE/MIMIC-CXR questions, LLaVA-Med-7B.
- Statistical clusters: 25 patients.
- Shared frontend: `vicuna_v1`, RULE binary instruction, 64-token greedy
  complete-sentence generation, and the normalized RULE parser.
- Fixed transport weight: `alpha=0.30`.
- LET layer: `L-12`; VISTA SLA window: released default layers `25--30`.

This experiment isolates VISTA's Self-Logits Augmentation (SLA). It does not
test VISTA's separately calibrated Visual Steering Vector.

## Results

| Method | Acc. | BAcc. | Delta | Rescue/Harm | Words |
|---|---:|---:|---:|---:|---:|
| Greedy | 71.88 | 72.42 | -- | -- | 13.16 |
| LET, normalized single layer | 76.56 | 73.97 | +4.69 | 16/10 | 17.62 |
| LET, unnormalized single layer | 75.78 | 75.30 | +3.91 | 7/2 | 12.83 |
| VISTA released SLA window | 75.78 | 75.30 | +3.91 | 7/2 | 12.76 |
| VISTA window + final RMSNorm | 76.56 | 74.47 | +4.69 | 15/9 | 14.93 |

No pilot comparison is significant: LET versus greedy has exact McNemar
`p=0.327`; LET versus normalized-window VISTA has 3/3 discordance and `p=1`.

## Mechanism Diagnostics

The centered augmented-logit norm relative to the final-logit norm is `0.0265`
for unnormalized `L-12`, `0.0534` for released VISTA SLA, `0.6160` for LET,
and `0.6887` for normalized-window VISTA. Thus the released SLA logits are on
a very different scale for this LLaVA-Med checkpoint. Applying the model's
final RMSNorm restores comparable scale.

Normalization changes the operating point: LET recovers more positives but
also breaks more negatives. It does not establish unique superiority.
Normalized single-layer LET and normalized-window VISTA have identical
accuracy and agree on 95.31% of parsed decisions.

## Claim Boundary

The pilot **does not support** the statement that VISTA-style augmented logits
fail while LET alone works. It instead supports:

1. positive early/final logit mixing is a promising family on this protocol;
2. architecture-aware final normalization materially controls its strength;
3. single-layer versus window aggregation mainly changes the clinical
   operating point in this sample;
4. LET's current evidential advantage is its completed 3,466-question result,
   not operator superiority over VISTA SLA.

A paper comparison against VISTA requires a frozen full run of both SLA and
the complete VSV+SLA method. Until then, VISTA must not be described as a
failed baseline.
