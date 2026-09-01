# C63 — Entropy-Preserving Rank Transport (EPRT): formula audit

**Decision: strict NO-GO as a new decoding primitive.**

Let `p` be the target VLM next-token distribution and let `s_v` be an independent
visual-evidence score for vocabulary item `v`.  EPRT keeps the multiset of probabilities
but permutes which token receives which mass:

\[
p'_v=p_{\pi(v)},\qquad
\pi^*\in\arg\max_\pi\sum_vs_vp_{\pi(v)}.
\]

## Exact reduction

By the rearrangement inequality, the optimizer simply sorts `s` and assigns the largest
probability to the largest evidence score, the second largest to the second score, etc.
It is therefore a linear assignment / rank-matching operation.  If permutation is limited
to a shortlist or bounded rank displacement, it is precisely shortlist reranking with a
rank constraint.

Every symmetric functional of the **one-step** probability vector is preserved, including
Shannon and all Renyi entropies:

\[
H_\alpha(p')=H_\alpha(p).
\]

This is true for the trivial reason that entropy is invariant under relabeling outcomes;
it is not a new conservation theorem.  It also does not imply preservation of sequence
entropy: changing the sampled/argmax token changes the next context and hence all future
conditional distributions.

## Why it is not conservative grounding

The top-token identity after full rank transport is determined entirely by the specialist
ranking, while the large VLM contributes only a sorted confidence profile.  Thus it is more
aggressive than ordinary score fusion, not safer.  If the specialist's top concept is wrong,
EPRT assigns it the VLM's largest probability even when the VLM originally gave it negligible
mass.  Conversely, restricting swaps to nearby/top-k tokens turns it back into ordinary
reranking/assignment.

A radiology classifier also does not define an ordered score for every subword, function word,
punctuation mark, and syntactic continuation in the VLM vocabulary.  Extending finding scores
to full-vocabulary token scores requires an ontology-to-token model; that model is the actual
expert/logit interface and reintroduces the fusion/alignment problem EPRT claims to avoid.

## Collision neighborhood

- Rearrangement inequality: optimal same-order pairing of two sorted sequences.
- Linear assignment / optimal transport on a permutation polytope.
- Learning-to-rank and rank aggregation on permutations (Mallows/Luce/Borda families).
- Generation reranking laws: https://proceedings.neurips.cc/paper_files/paper/2024/file/c8b2f897e45770595656a79a9ad91e89-Paper-Conference.pdf
- Differentiable sorting as optimal transport: https://arxiv.org/abs/1905.11885

## Verdict

EPRT has a neat sentence—"change token identity without changing uncertainty"—but its
mathematics is outcome relabeling plus the rearrangement inequality.  Full transport lets the
specialist replace the VLM decision; local transport is top-k reranking.  It should not be used
as the main idea and needs no GPU experiment.
