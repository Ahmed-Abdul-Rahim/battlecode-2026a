# Champion vectors

Every champion is a **16-dim int8 vector** in three blocks (`crates/bc-brain/src/champvec.rs`, mirrored in `ml/champions/descriptor.py`; a parity fixture keeps them identical):

| Dims | Block | Starts as |
| --- | --- | --- |
| 0–5 | stats: durability, basic damage, spell power, reach, mobility, utility | projection of the 24-dim kit descriptor |
| 6–11 | strengths / weaknesses | 0, learned |
| 12–15 | synergy keys | 0, learned |

The kit descriptor is built from champion metadata: base stats and growths, attack type and range, ability kinds and modifiers, passive class. That initial projection is the model's own representation of each champion.

**Where they're used:**
- The draft scorer adds learned synergy (eᵢᵀ S eⱼ for ally pairs) and counter (eᵢᵀ C eⱼ against revealed enemy picks) terms, in the style of DraftRec (P-040). These are skipped while S and C are zero.
- The draft writes this match's vectors into θ_match (slots 0–4 allies, 5–9 enemies).
- The router and heads see "my vector" and "nearest enemy's vector" as features (`champion_vectors` group).

## Fine-tuning from self-play

`python -m ml.champions.finetune runs/` reads match statistics: drafts and outcomes. It fits a logistic model:

```
strength(T) = Σᵢ w·eᵢ + Σ_{i<j} eᵢᵀ S eⱼ
P(A wins)   = σ(strength(A) − strength(B) + Σ_{i∈A,k∈B} (eᵢᵀ C eₖ − eₖᵀ C eᵢ))
```

- It uses analytic gradients in plain numpy, so it runs on any laptop.
- The stats block is L2-regularised towards the kit projection so it keeps its meaning.
- The trait and synergy blocks absorb what self-play shows: strengths, weaknesses and who pairs well. This follows character embeddings learned from behaviour (P-041, P-042).
- Output: `ml/champions/champ_vec_v1.json`, with vectors and quantised S/C.
- Tests check the gradients numerically and recover a planted Varg+Willow synergy.

For varied drafts, self-play needs the draft scorer to explore. Adding a draft-temperature knob is a backlog item, since current drafts are deterministic per map.

## Finding from the first report

`python -m ml.champions.report` (or the MCP `champion_report` tool) renders cosine similarity. With only the 6-dim stats block, **kit-twins are indistinguishable**: Cassi≈Prism, Vendetta≈Issac and Ryth≈Marcella all have cosine 1.0. That's expected, since the other blocks start at zero, but it means the learned blocks are what tell them apart. Until fine-tuning has run, don't rely on vectors to choose between twins.

## Changelog

| Version | Date | Change |
| --- | --- | --- |
| v0 | 2026-10-18 | Kit projection only (builtin kit table). |
