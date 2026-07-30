# Analysis plan — from three template papers to a 7a paper

## The argument (what the paper is *for*)

Avila 2019 is descriptive because it asks "what does 7a do?" A paper that argues asks a question
with two possible answers that matter. The question here:

> **Is area 7a on the perceptual pathway for heading, or is it already on the navigation
> pathway?**

Two concrete, falsifiable outcomes:

- **(i) Perceptual.** 7a carries heading information that is read out for the choice: significant
  choice probability, CP concentrated in the most sensitive neurons, congruent cells improving
  under cue combination and tracking behavioural weights. 7a takes its place in the
  MSTd→VIP→7a ladder.
- **(ii) Navigational.** 7a carries heading information but its trial-to-trial fluctuations are
  *not* read out for this choice — heading-driven signal without choice-driven signal (the
  Zaidel 2017 dissociation, coming out the opposite way from VIP). Combined with Avila's finding
  of *divergent* rather than convergent linear/angular codes, and 7a's retrosplenial /
  hippocampal connectivity, this says 7a's heading signal is formatted for allocentric readout,
  not for a left/right discrimination.

Either answer is publishable, and (ii) is the more interesting one because it breaks the ladder.
The framing must be set up so the result is informative *whichever way it goes* — that is the
difference from Avila 2019.

**Prior expectation from Eric's plots:** low rates, single-neuron thresholds 10–180° vs ~3°
behaviour, and spike–choice correlations mostly |r| < 0.15. That points toward (ii). Which means
the paper's burden is to show the heading signal is genuinely *there* (so the null CP is not just
"7a is unresponsive") — a population decoder that recovers heading well while CP stays at chance
is the strongest version of this.

---

## Figure plan

### Fig 1 — Behaviour (template: Gu 2008 Fig 1–2)
1. Task schematic + heading set (0, ±1.5, ±3, ±6, ±12°).
2. Example session psychometric functions, three modalities, cumulative-Gaussian fits.
3. All sessions: threshold ves / vis / com, paired.
4. **Measured combined threshold vs optimal prediction** σ_pred = √(σ_ves²σ_vis²/(σ_ves²+σ_vis²)).
   Scatter + unity line + Wilcoxon signed-rank.
5. *(If cue-conflict trials exist)* measured vs predicted visual weight.

**Statistics:** per-session MLE fits with lapse rate; bootstrap CIs on thresholds; Wilcoxon
signed-rank for measured vs predicted; report n sessions and n monkeys separately.

### Fig 2 — 7a heading responses during the active task (template: Gu 2007 Fig 3–4)
1. Example unit PSTHs by heading × modality.
2. Heading tuning curves (rate over the middle stimulus window).
3. Fraction of units with significant heading tuning per modality (ANOVA / regression).
4. DDI distributions per modality + pairwise (replicating Eric's `image5` but for HD).
5. Congruency: vestibular vs visual tuning slope scatter → **congruent / opposite / untuned**
   classification, and the congruency index distribution.

### Fig 3 — Single-neuron sensitivity (template: Gu 2007 Fig 5; Gu 2008 Fig 5)
1. Example neurometric functions (antineuron ROC) with psychometric overlaid.
2. Neuronal threshold distributions per modality.
3. Neuronal vs psychophysical threshold ratio.
4. Combined vs single-cue neuronal threshold, **split by congruent vs opposite cells**.
   Gu 2008's key result: congruent cells improve, opposite cells degrade. If Eric's
   "combined is worse" holds, check whether it is carried by opposite cells.

### Fig 4 — Choice-related activity (template: Gu 2007 Fig 6–7)
1. Example CP distributions (preferred vs null choice, z-scored within heading).
2. Grand CP histogram per modality, permutation test vs 0.5.
3. CP vs neuronal threshold (Gu 2007: CP is carried by the sensitive neurons).
4. CP time course.

### Fig 5 — Sensory vs decision components (template: Zaidel 2017, via the Noel review)
1. Partial correlation of rate with choice given heading, and with heading given choice.
2. Position 7a against published MSTd and VIP values.

### Fig 6 — Population decoding (this is the part the template papers did *not* do, and the part
that rescues the story given low single-neuron sensitivity)
1. Linear decoder of heading from simultaneously recorded units (the U-Probe gives 16 channels)
   → population neurometric threshold vs behaviour.
2. Decoder of *choice* from the same population, and the residual after conditioning on heading.
3. Population threshold in combined vs predicted from unimodal populations.

---

## Analysis primitives to implement

| Primitive | Definition | Module |
|---|---|---|
| Cumulative-Gaussian psychometric fit | MLE, params μ (PSE), σ (threshold), optional λ (lapse) | `behavior.py` |
| **Choice-coding resolver** | recover P(rightward) from the stored `choice` field — see `03_data_questions.md` | `behavior.py` |
| Optimal threshold prediction | σ_pred = √(σ₁²σ₂²/(σ₁²+σ₂²)) | `integration.py` |
| Predicted cue weights | w₁ = σ₂²/(σ₁²+σ₂²) | `integration.py` |
| Empirical weights from conflict | from PSE shift across Δ | `integration.py` |
| ROC / AUC | exact, via Mann–Whitney U | `neurometrics.py` |
| Neurometric function | antineuron ROC: p(θ) = AUC(r(θ) vs r(−θ)) | `neurometrics.py` |
| Choice probability | z-score rate within heading, pooled ROC pref vs null choice | `neurometrics.py` |
| Permutation test for CP | shuffle choice labels within heading, ≥2000 iters | `neurometrics.py` |
| DDI | (r_max − r_min)/(r_max − r_min + 2√(SSE/(N−M))) | `neurometrics.py` |
| Congruency index | correlation of ves and vis tuning curves; sign of slope product | `neurometrics.py` |
| Partial correlations | r(rate, choice \| heading), r(rate, heading \| choice) | `neurometrics.py` |

## Output requirements

JP wants **panels in Adobe Illustrator**. So:
- All figures exported as **PDF with `pdf.fonttype = 42`** (TrueType, editable text) and
  **SVG with `svg.fonttype = 'none'`**. No rasterised text, ever.
- One panel per file, named `figXY_panel.pdf`, so panels can be assembled in Illustrator.
- Every panel writes a sidecar `.csv` of the plotted values and a `.json` of the stats, so numbers
  in the manuscript are traceable to a file rather than re-read off a plot.
Handled by `seven_a/style.py`.
