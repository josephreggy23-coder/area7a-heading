# area7a-heading

Visuo-vestibular heading discrimination in macaque posterior parietal **area 7a**.

Noel Lab, University of Minnesota. Data collected by Eric Avila (Angelaki lab); analysis by
Joseph Regito David.

## The question

Area 7a responds to both optic flow and vestibular self-motion (Avila et al. 2019,
*Cereb Cortex* 29:3932), but as Noel & Angelaki (2022, *Annu Rev Psychol* 73:103) put it, the
correlation between neural activity and heading judgments has never been reported there.

This project asks whether **7a is on the perceptual pathway for heading, or already on the
navigation pathway** — reproducing the analyses of three papers in a new area:

- Gu, DeAngelis & Angelaki 2007, [10.1038/nn1935](https://doi.org/10.1038/nn1935)
- Gu, Angelaki & DeAngelis 2008, [10.1038/nn.2191](https://doi.org/10.1038/nn.2191)
- Fetsch, Pouget, DeAngelis & Angelaki 2011, [10.1038/nn.2983](https://doi.org/10.1038/nn.2983)

Both outcomes are informative — see [`notes/03_analysis_plan.md`](notes/03_analysis_plan.md).

## Layout

```
analysis/seven_a/     the toolkit
  matio.py            .mat loading (v7 and v7.3) + a describe() tree printer
  behavior.py         psychometric fitting; recovery of the choice sign convention
  neurometrics.py     ROC, neurometric functions, choice probability, DDI, congruency
  integration.py      optimal cue-combination predictions and measured weights
  synth.py            ground-truth simulator
  style.py            Illustrator-safe figure export
analysis/scripts/     runnable analyses
analysis/tests/       ground-truth tests
notes/                literature notes, plot inventory, analysis plan, open questions
```

Data and source PDFs are deliberately untracked — see `.gitignore`.

## Getting started

```bash
python -m pytest analysis/tests -q
```

```bash
python analysis/scripts/demo_synthetic.py --outdir figures/demo
```

The demo runs the full pipeline on simulated sessions and writes vector panels with CSV/JSON
sidecars. When the recordings arrive, only the loader changes.

## The choice-coding gotcha

The stored `choice` field does not straightforwardly mean left or right — plotting it directly
gives a bell curve where a sigmoid belongs. That is the signature of a variable coded *relative
to the stimulus* rather than in absolute space. `behavior.resolve_choice_coding()` fits all four
candidate mappings, selects by likelihood, and reports monotonicity and symmetry diagnostics so
the choice is auditable.

One consequence: under stimulus-relative coding, zero-heading trials carry no recoverable
direction — and those are exactly the trials where choice probability is most informative. See
[`notes/04_data_questions.md`](notes/04_data_questions.md).
