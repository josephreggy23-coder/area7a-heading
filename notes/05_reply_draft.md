# Follow-up email to Jean-Paul

Thread: "Visuo-vestibular single neuron data in monkeys" (noelx071@umn.edu, 2026-07-29).
Joseph already replied 2026-07-30: *"Thanks so much for the opportunity. I will get this started
right away."* — so this is a progress update, not a first reply.

**Dropbox status:** the invite to the folder **"VisVest Eric"** was received (twice, 2026-07-29,
from jeanpaulnc@gmail.com) and the account is active. Do **not** ask JP to re-send it. The actual
obstacle is that the folder exceeds the 2 GB Dropbox Basic quota, so "Add to Dropbox" fails —
files have to be downloaded individually from the web view instead. The draft mentions this as a
logistical aside rather than a blocker, and asks for the HD file directly as a convenience.

**Superseded:** the cue-conflict question has been dropped from the email. Gu 2008 estimated
neuronal weights from tuning curves by least squares with no conflict manipulation, so its absence
is no longer a hard blocker — see `03_analysis_plan.md`.

*Draft below — a Gmail draft has been created but nothing has been sent.*

---

**Subject:** Re: Visuo-vestibular single neuron data in monkeys

Hi Jean-Paul,

Quick progress update. I've read the three PDFs and gone through Eric's 2018 plots carefully, and
I've built out most of the analysis pipeline so it's ready when I pull the data down from the
Dropbox folder — I'll follow the README for the MATLAB load, then export to Python from there.

**On the framing.** I take your point about Avila 2019 being descriptive. Reading it next to your
review, the question that seems to make this data argue rather than describe is: *is 7a on the
perceptual pathway for heading, or already on the navigation pathway?* Framed that way both
outcomes say something. If 7a shows significant choice probability concentrated in its most
sensitive neurons, with congruent cells improving under cue combination, it slots into the
MSTd→VIP ladder. If instead it carries heading information whose trial-to-trial fluctuations
aren't read out for the choice — the Zaidel 2017 dissociation coming out opposite to VIP — that's
the more interesting result, because it breaks the ladder, and it fits with Avila's finding of
divergent rather than convergent linear/angular codes plus the retrosplenial connectivity.

Eric's plots make me suspect the second. Firing rates in the HD task look like 1–5 spk/s, and his
neuronal thresholds run 10–180° against a ~3° psychophysical threshold, so single 7a neurons are
one to two orders of magnitude less sensitive than the monkey. That means the argument probably
has to be a population one rather than a single-neuron one — and if the burden is to show the
heading signal is genuinely there so a null CP doesn't just read as "7a is unresponsive," then a
population decoder that recovers heading well while CP stays at chance is the strongest version of
it. The 16-channel array recordings should support that, as long as I can tell which units were
recorded simultaneously.

**On the choice coding.** Your recollection looks right, and I think there's a clean reason for
it. A bell shape where a sigmoid belongs is the signature of a variable coded relative to the
stimulus rather than in absolute space — something accuracy-like. If the stored value means
"correct," then plotting P(that value) against heading gives a U; if it means "error," you get the
inverted-U you remember. Either way absolute direction comes back by flipping the mapping on one
side of zero, exactly as you described. I've written it so all four candidate mappings get fit and
the one that actually produces a monotonic sigmoid is selected by likelihood, with the
diagnostics reported, so we can see why rather than taking it on faith.

One consequence worth flagging now: if the coding really is stimulus-relative, then **zero-heading
trials can't be decoded at all** — there's no correct answer at 0°, so the direction information
isn't recoverable from that field. That's awkward, because 0° is exactly where choice probability
is most informative, since there's no stimulus drive to remove. Do you know if there's a separate
field anywhere recording the rewarded side or the target the animal actually chose? If so it would
rescue what are probably the most valuable trials in the dataset.

**Two questions that gate whole analyses:**

1. Are there cue-conflict (Δ) trials? Gu 2008 used Δ = ±4° to measure the weight the animal placed
   on each cue. Without them we can still test the threshold prediction, but not the weight
   prediction, which is half of that paper's argument.
2. Is visual coherence varied across trials? The Fetsch 2011 result *is* the reliability
   manipulation, so if everything is at one coherence that paper isn't reproducible here and the
   story rests on the other two. I'd rather know before building toward it.

**Two things in Eric's plots I want to flag.** In his example HD session the psychometric
thresholds are ves 4.05°, vis 3.50°, combined 2.65° — and the optimal prediction from the two
single cues is also 2.65°. Exact to three digits, which is either a very clean result or a number
that got back-computed somewhere. Worth checking across all sessions. Separately, the *neurometric*
thresholds in that same session go the other way: ves 13.65, vis 20.37, combined 35.59. Combined
being worse than either single cue is the opposite of the behaviour and the opposite of Gu 2008's
congruent cells. Could be a real effect carried by opposite cells, could be an artefact of the low
rates — I can't tell from the figure alone, so it's the first thing I'll check.

**Where the code is.** I've got psychometric fitting, the choice-coding resolver, ROC and
antineuron neurometric functions, choice probability with permutation tests, DDI and congruency
classification, partial correlations, and the cue-integration predictions. It runs end to end and
recovers known values on simulated sessions, which is as far as I can validate it until I have the
real thing. Panels export as vector PDF/SVG with live text so they open cleanly in Illustrator,
with a CSV of the plotted values and a JSON of the stats next to each one, so every number in a
figure is traceable. Happy to share the repo if that's useful.

No rush on any of this — I'll keep going on the parts that don't depend on the answers.

Best,
Joseph
