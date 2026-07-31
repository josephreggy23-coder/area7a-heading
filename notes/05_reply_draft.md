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

*Text below mirrors the Gmail draft. Nothing has been sent.*

---

**Subject:** Re: Visuo-vestibular single neuron data in monkeys

Hi Jean-Paul,

Quick progress update. I've read the three PDFs, gone through Eric's 2018 plots carefully, and
built out the analysis pipeline so it's ready to run the moment I can get the data down from the
Dropbox folder.

One logistical thing: the shared folder is larger than my Dropbox Basic quota, so "Add to Dropbox"
won't take it. I'm working around it by downloading files individually from the web view rather
than adding the folder, but if it's easy on your end, pointing me at just the HD file (or dropping
it somewhere I can pull directly) would save some time. Not urgent — I have plenty to do either
way.

**On the framing.** I take your point about Avila 2019 being descriptive. Reading it next to your
review, the question that seems to make this data argue rather than describe is: *is 7a on the
perceptual pathway for heading, or already on the navigation pathway?* Framed that way both
outcomes say something. If 7a shows significant choice probability concentrated in its most
sensitive neurons, with congruent cells improving under cue combination, it slots into the
MSTd/VIP ladder. If instead it carries heading information whose trial-to-trial fluctuations
aren't read out for the choice — the Zaidel 2017 dissociation coming out opposite to VIP — that's
the more interesting result, because it breaks the ladder, and it fits with Avila's divergent
rather than convergent linear/angular codes plus the retrosplenial connectivity.

Eric's plots make me suspect the second. Firing rates in the HD task look like 1–5 spk/s, and his
neuronal thresholds run 10–180° against a ~3° psychophysical threshold, so single 7a neurons are
one to two orders of magnitude less sensitive than the monkey. That means the argument probably
has to be a population one rather than a single-neuron one. If the burden is to show the heading
signal is genuinely there, so that a null CP doesn't just read as "7a is unresponsive," then a
population decoder that recovers heading well while CP stays at chance is the strongest version of
it. I've written that decoder — it turns simultaneously recorded units into a neurometric
threshold directly comparable to behaviour, and residualises against heading before decoding
choice so an above-chance result can't just be the decoder reading the stimulus. Whether it works
depends on being able to tell which units were recorded simultaneously, which I'll check first
thing.

**On the choice coding.** Your recollection looks right, and I think there's a clean reason for
it. A bell shape where a sigmoid belongs is the signature of a variable coded relative to the
stimulus rather than in absolute space — something accuracy-like. If the stored value means
"correct," then plotting P(that value) against heading gives a U; if it means "error," you get the
inverted-U you remember. Either way absolute direction comes back by flipping the mapping on one
side of zero, exactly as you described. I've written it so all four candidate mappings get fit and
the one that actually produces a monotonic sigmoid is selected by likelihood, with the diagnostics
printed, so we can see why rather than taking it on faith.

One consequence worth flagging now: if the coding really is stimulus-relative, then **zero-heading
trials can't be decoded at all** — there's no correct answer at 0°, so the direction information
isn't recoverable from that field. That's awkward, because 0° is exactly where choice probability
is most informative, since there's no stimulus drive to remove. Do you know if there's a separate
field anywhere recording the rewarded side or the target the animal actually chose? If so it would
rescue what are probably the most valuable trials in the dataset.

**One question that gates a whole analysis:** is visual coherence varied across trials? The
Fetsch 2011 result *is* the reliability manipulation, so if everything is at a single coherence
that paper isn't reproducible here and the story rests on the other two. I'd rather know before
building toward it.

I also went back to the Gu 2007 and Gu 2008 methods in PMC to make sure I was implementing what
they actually did rather than something plausible. Most of it matched, but two things didn't, and
both are worth knowing. First, they compute firing rates over the middle 1 s of the stimulus,
which matters here because 7a time courses are diverse enough that a window including onset
transients measures something different. Second, the congruency index is the product of the two
rate-versus-heading correlations, not the correlation between the two tuning curves. Both are
fixed now. Usefully, it also turns out Gu 2008 estimated the neuronal cue weights from tuning
curves by least squares, with no cue-conflict manipulation — so even if this dataset has no
conflict trials, we'd lose the behavioural weight comparison but keep the neuronal one, which is
the half that speaks to how 7a itself combines cues.

**Two things in Eric's plots I want to flag.** In his example HD session the psychometric
thresholds are ves 4.05°, vis 3.50°, combined 2.65° — and the optimal prediction from the two
single cues is also 2.65°. Exact to three digits, which is either a very clean result or a number
that got back-computed somewhere. Worth checking across all sessions. Separately, the *neurometric*
thresholds in that same session go the other way: ves 13.65, vis 20.37, combined 35.59. Combined
being worse than either single cue is the opposite of the behaviour and the opposite of Gu 2008's
congruent cells. Could be a real effect carried by opposite cells, could be an artefact of the low
rates — I can't tell from the figure alone, so it's the first thing I'll check.

**Where the code is.** Psychometric fitting, the choice-coding resolver, ROC and antineuron
neurometric functions, choice probability with permutation tests, DDI and congruency, partial
correlations, cue-integration predictions, and the population decoders. It runs end to end and
recovers known values on simulated sessions, which is as far as I can validate it until I have the
real thing. There's also a script that runs first against the real file and reports the structure,
what the choice field actually contains, and whether the coherence and conflict fields exist.
Panels export as vector PDF/SVG with live text so they open cleanly in Illustrator, with a CSV of
the plotted values and a JSON of the stats next to each one, so every number in a figure is
traceable. Happy to share the repo if that's useful.

Best,
Joseph
