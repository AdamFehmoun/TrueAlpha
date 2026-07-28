# BACKLOG — TrueAlpha

Engineering debt ledger. Opened 2026-07-27, at the close of M2 (foundation) and
before the `v1.0-foundation` tag.

**This file exists because of Rule 8 of `POSTMORTEM.md`**, which amends the
backlog practice inherited from TrueSight with one clause that TrueSight never
had:

> tout item 🔴 porte une date d'échéance ; un 🔴 échu bloque les features
> exactement comme une CI rouge.

So the ledger has teeth:

- **🔴 = blocking, with a due date.** One overdue 🔴 and no feature commit may
  land until it is closed or its date is *renegotiated in writing in this file*
  (moving a date is allowed; moving it silently is not).
- **🟠 = due, not blocking.** Carries a date; slipping it is logged, not fatal.
- **🟢 = declared and accepted.** No date. A 🟢 is a decision, not a debt; it
  must state why it is acceptable, and what would turn it back into a 🔴.

Every item names the rule it serves, the evidence, and who opened it. Rule 10's
closing line is the reason this file is not decoration:

> Le dernier 1 % — publier, taguer, annoncer, fermer — n'est pas du polish :
> c'est la seule partie qui transforme du code en actif.

---

## 🔴 Blocking

### B1 — Permuted-label leakage test, written before the model
**Rule 2** (« Le test anti-leakage s'écrit AVANT le modèle »).
**Due: 2026-08-05 (révisé le 28/07, voir « Ré-datation » ; date initiale
2026-08-03). Hard gate: no ML commit may land before this test is green.**

The split machinery is already guarded (`assert_no_leakage`, purge/embargo,
per-fold and segment-level calendar + contiguity checks, all mutation-tested).
What does not exist is the control that catches leakage *through the features*:
permute the labels, refit, and assert the out-of-sample Sharpe collapses to
within noise of zero. Without it, an ML signal that scores well is
indistinguishable from one that has been shown the answer.

Rule 2 is explicit that this is written *before* the model, not after a good
result appears. The GAF-CNN notebook was invalidated by three leaks that were
diagnosed in prose and never executed as code; this item exists so that cannot
recur.

*Opened by the jury, audit n°7 (2026-07-27).*

### B2 — Property-based tests on the engine invariants
**Rule 5** (« Property-based tests sur les invariants … comme dans glicko2-ts —
c'est ton meilleur pattern, applique-le au cœur »).
**Due: 2026-08-04 (révisé le 28/07, voir « Ré-datation » ; date initiale
2026-08-02).**

`hypothesis` appears in neither `requirements.txt` nor any test file. The suite
is 160 example-based tests plus a 12-mutation audit — strong, but both methods
only exercise cases someone thought of. Rule 5 names three properties by hand:

1. PnL is exactly zero when no trade is taken;
2. cost is strictly performance-decreasing — for the same signal, raising
   `cost_rate` never raises net return, and raises it strictly whenever
   turnover is non-zero;
3. symmetries — sign conventions and the flat/long boundary.

Mutation testing proves the existing tests bite. Property testing generates the
inputs nobody wrote down. They are complementary and only the first is in place.

*Opened by the jury, audit n°7 (2026-07-27).*

### B3 — AR(1) baseline in the published table
**Rule 3** (« Aucune métrique de modèle n'est présentable sans les baselines
naïves dans le même tableau : classe majoritaire, buy-and-hold, momentum simple,
AR(1) »).
**Due: 2026-08-07 (révisé le 28/07, voir « Ré-datation » ; date initiale
2026-08-05). Hard gate: no model metric is published without it in the same
table.**

The repo publishes two of the four named baselines: buy & hold
(`strategies/buy_and_hold.py`) and MA 20/50 crossover
(`strategies/ma_crossover.py`, the simple-momentum family). "Majority class" has
no meaning in returns space; its equivalent — always-long — *is* buy & hold, so
that one is satisfied by mapping and this substitution is declared here rather
than assumed. **AR(1) is genuinely absent** (grep: zero occurrences) and is the
one that matters most against an ML signal, because a one-lag autoregression is
the cheapest thing that can explain away a claimed edge.

*Opened by the jury, audit n°7 (2026-07-27).*

### B10 — Exit-fee re-billing is one bar off in timing (trigger fired, was 🟢)
**Moved to blocking 2026-07-28: its own written trigger fired. Due: 2026-08-12
(with B5).**

`exit_fee_bias_bps` re-bills the uncharged exit at the **last** bar of its
segment; a continuous book would bill it at the **first** bar of the next
segment. Exact in amount, one bar off in timing. The written trigger — "turns 🔴
the moment `n_segments > 1`" — fired at the B4-B run: BTC rolling has
`n_segments = 2`. Mitigating fact, stated precisely: the observed segment
boundary (fold 1→2, the 5/50 → 5/100 parameter change) carries an outgoing
position of **0.0**, so there is no unbilled boundary exit and the timing
question has zero monetary effect on the published figures today — the only
uncharged exit remains the terminal one. The item is blocking because the next
run with a non-zero outgoing position at a parameter change will make the
one-bar timing approximation a real number, and the treatment must be decided
before that happens, not after.

---

## 🟠 Due, not blocking

### B5 — Re-selection must actually bite
**Rule 6 / walk-forward validity.** **Due: 2026-08-12 (révisé le 28/07, voir
« Ré-datation » ; date initiale 2026-08-10).**

`results/metrics.json` reports `n_segments: 1` on all four published runs: the
parameter grid selects the same point on every fold, so the 20 folds splice into
a single backtest and the walk-forward protocol never re-selects anything. It is
correct machinery producing a degenerate outcome — which is documented, but
means the protocol is currently *decorative*. Either the grid must be wide
enough that selection changes across folds, or the degeneracy must be stated as
a finding rather than carried as a feature. Ties to B10.

**Addendum 2026-07-28 (B4-B run):** the premise fell. On the 3,240-bar span the
selection moved — three distinct parameter sets across the four runs (5/50,
5/100, 10/20), and BTC rolling re-selects MID-RUN (5/50 on fold 1, 5/100 after:
`n_segments = 2`, the project's first real parameter change). Re-selection now
demonstrably bites. What remains of this item at its due date is the narrower
question of whether the 13-combo grid gives selection enough room to be
meaningful, or whether it should be widened deliberately.

### B8 — Monthly README prose review
**Rule 7** (« Revue mensuelle obligatoire : chaque phrase du README est soit
vérifiée contre le code, soit supprimée »). **First: 2026-08-01, then monthly.**

The *numeric* half of Rule 7 is mechanically enforced:
`python -m scripts.generate_results --check` fails CI when a published figure
diverges from `results/metrics.json`. The *prose* half has no enforcement and no
schedule — and eight projects out of eight failed on prose, not on figures.

### B14 — The README mutation table is hand-written and tied to nothing
**Rule 7.** **Due: 2026-07-31.**

`generate_results.py` states "numbers are never written by hand", yet the
mutation table in the README is transcribed by hand from the audit output, and
no check ties the two together: the kill counts can drift silently the day the
suite grows (it has grown at every commit). Proposal — to write, deliberately
not implemented in the commit that opens this item:
`python -m scripts.mutation_audit --check` re-runs the audit, renders its own
table, compares it to the README's mutation block, and exits 1 on any drift —
the same contract `generate_results --check` already enforces for the results.

*Opened 2026-07-28 (B4-A block).*

---

## 🟢 Declared and accepted

### B9 — The mutation audit is not CI-enforced
It reruns the whole suite once per mutation (12 × ~11 s today) and is invoked by
hand: `python -m scripts.mutation_audit`. Accepted because the audit is a
periodic instrument, not a per-commit gate, and because it exits 1 on any
deviation so it cannot pass silently when run.

**Turns 🔴 if** the suite gets fast enough to run it per-commit, or if a
mutation table entry is ever found stale in an audit.

### B11 — M13 is a guard with no reachable caller
The segment-level positional-contiguity assertion in `engine/evaluate.py` cannot
fire on any untampered input: the splice rule appends fold *k* only when
`splits[k].test[0] == splits[k-1].test[-1] + 1`. By the letter of Rule 6 ("pas
de module sans appelant") it is unreachable code. Accepted because a defensive
invariant assertion is not a feature, and because it is now declared in
`EXPECTED_SURVIVORS` with its reasoning, which converts a dead branch into a
tripwire on the splice rule itself.

**Turns 🔴 if** any test starts killing M13 — that means the splice rule changed
and the invariant no longer holds.

### B12 — The 1h series are hash-pinned and unused
`manifest.json` pins BTC/USDT and ETH/USDT at 1h (77 628 rows each on the B4
span, 132 shared exchange-maintenance holes pinned by timestamp — figures
refreshed 2026-07-28 at the B4-B download; previously 26 303 rows / 1 hole on
the 2022-2024 span) and no published number uses them, which the README states
explicitly. Accepted because data is not code and the non-use is declared at
the point of presentation — the opposite of Fahm.io's pgvector, where the UI
button existed and the pipeline was never called.

**Turns 🔴 if** any 1h figure is published, or if the README stops saying they
are unused.

### B15 — Family-wise error: the tested-hypotheses counter stands at 2
The B4 span extension is the SECOND test of the same hypothesis — "the MA
crossover family has an edge" — on overlapping data: 2022-2024 published, and
2017-2026 pre-registered at B4-A then EXECUTED at B4-B (2026-07-28). Every
additional test of the family inflates the probability that one of them clears
a significance bar by luck; at this scale the honest mitigation is to count and
say it. **Counter: 2 — both executed.** Counting rule, written down 2026-07-28:
the counter increments when a NEW test of the family is *pre-registered*, not
when it is executed — executing pre-registered test n°2 does not create a test
n°3, and incrementing at execution would double-count every future
pre-registered run. Cite the counter next to any significance claim.

**Turns 🔴 if** a significance claim is ever published without the family
count, or if the counter passes 5 without a formal multiple-comparisons
correction (deflated Sharpe ratio / Bonferroni) entering the protocol.

*Opened 2026-07-28 (B4-A block).*

---

### 2026-07-28 — Ré-datation du backlog pour absence déclarée (02→08/08)
Adam est en voyage du 2 au 8 août, capacité : un bloc pré-spécifié par soirée.
Les échéances tombant dans cette fenêtre sont révisées AVANT de tomber :
B2 02/08 → 04/08 · B1 03/08 → 05/08 · B3 05/08 → 07/08 · B5 10/08 → 12/08.
B4 (empan) est AVANCÉ au 30/07 : il déplace tous les nombres publiés et exige
un audit à froid, donc il se fait à portée de bureau, pas en voyage.
B6 et B13 sont clos le 28/07, en avance.
Motif : une date révisée à l'avance est une décision, une date manquée est une
dérive. Règle 10.

---

## Closed

### B13 — The mutation audit could publish counts from a dirty base and die without restoring
**Rule 5 / Rule 7.** **Opened and closed 2026-07-28 — the trace is worth the line.**

The incident (provoked by the jury, logged): `scripts/mutation_audit.py` was
launched with a 120 s timeout on a ~3 minute job. The SIGTERM killed it
mid-flight LEAVING `engine/evaluate.py` MUTATED on disk; the next pytest
produced 2 phantom failures, and the re-audit published FIVE inflated kill
counts from the dirty base before catching itself on the first damaged anchor.
Two defects, both in the instrument, none in the engine: (1) it started without
verifying its base was clean; (2) it did not guarantee restoration when dying.

The fix (D2): `assert_pristine_base` derives the target set from `MUTATIONS`,
verifies each file is byte-identical to HEAD and refuses to print a single
count otherwise; the whole mutation loop is wrapped in `try/finally` with
SIGTERM/SIGINT handlers that restore the originals then re-raise; and the audit
re-verifies every target's sha256 at the end — success or failure — printing
`sources restored: OK` or failing loudly. Two targeted tests pin both defects
(`tests/test_mutation_audit.py`); the restoration-despite-exception test is the
one that would have caught the incident.

### D2 gate incident — exit codes swallowed by a pipe (operator error, logged)
**Rule 4 / Rule 7. Occurred and closed 2026-07-28 — logged with the same
prominence as the jury-provoked B13, because an asymmetry between memorialized
jury errors and buried operator errors is exactly the drift Rule 5 exists to
stop.**

The local gate that guarded the D2 commit piped `ruff format --check` and
`mypy` through `tail`, which swallowed their non-zero exit codes: D2 (`9f8c0ea`)
was pushed with one unformatted file and two strict-typing errors, and its CI
run **30322209543 failed**. The failure class is B13's — an instrument lying by
truncation — caught in the very commit that was hardening the instrument. Fixed
forward (no history rewrite) by `b9bdae8`: tests patch `subprocess.run` at its
source module, the file is formatted, the gate runs under `set -o pipefail`
with a reported exit per step, and its CI run **30322287611 succeeded**. Every
gate script since carries the same discipline.

### B4 — Buy the calendar span, not the sampling frequency (closed by the B4-B run)
**Rule 1 / statistical power. Closed 2026-07-28, two days ahead of its advanced
due date (2026-07-30).**

`SE(Sharpe_ann) ≈ 1/√(years)` depends on the calendar span alone. The archived
219-bar OOS was 0.600 yr — smallest detectable Sharpe at |t| > 2: **2.58**, a
bar no honest strategy clears. The entry's original justification ("~7.4 yr
brings that floor to 1.63") predated the frozen END and was one revision behind
its own pre-registration; corrected here per audit reservation R6: the probed
span is 2017-08-17 → 2026-06-30 = **8.87 yr**, its 648-bar OOS = **1.774 yr**,
and the floor becomes **1.50** — the pre-registration's number, now the
published run's number. Going to 1h bars over the same span does *not* help,
and the 1h series in `manifest.json` must not be mistaken for added power.

| # | Item | Rule | Closed | Commit |
|---|---|---|---|---|
| — | Boundary calendar holes internal to a spliced window unchecked | 2 | 2026-07-27 | `4a044ac` (M12) |
| — | Exit-fee bias asserted in prose but not computed | 3, 7 | 2026-07-27 | `4a044ac` |
| — | Segment-level contiguity guard covered by no mutation | 5 | 2026-07-27 | `1b86c76` (M13) |
| — | `POSTMORTEM.md` governing the repo from outside the repo | 7, 8 | 2026-07-27 | commit 10 (`f59d284`) |
| — | No debt ledger with due dates (Rule 8's amended clause had nothing to operate on) | 8 | 2026-07-27 | commit 10 (`f59d284`) — this file |
| B6 | GitHub Actions Node 20 deprecation | 4 | 2026-07-28 | `0d567c0` (D1) — annotation list verified empty on run 30321961542 |
| B13 | Mutation audit: dirty base accepted, no restoration guarantee on death | 5, 7 | 2026-07-28 | `b9bdae8` (D2 + its gate fix; `9f8c0ea` alone shipped red — see the D2 gate incident above) |
| B7 | Rule 1's "hors git" clause: amended in writing (50 MB threshold), not silently deviated from | 1, 7 | 2026-07-27 | commit 10 (`f59d284`) — AMENDEMENTS section of POSTMORTEM.md |
| — | D2 gate incident: pipe swallowed format/mypy failures, D2 pushed red | 4, 7 | 2026-07-28 | `b9bdae8` — runs 30322209543 (failure) → 30322287611 (success) |
| B4 | Span extension 2017-08-17 → 2026-06-30: power floor 2.58 → 1.50, four predictions confronted | 1 | 2026-07-28 | B4-B — this commit |
