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
**Due: 2026-08-03. Hard gate: no ML commit may land before this test is green.**

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
**Due: 2026-08-02.**

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
**Due: 2026-08-05. Hard gate: no model metric is published without it in the
same table.**

The repo publishes two of the four named baselines: buy & hold
(`strategies/buy_and_hold.py`) and MA 20/50 crossover
(`strategies/ma_crossover.py`, the simple-momentum family). "Majority class" has
no meaning in returns space; its equivalent — always-long — *is* buy & hold, so
that one is satisfied by mapping and this substitution is declared here rather
than assumed. **AR(1) is genuinely absent** (grep: zero occurrences) and is the
one that matters most against an ML signal, because a one-lag autoregression is
the cheapest thing that can explain away a claimed edge.

*Opened by the jury, audit n°7 (2026-07-27).*

---

## 🟠 Due, not blocking

### B4 — Buy calendar span, not sampling frequency
**Rule 1 / statistical power.** **Due: 2026-08-10.**

`SE(Sharpe_ann) ≈ 1/√(years)` depends on the calendar span alone, not on the
number of bars. The 219 out-of-sample daily bars are 0.600 yr, so the smallest
Sharpe detectable at |t| > 2 is **2.58** — a threshold no honest strategy will
clear. Binance BTC/USDT history reaches back to 2017-08; ~7.4 yr brings that
floor to **1.63**. Going to 1h bars over the same span does *not* help, and the
1h series already in `manifest.json` must not be mistaken for added power.

### B5 — Re-selection must actually bite
**Rule 6 / walk-forward validity.** **Due: 2026-08-10.**

`results/metrics.json` reports `n_segments: 1` on all four published runs: the
parameter grid selects the same point on every fold, so the 20 folds splice into
a single backtest and the walk-forward protocol never re-selects anything. It is
correct machinery producing a degenerate outcome — which is documented, but
means the protocol is currently *decorative*. Either the grid must be wide
enough that selection changes across folds, or the degeneracy must be stated as
a finding rather than carried as a feature. Ties to B10.

### B7 — Rule 1's "hors git" clause: amend it or comply with it
**Rule 1 / Rule 7.** **Due: 2026-07-29 (commit 10).**

Rule 1 says the data is stored "en parquet avec un manifest … **hors git ou en
DVC**". The four parquet files *are* committed (2.44 MB; `.git` pack = 1.99 MB).
The rule's intent — from pattern A.8, the 77 MB dataset committed to TrueSight —
is not violated at this size, and committing them arguably *strengthens* Rule
1's acceptance test, since `git clone` plus one command reproduces the published
numbers with no external fetch. But the rule as written says otherwise, and Rule
7 forbids leaving a claim unverified.

So: amend the clause in writing with a size threshold and the justification, or
move the data out. **A silent deviation is the one option that is not
available.**

### B8 — Monthly README prose review
**Rule 7** (« Revue mensuelle obligatoire : chaque phrase du README est soit
vérifiée contre le code, soit supprimée »). **First: 2026-08-01, then monthly.**

The *numeric* half of Rule 7 is mechanically enforced:
`python -m scripts.generate_results --check` fails CI when a published figure
diverges from `results/metrics.json`. The *prose* half has no enforcement and no
schedule — and eight projects out of eight failed on prose, not on figures.

---

## 🟢 Declared and accepted

### B9 — The mutation audit is not CI-enforced
It reruns the whole suite once per mutation (12 × ~11 s today) and is invoked by
hand: `python -m scripts.mutation_audit`. Accepted because the audit is a
periodic instrument, not a per-commit gate, and because it exits 1 on any
deviation so it cannot pass silently when run.

**Turns 🔴 if** the suite gets fast enough to run it per-commit, or if a
mutation table entry is ever found stale in an audit.

### B10 — Exit-fee re-billing is one bar off in timing
`exit_fee_bias_bps` re-bills the uncharged exit at the **last** bar of its
segment; a continuous book would bill it at the **first** bar of the next
segment. Exact in amount, one bar off in timing, and without any effect on the
published figures today: one segment, one terminal exit (see B5).

**Turns 🔴 the moment `n_segments > 1`**, i.e. the day B5 is closed.

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
`manifest.json` pins BTC/USDT and ETH/USDT at 1h (26 303 rows each, one shared
hole at `2023-03-24T13:00:00Z`) and no published number uses them, which the
README states explicitly. Accepted because data is not code and the non-use is
declared at the point of presentation — the opposite of Fahm.io's pgvector,
where the UI button existed and the pipeline was never called.

**Turns 🔴 if** any 1h figure is published, or if the README stops saying they
are unused.

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

| # | Item | Rule | Closed | Commit |
|---|---|---|---|---|
| — | Boundary calendar holes internal to a spliced window unchecked | 2 | 2026-07-27 | `4a044ac` (M12) |
| — | Exit-fee bias asserted in prose but not computed | 3, 7 | 2026-07-27 | `4a044ac` |
| — | Segment-level contiguity guard covered by no mutation | 5 | 2026-07-27 | `1b86c76` (M13) |
| — | `POSTMORTEM.md` governing the repo from outside the repo | 7, 8 | 2026-07-27 | commit 10 (`f59d284`) |
| — | No debt ledger with due dates (Rule 8's amended clause had nothing to operate on) | 8 | 2026-07-27 | commit 10 (`f59d284`) — this file |
| B6 | GitHub Actions Node 20 deprecation | 4 | 2026-07-28 | `0d567c0` (D1) — annotation list verified empty on run 30321961542 |
| B13 | Mutation audit: dirty base accepted, no restoration guarantee on death | 5, 7 | 2026-07-28 | D2 — this commit |
