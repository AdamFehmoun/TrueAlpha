# POSTMORTEM — Bilan d'ingénierie de tous mes projets

**Date** : 2026-07-17
**Objectif** : apprendre de mes propres décisions avant de démarrer TrueAlpha (projet phare de recherche quant).
**Méthode** : 8 audits parallèles en profondeur (code, git log, structure, tests, CI, README, sécurité, reproductibilité), croisés avec les audits antérieurs (`github-audit.md` 05/2026, `cv-project-sheets.md` et `fahm-io-deep-dive.md` 07/2026). Chaque affirmation ci-dessous est ancrée dans un fichier, un commit ou un output réel.
**Ton demandé** : dur et honnête. C'est le cas.

---

## Vue d'ensemble

| Projet | Période | Commits | Tests | CI | État final | Cause de mort |
|---|---|---|---|---|---|---|
| **TrueSight** | 03→07/2026 (4 mois) | 161 | 360 (logique pure seulement) | Créée à J-2 | En prod, actif | — (vivant, mais dette listée non payée) |
| **QG Platform** | 05→06/2026 (6,5 sem) | 98 (5 auteurs) | 45 (HTTP seulement) | **Rouge depuis le jour 1** | Mort le jour de la démo | Deadline atteinte → plus de raison de continuer |
| **QG Agents** | 05→06/2026 (1 mois) | 44 | Décoratifs (0% pipeline) | Aucune | Fossile (le dev a continué dans la copie vendored) | Vendoring par `cp` → divergence → abandon |
| **Fahm.io** | 02→03/2026 (9 jours actifs) | 57 | **0** | Aucune | Prototype figé, repackagé en vitrine | Sprint épuisé, jamais repris |
| **glicko2-ts** | 05/2026 (2 commits) | 2 | 70 (excellents) | Verte, mais ne publie pas | **Jamais publié sur npm** | Le dernier 1% (une commande) jamais exécuté |
| **Portfolio-data** | 11/2025 (1 nuit) | 7 | 0 | Aucune | Notebook « Final » **syntaxiquement cassé** | Committé en plein vol à 2h40 du matin |
| **GAF-CNN** | ~05/2026 | non versionné | — | — | Conclusion contredit ses propres outputs | Diagnostic écrit mais jamais exécuté |
| **Java-Game** | 11/2025 | 1 | 0 | — | TP scolaire dumpé en un commit | N'a jamais été un projet |

Chiffre à retenir : **sur 8 projets, 1 seul est vivant** (TrueSight), et **0 n'a été terminé** au sens « livré, reproductible, et fermé proprement ».

---

# Analyses par projet

## 1. TrueSight (`~/code/codm-tournament-saas`) — le vaisseau amiral, sauvé in extremis

**Fiche** : SaaS tournois CODM. Next.js 16 / React 19 / TS strict, Supabase (136 migrations, 216 policies RLS), Stripe Connect, GPT-4o Vision fine-tuné, Discord bot. ~65 000 lignes TS + ~13 100 SQL. 161 commits, 03/2026→07/2026, prod réelle (`truesight.team`, ~160 users, 2 tournois, 847 lignes de stats OCR).

### Bonnes décisions (à reproduire)
- **`BACKLOG.md` comme mémoire d'ingénierie — la meilleure pratique de tout le portfolio.** 32 Ko de findings classés 🔴/🟠/🟢, datés, avec preuves `fichier:ligne`, re-vérifications datées, décisions produit actées. C'est ce qui a rendu possible le rattrapage sécurité de juillet.
- **Migrations SQL écrites comme des post-mortems** : `20260712100500_lockdown_function_grants.sql` documente la root cause, l'ordre de déploiement et le correctif durable (deny-by-default via `ALTER DEFAULT PRIVILEGES ... REVOKE`). `20260710120000` : GRANT SELECT par colonne, 18 colonnes explicites.
- **Kill switches motivés et gated côté serveur** : `lib/flags.ts` — chaque flag documente le POURQUOI (risque ANJ, légal), et `app/actions/betting.ts:17` re-vérifie le flag serveur car « the RPC REVOKE does not cover this action ».
- **Idempotence Stripe par insert-as-lock** (`app/api/stripe/webhook/route.ts:79-88`) : l'insert de `stripe_event_id` fait office de verrou, replay → `duplicate: true`.
- **Tests unitaires réels sur les algos** : 18/20 suites importent le vrai code (brackets SE/DE/RR/Swiss, Glicko-2, parimutuel, veto).
- **RUNBOOK avec baseline de restore chiffrée** (« A restore is faithful iff all seven numbers match »).
- **Secrets impeccables** : aucun `.env` jamais commité, crons sous `CRON_SECRET`, webhooks Discord Ed25519.

### Mauvaises décisions / anti-patterns
- **CI inexistante pendant 4 mois sur un produit en prod qui touche de l'argent.** `.github/workflows/ci.yml` créé le 13/07 — 2 jours avant le dernier commit. Pendant 4 mois, ~340 tests ne tournaient nulle part automatiquement. Et le lint y est `continue-on-error` avec 85 erreurs connues.
- **Sécurité rattrapée, pas conçue.** `GRANT EXECUTE ON ALL FUNCTIONS TO authenticated` (migration du 27/05) → toutes les RPC SECURITY DEFINER (argent, Mana, PII) appelables par n'importe quel compte pendant **~6 semaines**, corrigé le 12/07. PII de 155 profils lisible par `anon` jusqu'au 10/07. Le webhook Stripe non idempotent était identifié dans `PROJECT_CONTEXT.md` dès **mars**, traité le 13/07.
- **Sur-scope massif : ~40 % du produit construit puis désactivé.** Les 6 flags sont tous à `false`. Un système de paris **complet** (multi-paris, commandes Discord, leaderboards, DM) construit **puis** désactivé pour risque légal — la question légale devait précéder la construction, pas la suivre. 14 phases planifiées (white-label multi-tenant…) pour 160 users réels.
- **Zéro test sur les chemins réellement critiques** : `advance_winner` (SQL SECURITY DEFINER), 216 policies RLS, webhook Stripe, pipeline OCR — rien. Pire : `__tests__/stripe-payouts.test.ts` teste une **copie miroir** de la logique (« miroir de tournaments.ts », 0 import du code de prod) → le test passe même si la prod régresse.
- **Landing avec stats inventées** : « 10K+ Players / 500+ Tournaments / 99.9% Uptime » en dur + faux feed « LIVE », sur un produit vendu comme « source de vérité », corrigé seulement le 13/07. Indéfendable.
- **Docs contradictoires** : `PROJECT_CONTEXT.md` fossile (table renommée, archi bot fausse, ports différents de CLAUDE.md), README boilerplate create-next-app pendant 4 mois, refonte README **non commitée**.
- **77 Mo de dataset ML dans git** ; fichiers de 1 400-1 500 lignes ; 108 `as any` ; 35 branches locales mortes ; 0 tag ; historique bipolaire (mai : 2 commits ; juillet : 116, soit 72 % du projet).
- **Ops incomplet** : restore jamais exécuté end-to-end, Storage exclu des backups, 0 down migration — dettes **connues, écrites, non payées**.

**Verdict brut** : un solo dev techniquement fort a construit en 4 mois un produit dont l'ambition était sans rapport avec sa réalité. Le professionnalisme (CI, sécurité, idempotence) est concentré sur les 2 dernières semaines, en crunch post-audit, alors que la plateforme exposait des PII réelles pendant des semaines. Ce qui sauve le projet, c'est une discipline documentaire hors norme — mais elle documente une dette qu'elle ne rembourse pas.

---

## 2. QuantGenesis Platform (`~/quantgenesis-platform`) — bien conçu, vitrine mensongère, mort à la deadline

**Fiche** : plateforme « white-box meta-trading » (intent NL → 5 agents LLM → spec → code VectorBT → backtest sandbox E2B). FastAPI/Python + Next.js. ~3 500 lignes Python hors tests, 21 fichiers de prompts. 98 commits, 5 auteurs (Adam 52 = lead), 11/05→25/06/2026. **Arrêt net le 25/06 à 00h51 — le jour exact de la démo (« S7 — Public release »)**.

### Bonnes décisions (à reproduire)
- **Anti-look-ahead systématique** : 25 occurrences de `shift(1)` dans les 4 templates, exécution sur l'Open suivant, agent « Critique » dédié à la chasse au look-ahead avec retry-loop borné et jamais mis en cache.
- **Plafond de coût API codé et bloquant** : `BudgetExceededError` levée AVANT chaque appel Anthropic, chaque appel tracé dans `~/.quantgenesis_budget.json` — et le fichier prouve que ça a tourné en réel.
- **Contrat de métriques honnête** : en FALLBACK/ERROR, toutes les métriques sont `None`, jamais `0.0` (« fail-safe : on ne ment pas au front ») ; le cache ne stocke que les runs SUCCESS ; le front propage `null` en « n/a ».
- **Sandboxing sérieux** : AST pré-exécution (imports/`eval` bloqués), microVM E2B, « on ne retry jamais une security_violation ».
- **Tests API bien isolés** (E2B/LLM/yfinance/Redis stubbés) ; **clamping + invariants** sur les paramètres générés par LLM ; **ADRs datés, prompts v1→v11, commits conventionnels** qui documentent même les erreurs (« restaure le module complet tronqué par 62d02ef »).

### Mauvaises décisions / anti-patterns
- **CI rouge depuis le jour 1, jamais réparée en 6 semaines.** Le job frontend appelle `npm run type-check`/`lint` — **scripts inexistants** ; le job backend exige un Postgres sans bloc `services:`. Les deux jobs échouent à 100 % depuis le commit initial, et **97 commits sont passés dessus sans que personne ne regarde**. Le « CI/CD » du README est un mensonge par omission.
- **Le modèle de spread est du théâtre.** Le spread par classe d'actif est calculé, injecté en tête du code sandbox (`SPREAD = 0.0005`)… et **aucun template ne lit jamais la variable**. 2 commits, 1 ticket et des tests dédiés pour une feature qui n'a jamais modifié un seul backtest. Les tests vérifient que le spread est *transmis*, pas *utilisé* — du testing de plomberie.
- **Sur-vente du récit « l'IA génère le code »** : le README vend de la génération de code par IA ; la réalité est un sélecteur de 4 templates par mots-clés (`codeur.py:7` l'avoue : « Le Codeur est DETERMINISTE »). Le vrai codegen LLM a été branché puis revert **le jour même**.
- **Reproductibilité quant inexistante** : tous les backtests font `yf.download(period="2y")` → la fenêtre de données dépend du jour d'exécution ; données jamais versionnées (le dossier `data/` du README est **vide**) ; zéro walk-forward, zéro out-of-sample (grep confirmé) ; LLM appelé sans `temperature` fixée alors que le cache le suppose « déterministe ». Les métriques montrées au jury étaient des instantanés in-sample invérifiables.
- **L'executor ment par défaut** : métriques initialisées à `0.0` + `status: SUCCESS` même si rien n'a été parsé → un stdout vide produit un backtest « réussi ». Et `!pip install` sans pin de version à chaque run.
- **Zéro test sur le cœur quant** : templates, selector, validateur AST — rien en pytest ; les « 10 cas d'injection » vantés en commit sont des scripts manuels hors CI.
- **Artefacts commités** (`sandbox_logs.db`, `.coverage`, `tsconfig.tsbuildinfo`, `git_log.txt`, `*Zone.Identifier`), URL ngrok morte en dur dans le compose, README avec arborescence fausse et roadmap fantôme (Black-Litterman/HRP/walk-forward : aucune trace dans le code).
- **Vendoring** : `agents/` est une copie manuelle du repo sibling resynchronisée par `cp` ; le Quickstart du README ne peut pas fonctionner sans un second repo jamais mentionné.

**Verdict brut** : nettement au-dessus de la moyenne étudiante sur la conception défensive, mais la vitrine ment sur deux points structurants (codegen IA, coûts de transaction), la CI est rouge depuis 6 semaines sans que personne ne la regarde, et aucun backtest n'est reproductible. Le dépôt s'arrête le jour de la deadline : c'est un livrable de soutenance, pas une plateforme. Réutiliser les garde-fous, jeter le récit marketing.

---

## 3. QuantGenesis Agents (`~/quantgenesis-agents`) — tué par un `cp`

**Fiche** : pipeline 5 agents Claude (Brainstormer → Chef de Projet → Architecte → Critique → Conformité AI Act). Python 3.11, ~2 257 lignes + 1 862 lignes de prompts (22 fichiers, Architecte v1→v10). 44 commits (33 Adam / 6 Paul), 11/05→11/06/2026. **Abandonné avant le feature freeze : le dev a continué dans la copie vendored du repo platform, jamais rétro-porté.**

### Bonnes décisions (à reproduire)
- **Prompts versionnés immuables, traités comme du code** : « nouveau fichier = nouvelle version, jamais écraser », chaque bump tracé par un commit dédié (`prompts(architecte): v4 - anti-look-ahead-bias shift+1 obligatoire`). L'historique git raconte l'évolution du raisonnement quant — rare et précieux.
- **Garde-fou budgétaire avant chaque appel**, coût/tokens tracés, CLI dédiée — presque aucun projet étudiant LLM ne fait ça.
- **Codeur déterministe plutôt que codegen LLM** : décision d'architecture mature (coût zéro, code toujours valide, LLM réservé à la réflexion).
- **Ne jamais faire confiance au LLM pour les données critiques** : disclaimers légaux en constante Python, timestamp halluciné écrasé côté serveur, réparation de JSON tronqué.
- **Contrat d'intégration propre** (`api.py` ne lève jamais, retour normalisé), `.env` gitignoré dès J1, aucun secret dans l'historique.

### Mauvaises décisions / anti-patterns
- **Le péché capital : ~3 000 lignes dupliquées par `cp` entre les deux repos.** `budget.py` et `api.py` octet pour octet identiques côté platform ; 19 fichiers de prompts dupliqués ; resync **à la main** (la preuve est dans les permissions `cp` de `.claude/settings.local.json` et les commits « resync copie vendored »). Résultat prévisible : divergence (platform a v11 et des templates absents ici) et **mort du repo d'origine**, qui se fait encore passer pour la source de vérité. Il fallait un package pip, pas un `cp`.
- **README mensonger, installation impossible** : prescrit `cp .env.example .env` — le fichier **n'existe pas** ; `infisical login` — aucune config nulle part ; « outputs/ gitignorés » — **faux, 18 JSON commités (726 Ko)**.
- **Dépendances fausses** : `pyproject.toml` déclare 2 deps ; le code importe numpy/pandas, les tests exigent pytest/vectorbt — zéro occurrence dans `uv.lock`. `uv sync` + la commande documentée = échec garanti.
- **Tests décoratifs** : 154 lignes pour 2 257 de code ; les 5 vrais tests couvrent uniquement le code de Paul ; `test_brainstormer.py` n'est pas un test — un script qui fait 4 appels Opus **facturés** sans une seule assertion. 0 % de couverture sur orchestrator/budget/codeur. Aucune CI.
- **Bug réel de défaut dangereux** : `orchestrator.py:55` — un JSON imparsable du Brainstormer donne un score de faisabilité par défaut de **10** (au lieu de 0) → le pipeline continue sur du garbage en payant 4 appels Opus de plus.
- **L'audit trail « AI Act » ment** : `self.version = "v1"` figé dans `base.py:26` → chaque compliance log enregistre « v1 » alors que l'Architecte tournait en v10. Pour un projet dont l'argument central est l'auditabilité, la seule métadonnée qui compte est fausse.
- **Tout-Opus non tiéré** (même le résumeur), tarification **surestimée ~3×** dans `budget.py`, limite dupliquée en dur, fichier budget sans verrou (check-then-act).
- **Features jamais branchées** : `codeur.py` et `agent_quant.py` (357 l.) importés nulle part par l'orchestrateur de CE repo ; `.venv` de 25 Mo non gitignoré ; `__pycache__` orphelin d'un module supprimé ; prints de debug en prod, aucun `logging`.

**Verdict brut** : un prototype honorable — versioning de prompts, budget bloquant, codeur déterministe sont au-dessus du niveau habituel — exécuté avec une discipline qui s'effondre dès qu'il faut industrialiser. Le vendoring par `cp` a tué le repo en 4 semaines. Le README ment, les tests sont décoratifs, et la traçabilité — l'argument de vente — est du théâtre.

---

## 4. Fahm.io (`~/projet-asso-fintech`) — une fintech sans un seul test

**Fiche** : SaaS de trésorerie multi-tenant pour assos étudiantes. Next.js 16 / TS strict, Supabase RLS, Stripe, GPT-4o (OCR + classification), Upstash. ~8 850 lignes TS + ~1 025 SQL. 57 commits, solo, **9 jours actifs seulement** (3 rafales : 9/02, 24/02, 8/03) puis 2 commits de vitrine en mai. Inactif depuis ~4,5 mois.

### Bonnes décisions (à reproduire)
- **Argent en centimes entiers partout**, convention tenue sur ~20 sites.
- **Env typé fail-fast** (`src/lib/env.ts` : schéma Zod complet, crash au boot avec message précis) + `.env.example` documenté. **Aucun secret jamais commité** (vérifié sur tout `git log -p`).
- **Webhook Stripe sérieux** : signature avant tout write, idempotence `UNIQUE stripe_event_id` + fallback 23505, insertion en `pending` (validation humaine) plutôt que comptabilisation automatique.
- **Vrai audit de sécurité suivi d'effets** : la migration `20260226_security_rls_fix.sql` corrige des failles réelles documentées dans le SQL même, avec défense en profondeur (RLS + `REVOKE anon` + `requireMembership`).
- **Sanitisation anti prompt-injection sur les données venant de la DB**, pas seulement sur l'input utilisateur — threat model rare.
- **README final honnête** (« What's built vs. roadmap » admet l'absence de tests et le pgvector non branché).

### Mauvaises décisions / anti-patterns
- **Zéro test, zéro CI, sur une app qui manipule de l'argent.** Le seul filet revendiqué (« hook pre-push ») **n'existe pas** — `.git/hooks/` ne contient que des samples. Pour de la fintech, c'est disqualifiant.
- **La conséquence directe : la feature phare est cassée et personne ne l'a vu.** Le bouton « Acheter le billet » exige le rôle… **trésorier** (un fix de sécurité a tué le cas d'usage acheteur), et la page publique n'a aucun bouton d'achat. Un client ne peut littéralement pas payer. Symptôme exact de l'absence de tests.
- **La base de données n'est pas reconstructible.** La migration de base fait **0 octet** ; `grep CREATE TABLE` sur les migrations = 0 résultat ; le vrai schéma vit hors chaîne de migration ; les policies Storage du bucket `receipts` ne sont dans **aucun** fichier versionné — la sécurité des justificatifs dépend de clics dashboard perdus.
- **Faille confirmée exploitable** : `get_hierarchical_budget` — SECURITY DEFINER, sans check d'autorisation, sans `search_path` pinné, `GRANT ALL TO anon` → n'importe qui avec la clé anon publique lit le budget de n'importe quelle org. (Fix prêt dans `fahm-io-deep-dive.md`.) Plus `get-receipt-url.ts` : URL signée sur chemin arbitraire sans check d'appartenance.
- **Sur-scope massif en 9 jours** : multi-tenant RBAC/RLS + OCR + LLM + pgvector + Stripe Connect + CSV + rapprochement + PDF + audit log + PWA. Résultat : le pipeline pgvector (colonne, index ivfflat, RPC, bouton sync) est **entièrement construit et jamais appelé** — dead code coûteux en prod.
- **Debug par déploiement** : 5 commits en 15 minutes le 8 mars pour faire marcher une route, plus un « force redeploy ». La prod Vercel servait d'environnement de test.
- **Repo pollué** : `repomix-output.xml` (213 Ko, re-commité 3×), 2 lockfiles concurrents, 4 fichiers vides, `supabase/.temp` (fuite du project-ref), `architecture.txt` périmé, composants clients de 600+ lignes, 18 `any`.
- **Historique amputé** : premier commit « Reset: Nouvelle architecture propre » = tout l'historique antérieur (et ses leçons) jeté.

**Verdict brut** : un dev solo qui apprend vite et applique de vrais patterns de production — mais qui a construit une fintech sans un seul test, et ça se voit : la billetterie est cassée pour tout acheteur et personne ne l'a détecté. Développé en 3 sprints frénétiques, débogué en prod, abandonné de fait le 8 mars ; les commits de mai sont du polissage de vitrine, pas de la vie du produit.

---

## 5. glicko2-ts (`~/glicko2-ts`) — excellent, et mort-né

**Fiche** : lib TypeScript Glicko-2, zéro dépendance runtime, extraite de TrueSight. 469 lignes src, 652 lignes tests (70 tests), tsup dual ESM/CJS, Biome, CI Node 18/20/22. **2 commits** : « initial release » (24/05, amendé le 25/05) + scoping npm (**17/07 à 14h30 — aujourd'hui, quelques heures avant cet audit**). Tag `v0.1.0`. **Jamais publié sur npm. Zéro consommateur.**

### Bonnes décisions (à reproduire — c'est le gold standard du portfolio)
- **Validation contre le papier de Glickman** : l'exemple canonique complet reproduit à 1e-6 (`toBeCloseTo(1464.0506705, 6)`), avec commentaire honnête sur l'écart d'arrondi du papier. Le test le plus important qu'une lib Glicko-2 puisse avoir.
- **Property-based testing (fast-check)** : monotonie des victoires, symétrie gain/perte, E(A,B)+E(B,A)=1, finitude, rejets — rarissime à ce niveau de soin.
- **Test de non-régression du piège classique** : `updatePeriod` batch prouvé différent d'une boucle naïve de `rate()`.
- **Packaging au cordeau** : `exports` complets, dual build, `prepublishOnly: pnpm verify`, seuils de couverture imposés (95 %), couverture réelle 100 %/97,3 %.
- **Algorithme d'Illinois fidèle au papier** (pas la version Newton des ports douteux), JSDoc avec références aux étapes du papier.

### Mauvaises décisions / anti-patterns
- **8 semaines pour ne pas taper une commande de 30 secondes.** Le README initial l'avouait : « the npm package is not yet published ». Cette note est restée du 24/05 au 17/07. Le commit de scoping date d'aujourd'hui — un réflexe déclenché par l'audit, pas une publication. **Une lib non publiée n'est pas une lib, c'est un exercice.**
- **Historique mono-commit maquillé** : 4 102 lignes dans un unique « initial release », amendé après coup. Zéro trace de l'extraction depuis TrueSight — la provenance revendiquée par le README est invérifiable.
- **CI qui teste mais ne publie jamais** : pas de job release, pas de `NPM_TOKEN`, pas de changesets. L'absence de pipeline de publication est la preuve structurelle que publier n'a jamais été une étape planifiée.
- **Tag incohérent** : `v0.1.0` pointe sur le paquet non scopé ; renommage sans bump ni CHANGELOG (il n'y en a aucun).
- **Sur-vente marginale exactement là où le projet vend de la rigueur** : benchmark « ~2.1 M ops/sec » = résultat codé en dur d'un micro-bench artisanal de 31 lignes ; README dit « enforced 97%+ » alors que le seuil réel est 95 %.
- **Code mort défensif** qui avale silencieusement un état impossible (`if (!entry || !next) continue`) ; doublons de `Rating` dans `updatePeriod` silencieusement écrasés — trou dans l'argument central d'atomicité.

**Verdict brut** : le paradoxe le plus net du portfolio. Objectivement de qualité publiable — et objectivement mort : 2 commits en 8 semaines, zéro consommateur, zéro publication. Tout l'effort dans les 99 % techniques, rien dans le 1 % qui crée de la valeur. On re-polit l'emballage au lieu de livrer.

---

## 6. Portfolio-data (`~/Portfolio-data`) — le « Final » qui ne s'exécute pas

**Fiche** : EDA salaires Data Science (CSV Kaggle 93 597 lignes), régression « prédiction salaire », simulation d'épargne. 100 % notebooks, 348 lignes de code, 0 cellule markdown. 7 commits : 6 en ~30h (le cœur entre **00h50 et 02h40 du matin**), 1 cosmétique 6 mois plus tard. Abandonné.

### Bonnes décisions
- Les chiffres du README correspondent aux sorties réelles (pas de résultats inventés).
- Honnêteté statistique ponctuelle (« échantillon trop faible de n=2 » signalé).
- Croisement de sources (Kaggle vs grilles Silkhom) et détection d'un biais d'échantillonnage — le seul apport analytique réel.
- Commits atomiques et descriptifs au départ ; aucun secret.

### Mauvaises décisions / anti-patterns
- **Le notebook committé est syntaxiquement cassé** (liste jamais fermée, cellule 12) → le livrable principal ne s'exécute pas. Committé sous le message « Final ».
- **Une cellule contredit sa propre sortie** : code édité après exécution (`/1000` dans le code, `×1000` dans la sortie) → toute réexécution donne un résultat 10⁶ fois différent. La définition même du résultat non vérifiable.
- **Cellules en ordre inverse d'exécution** (execution_count 28→3 de haut en bas), chargement du CSV en bas du notebook. « Run All » plante à la cellule 0.
- **« Prédiction IA » survendue** : régression à 1 feature fabriquée (mapping arbitraire EN=1…EX=10), fit sur 100 % des données, `train_test_split` importé jamais utilisé, **zéro métrique** (pas de R²), et un « 198208.73 $ » à deux décimales — fausse précision d'un modèle jamais validé.
- **Constantes magiques non sourcées** qui portent 100 % de la conclusion phare (taux d'imposition, coûts de la vie inventés, 2 conventions de change contradictoires dans le même notebook), sans analyse de sensibilité.
- **Dataset 5 Mo commité sans provenance** (pas d'URL, licence, date, hash), « (2024) » dans le README pour des données 2020-2025.
- **Zéro infra** : pas de requirements.txt, .gitignore, LICENSE, tests, CI. Commentaires LLM-tuteur non réappropriés (« Tu devras ajouter… », « L'ajout magique »).

**Verdict brut** : projet d'une nuit figé en plein vol et présenté comme un livrable. Les chiffres sont honnêtes mais personne ne peut les régénérer — reproductibilité nulle au sens strict. La valeur est analytique (bon réflexe critique sur la donnée), soutenue par aucune pratique d'ingénierie.

---

## 7. Notebook GAF-CNN (`~/Market_Regime_GAF_CNN.ipynb`) — l'autopsie sans corps

**Fiche** : classification de régimes (Bull/Bear/Volatile) sur 5 actifs, GAF → CNN, baselines LogReg/RF, MobileNetV2, ablation, Grad-CAM. 30 cellules, exécution séquentielle 1→14, **mais 2 cellules jamais exécutées : précisément le diagnostic des fuites (§10.1) et le Grad-CAM**. Re-cadré en en-tête comme « negative result ». Non versionné dans un repo. **C'est le projet le plus proche de TrueAlpha — chaque ligne ci-dessous est un avertissement direct.**

### Bonnes décisions
- **Le re-cadrage honnête en negative result** (« le 99,8 % du RF ne mesure aucune compétence prédictive : fuite de labels ») — la bonne posture de recherche, par écrit, en tête de document.
- **Le diagnostic §10.1 est analytiquement excellent** : circularité labels/features, fuite temporelle par chevauchement (28/32 points partagés entre fenêtres voisines + split aléatoire), et la vraie cause de l'effondrement du CNN (le GAF normalise chaque fenêtre et **efface μ et σ — exactement ce qui définit les labels**).
- Baselines présentes avec critère assumé (« si le CNN ne bat pas les baselines, le projet n'a pas de valeur ajoutée ») ; scaler fit sur le train seul ; métriques du tableau final toutes traçables dans les outputs ; seed fixé ; fonctions nommées ; pas de chemins/clés en dur.

### Mauvaises décisions / anti-patterns
- **La conclusion (§12) est un mensonge résiduel** : jamais réécrite, elle affirme encore « le CNN dépasse significativement les baselines » — contredit frontalement par l'output juste au-dessus (CNN 0.339 vs RF 0.998) ; « BatchNorm+Dropout essentiels » alors que l'ablation montre l'**inverse** ; « Grad-CAM montre… » alors que la cellule n'a jamais tourné. Trois claims faux dans la conclusion d'un notebook qui se présente comme honnête.
- **Les preuves de l'honnêteté n'existent pas** : les vérifications « en 10 lignes » annoncées au §10.1 n'ont jamais été exécutées. Un notebook d'analyse d'échec dont le diagnostic n'a pas tourné, c'est une autopsie sans corps.
- **Le code n'a jamais été corrigé, seulement commenté** : le split shuffle sur fenêtres à 87 % chevauchantes est toujours là, aucun split temporel purgé, les seuils de labels sont toujours calculés sur **tout** le dataset (test inclus).
- **Tâche descriptive vendue comme prédictive** : on classe une fenêtre par ses propres statistiques — aucune prévision du futur, donc aucune valeur quant, pas de backtest, pas de Sharpe.
- **Labels poolés sur actifs hétérogènes** (BTC et GLD dans le même percentile de vol) → « Volatile » encode l'identité de l'actif ; baseline naïve (classe majoritaire = 0.339 = exactement le score du CNN) jamais nommée.
- **Ablation sur-interprétée** : 1 seed, 1 run/config, sur un modèle qui n'apprend rien (val_accuracy figée dès l'epoch 1).
- **Reproductibilité fragile** : pip install sans versions, données yfinance re-téléchargées (`auto_adjust=True` = valeurs qui changent rétroactivement), pas de déterminisme GPU — le seed 42 donne une illusion de reproductibilité.

**Verdict brut** : schizophrène — une tête d'une honnêteté exemplaire greffée sur un corps jamais corrigé. Un relecteur quant lira la conclusion, regardera le tableau juste au-dessus, et fermera le fichier. Le réflexe d'auto-critique est exactement celui qu'il faut emporter dans TrueAlpha — à condition de **l'exécuter**, pas seulement de le rédiger.

---

## 8. Java-Game (`~/Java-Game`) — n'aurait jamais dû être public

**Fiche** : TP « World of Zuul » (Barnes & Kölling), S1 ESIEE, 450 lignes dont ~30 % fournies par l'enseignant. **1 commit** (« Zul V17 » — même le nom est mal orthographié), 95 % du poids du repo = Javadoc générée + PDF.

**À retenir** : `.gitignore` propre, encapsulation correcte, `HashMap` pour les sorties — du respect de consignes, pas des décisions d'ingénierie.

**À ne pas refaire** : git utilisé comme clé USB (1 commit pour « V17 » versions locales) ; artefacts générés commités ; pas de `main` (inexécutable hors BlueJ) ; README boilerplate avec « USER INSTRUCTIONS: voila » ; bug de casse rendant le jeu quasi injouable (`"Up"` enregistré, lookup sensible à la casse) ; pattern Null Object déclaré mais code mort (branche inatteignable) ; magic strings dupliquées dans 2 fichiers ; renommage du repo pour masquer l'origine TP alors que `doc/logfile.txt` trahit « ESIEE S1 A3P TP 3 ».

**Verdict brut** : le jugement « à ne pas mettre sur un CV » est entièrement mérité. Comme TP de S1, honnête ; comme pièce de portfolio public, il dessert activement son auteur. À archiver en privé.

---

# SYNTHÈSE TRANSVERSALE

## A. Les patterns d'erreur qui reviennent chez toi, projet après projet

### 1. Tu ne finis jamais — tu t'arrêtes quand la pression externe disparaît
Le pattern le plus grave et le plus systématique. **QG Platform meurt le jour exact de la démo** (25/06, 00h51). QG Agents est abandonné avant son propre feature freeze. Fahm.io s'arrête après le 3e sprint. glicko2-ts attend 8 semaines une commande de 30 secondes. Portfolio-data est committé « Final » en plein vol à 2h40. Le README de TrueSight est resté un boilerplate pendant 4 mois et sa refonte est **encore non commitée**. Tu ne décides jamais « c'est fini » — tu cesses juste de revenir. Aucun projet n'a de tag de release, de CHANGELOG, ou de commit de clôture.

### 2. Tu construis des features que tu n'actives jamais (sur-scope structurel)
~40 % de TrueSight vit derrière des flags à `false`, dont un système de paris **complet** désactivé après coup pour une question légale qui devait être posée avant. Le pgvector de Fahm.io : colonne + index + RPC + UI de sync, **zéro appelant**. Le spread de QG Platform : calculé, injecté, testé — **jamais lu**. `codeur.py` et `agent_quant.py` de QG Agents : jamais importés par l'orchestrateur. Le leaderboard carrière de TrueSight : index construits, aucune page. À chaque fois, le coût est double : le temps de construction perdu, plus le mensonge implicite dans la vitrine (« regarde tout ce que ça fait » — non).

### 3. Tes docs et tes conclusions mentent — pas par malhonnêteté, par non-synchronisation
C'est ton pattern le plus dangereux pour un projet de **recherche**. La conclusion du GAF-CNN contredit le tableau juste au-dessus. La landing TrueSight affichait « 10K+ Players » pour 159 profils réels. Le README QG Platform vend du codegen IA qui est un sélecteur de templates, et un CI/CD qui échoue depuis le jour 1. Le README QG Agents documente une installation impossible et son audit trail AI Act enregistre « v1 » pour des prompts en v10. Fahm.io revendique un hook pre-push inexistant et un `bigint` qui est un `integer`. Même glicko2-ts, ton projet le plus rigoureux, écrit « 97%+ enforced » pour un seuil à 95 %. **Huit projets, huit écarts doc/réalité.** La cause commune : tu écris la doc/la conclusion à un instant T et le code bouge (ou n'a jamais été là) — et rien ne re-vérifie jamais.

### 4. Tu testes ce qui est facile, jamais ce qui est critique
360 tests sur TrueSight — zéro sur les RPC d'argent, les 216 policies RLS, le webhook Stripe, l'OCR ; et 2 suites testent des **copies miroir** du code (le pire des faux-positifs : un filet peint sur le sol). 45 tests sur QG Platform — zéro sur les templates de backtest et le validateur AST. Fahm.io : **zéro test sur une fintech**, et la conséquence est mesurable : la billetterie est cassée et personne ne l'a su. QG Agents : un « test » qui brûle 0,35 € d'API sans une assertion. La seule exception — glicko2-ts, property-based, validé contre le papier — prouve que tu **sais** faire ; tu ne le fais que quand le périmètre est petit et pur.

### 5. La CI arrive trop tard, ou tourne rouge dans l'indifférence
TrueSight : 4 mois de prod avec de l'argent réel avant la première CI (créée à J-2 du dernier commit). QG Platform : CI rouge dès le jour 1, **97 commits passent dessus en 6 semaines sans un regard**. Fahm.io, QG Agents, Portfolio-data : aucune CI. Une CI rouge ignorée est pire qu'aucune CI : elle entraîne à ignorer les signaux.

### 6. Tes résultats quant ne sont pas reproductibles — systématiquement
QG Platform : `yf.download(period="2y")` → chaque jour une fenêtre différente, données jamais versionnées, in-sample pur, pas de walk-forward. GAF-CNN : données re-téléchargées avec `auto_adjust=True` (valeurs rétroactivement modifiées), pas de pin de versions, split shuffle avec fuite. Portfolio-data : notebook qui ne s'exécute pas, cellule éditée après exécution. Fahm.io : la DB elle-même n'est pas reconstructible depuis le repo. **Aucun de tes résultats numériques publiés n'est régénérable par un tiers.** Pour TrueAlpha, c'est LE point de vie ou de mort.

### 7. La sécurité et la rigueur arrivent en rattrapage, pas en conception
TrueSight : toutes les RPC SECURITY DEFINER ouvertes à tout compte pendant 6 semaines, PII lisible par `anon`, webhook Stripe identifié comme risque en mars et traité en juillet. Fahm.io : faille cross-tenant exploitable via la clé anon publique ; audit sécurité sérieux — mais après coup. Le pattern : tu es excellent en *audit* (tu trouves tes propres failles, tu les documentes admirablement) et faible en *prévention* (deny-by-default seulement après l'incident).

### 8. Hygiène de repo : tu committes tout ce qui traîne
77 Mo de dataset dans TrueSight. `repomix-output.xml` re-commité 3× dans Fahm.io + 2 lockfiles + 4 fichiers vides. `sandbox_logs.db`, `.coverage`, `git_log.txt`, `Zone.Identifier` dans QG Platform. 726 Ko d'outputs « gitignorés » (faux) dans QG Agents + `.venv` de 25 Mo hors gitignore. 95 % de Javadoc générée dans Java-Game. 5 Mo de CSV sans provenance dans Portfolio-data. Plus les cadavres : 35 branches mortes (TrueSight), ~10 (QG Platform), historiques détruits (« Reset » Fahm.io, amend glicko2-ts, mono-commits).

### 9. Le vendoring par copier-coller
Un cas unique mais si coûteux qu'il mérite sa ligne : ~3 000 lignes synchronisées par `cp` entre QG Agents et QG Platform ont tué le repo d'origine en 4 semaines et créé deux vérités divergentes.

### 10. « Documenter la dette » te sert d'absolution pour ne pas la payer
Le méta-pattern qui relie tout. Le BACKLOG de TrueSight liste admirablement des dettes… non payées (restore jamais exécuté, 85 erreurs lint « acknowledged », i18n 5 %). Le §10.1 du GAF-CNN diagnostique brillamment des fuites… jamais corrigées dans le code, dans des cellules… jamais exécutées. Les « honest caveats » de tes README sont excellents — et deviennent un permis de ne rien réparer. **Écrire « je sais que c'est cassé » n'est pas une correction. C'est un aveu daté.**

## B. Tes forces récurrentes (à capitaliser sur TrueAlpha)

1. **Les garde-fous d'exécution — ta vraie signature.** Budget API bloquant vérifié AVANT l'appel (QG ×2, tracé en réel), sandbox AST + microVM, contrat « null, jamais 0.0 » pour les métriques manquantes, clamping + invariants sur les params LLM, kill switches server-side motivés, idempotence Stripe insert-as-lock, `shift(1)` systématique. Tu penses « comment ça peut mal tourner » mieux que la plupart des devs seniors.
2. **La discipline documentaire d'élite** : BACKLOG.md avec preuves fichier:ligne, migrations écrites comme des post-mortems, prompts versionnés immuables v1→v11 dont l'historique git raconte le raisonnement, ADRs, RUNBOOK avec baseline chiffrée. C'est rare et précieux — il manque juste le remboursement (cf. A.10).
3. **La capacité d'auto-critique honnête** : re-cadrer un projet en « negative result », trouver tes propres failles de sécurité, écrire des sections « Limitations » exactes au grep. La matière première d'un bon chercheur.
4. **La rigueur algorithmique quand le périmètre est petit et pur** : glicko2-ts (validation au papier à 1e-6, property-based testing, Illinois) est du travail de qualité professionnelle. Le Glicko-2 et le parimutuel de TrueSight aussi.
5. **La vitesse de construction brute** : 65k lignes en 4 mois solo en prod ; une fintech multi-tenant en 9 jours. C'est une force réelle — et le carburant de ton sur-scope (A.2). À canaliser, pas à célébrer.
6. **L'hygiène des secrets — sans faute sur 8 projets** : aucune clé jamais commitée, vérifié sur les historiques complets. Env typés fail-fast, argent en centimes entiers. Ces réflexes-là sont acquis.

## C. Le cycle de vie type de tes projets (le méta-pattern)

> Démarrage en rafale nocturne → construction à grande vitesse avec de vrais garde-fous d'exécution → scope qui explose (chaque idée est construite immédiatement) → tests uniquement sur les parties pures et amusantes → doc/vitrine écrite au présent pour un futur qui n'arrivera pas → deadline externe (démo, rendu) → crunch final où la rigueur arrive d'un coup → **la deadline passe → le projet meurt le jour même** → des mois plus tard, une passe de « polissage de vitrine » (README, renommage) pour le CV, sans toucher au fond.

TrueAlpha n'aura **pas de deadline externe**. Si tu ne remplaces pas la pression du jury par une discipline interne, il mourra plus vite que les autres — ou pire : il survivra en produisant des résultats invérifiables.

---

# 10 RÈGLES POUR TRUEALPHA

Concrètes, vérifiables, conçues pour casser chacun des patterns ci-dessus.

### 1. Aucun résultat sans reproductibilité totale — dès le premier backtest
Toute donnée est téléchargée par un script versionné avec **dates absolues** (jamais `period="2y"`), stockée en parquet avec un manifest (source, date de téléchargement, hash SHA-256), hors git ou en DVC. Versions pinnées (lockfile), seeds fixés, déterminisme GPU activé. **Test d'acceptation : `git clone` + une commande régénère exactement les chiffres publiés, sur une autre machine.** Un résultat non régénérable n'existe pas — il est interdit de le noter où que ce soit. *(casse A.6 — le pattern qui a invalidé QG Platform et le GAF-CNN)*

### 2. Le test anti-leakage s'écrit AVANT le modèle
Avant toute feature ou label : split **temporel** avec purge/embargo (jamais de shuffle sur des fenêtres chevauchantes), et un test automatisé qui vérifie qu'aucune feature à l'instant t ne dépend de données > t et que les statistiques de labellisation sont fit sur le train seul. Le GAF-CNN a été invalidé par 3 fuites **connues de toi et documentées par toi** — cette fois le contrôle est du code qui tourne en CI, pas un paragraphe. *(casse A.6 et l'échec GAF-CNN)*

### 3. Baseline d'abord, conclusion après l'output — jamais l'inverse
Aucune métrique de modèle n'est présentable sans les baselines naïves dans le même tableau : classe majoritaire, buy-and-hold, momentum simple, AR(1). Et règle d'écriture absolue : **une phrase de conclusion ne peut citer que des chiffres présents dans un output exécuté du même run.** La conclusion se rédige en dernier, jamais en avance sur les résultats espérés. *(casse A.3 — le mensonge résiduel du GAF-CNN et les stats inventées de TrueSight)*

### 4. CI verte au commit 1, et une CI rouge bloque tout
Le premier commit du repo contient la CI (lint bloquant, typecheck, tests, et le test de repro de la règle 1). Règle dure : **CI rouge > 24h = interdiction de commit de feature jusqu'à réparation.** Pas de `continue-on-error`, pas de « 85 erreurs acknowledged ». QG Platform a prouvé qu'une CI rouge ignorée 6 semaines vaut moins que rien. *(casse A.5)*

### 5. Les tests couvrent le moteur, pas la périphérie — et importent toujours le code réel
Priorité de test inversée par rapport à tes habitudes : le calcul de PnL, les coûts de transaction, le split temporel, la construction des features — AVANT tout test d'API ou d'UI. Property-based tests sur les invariants (PnL nul si aucun trade, coûts strictement décroissants sur la performance, symétries) comme dans glicko2-ts — c'est ton meilleur pattern, applique-le au cœur. **Interdiction absolue des copies miroir** : un test qui n'importe pas le code de prod est supprimé. *(casse A.4)*

### 6. Une seule expérience à la fois, terminée avant la suivante
« Terminée » = code mergé + résultat reproductible + note d'expérience écrite (hypothèse, protocole, résultat, décision) — y compris et surtout pour les résultats négatifs. Interdiction de commencer l'expérience N+1 si la N n'a pas sa note. Interdiction de construire une infra « pour plus tard » : pas de feature derrière un flag `false`, pas de module sans appelant, pas de roadmap dans le README (les idées vont dans des issues, gratuites à écrire, gratuites à fermer). Le sur-scope t'a coûté ~40 % de TrueSight et le pgvector de Fahm.io. *(casse A.2)*

### 7. Le README est un contrat vérifié, pas une brochure
Chaque claim chiffré du README doit être la sortie d'une commande reproductible (idéalement générée par script en CI). Revue mensuelle obligatoire : chaque phrase du README est soit vérifiée contre le code, soit supprimée. Tu as 8 projets sur 8 avec un écart doc/réalité — la seule défense est mécanique. *(casse A.3)*

### 8. Cadence de commits imposée, dette avec date d'échéance
Commits petits et quotidiens les jours travaillés — jamais de « backup », « Final », mono-commit de 4 000 lignes ou « Reset » destructeur d'historique. Tags de version aux jalons. Et le BACKLOG (garde-le, c'est ta meilleure pratique) gagne une règle nouvelle : **tout item 🔴 porte une date d'échéance ; un 🔴 échu bloque les features exactement comme une CI rouge.** Documenter la dette sans échéance, c'est l'absolution sans pénitence — TrueSight l'a prouvé. *(casse A.1, A.10)*

### 9. Sécurité et budget by design, deny-by-default dès le jour 1
Ce que tu as appris en le payant : deny-by-default sur tout accès (le `ALTER DEFAULT PRIVILEGES ... REVOKE` de TrueSight, dès la **première** migration cette fois) ; toute question légale/compliance tranchée AVANT de construire la feature (pas de betting-system-then-flag-off) ; budget API bloquant vérifié avant chaque appel avec tarifs exacts et testés (pas la surestimation ×3 de QG Agents) ; limites en config, pas en dur ; secrets : continue exactement comme aujourd'hui, c'est ton sans-faute. *(casse A.7)*

### 10. Définis « fini » le jour 1 — et livre à date fixe, pas à perfection atteinte
Avant la première ligne de code, écris dans le repo la définition de « livré » (ex. : « rapport reproductible publié + repo taggé v1.0 + résultats régénérables par un tiers ») et une date. À cette date, tu livres l'état réel, proprement : tag, CHANGELOG, limitations honnêtes — même si c'est un résultat négatif (c'est de la recherche : un négatif propre EST un livrable). Un projet sans deadline externe meurt par abandon silencieux ; un projet « presque prêt » pendant 8 semaines est un projet mort (glicko2-ts). Et si tu extrais une lib : elle est publiée le jour où elle est prête, pas « publiable ». **Le dernier 1 % — publier, taguer, annoncer, fermer — n'est pas du polish : c'est la seule partie qui transforme du code en actif.** *(casse A.1 — ton pattern n°1)*

---

## Le mot de la fin

Tu n'as pas un problème de compétence — glicko2-ts, les garde-fous QuantGenesis et le rattrapage sécurité de TrueSight sont du travail de niveau professionnel. Tu as un problème de **véracité différée** : tes systèmes, tes docs et tes conclusions décrivent un état désiré que le code n'a pas (encore) atteint, et rien ne force jamais la convergence. Sur un SaaS étudiant, ça coûte des heures. Sur un projet de recherche quant, **c'est fatal** : un chercheur dont les conclusions contredisent ses propres outputs, dont les backtests ne sont pas régénérables et dont le README survend, perd la seule chose qui compte — la confiance dans ses chiffres. Les 10 règles ci-dessus ont un seul but commun : rendre le mensonge involontaire mécaniquement impossible sur TrueAlpha.

## AMENDEMENTS
### 2026-07-27 — Règle 1, clause « hors git ou en DVC »
Les 4 parquets de data/raw/ (2,44 Mo au total, pack .git = 1,99 Mo) sont
commités. La clause « hors git » visait le pattern A.8 (77 Mo de dataset ML
dans TrueSight) ; à cette taille elle ne s'applique pas, et commiter les
données renforce le test d'acceptation de la règle 1 (clone + une commande
régénère les chiffres, sans fetch réseau). Seuil retenu : au-delà de 50 Mo
cumulés, les données sortent du git ou passent en DVC. Amendement écrit, pas
dérogation silencieuse.
