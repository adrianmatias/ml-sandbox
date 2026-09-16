# Code standard for agent-written Python

**Compact on purpose.** Rules, not essays: a standard that takes ten minutes to read is a standard
that gets skimmed. Every rule carries the measurement that earned it — the arms that were built to
this standard, and the arm that was not, are in this repository and their numbers are in
`docs/FINDINGS.md` §29.

**How to use it.** Paste the "Drop-in rules" block into a project's `AGENTS.md` / `code.md`. The
rules are language-shaped for Python but the bolded clause in each is the portable part — the same
standard is written for any language by keeping the clauses and dropping the tool names.

---

## Drop-in rules

```markdown
## code standard (agent-written Python)

**Shape**
1. One domain model per project: entities concentrated in a `domain` module, not spread as dicts.
   Money is a decimal type, never a float; ids and amounts get semantic types, not bare `str`/`int`.
2. Entities are frozen dataclasses. Behaviour that belongs to an entity lives on it; free functions
   take and return entities.
3. Declarative over imperative: describe the data (types, schemas, tables) and let one interpreter
   walk it. No 300-line functions with nine locals.
4. Every function annotated, including `-> None`. `Any` is banned at the boundary: decode into a
   typed model first, then work in the typed domain.
5. Errors are a modelled hierarchy with one base per layer, not bare `raise ValueError`.

**Gates** (all of them enforced, none advisory)
6. `ty` (or the project's checker) is a gate — strict, warnings are errors.
7. `ruff` is a gate with a *stated* rule set. The one this repository uses:
   `reference/python-tooling/ruff.toml` — F, E, W, I, N, UP, B, A, C4, DTZ, T20, SIM, PTH, RUF,
   S, TRY, BLE, ARG, RET, PIE; tests ignore only `S101`.
8. Complexity is a gate: `python3 tools/complexity.py <paths>` — per function, budget 12 (15 in
   tests). Over budget means split the function; never raise the budget for one file.
9. Dead code is a gate: `vulture <package>` at **default** confidence, package only, tests excluded.
   Findings are fixed or whitelisted one by one.
10. Every gate must be seen to fail: `bash tools/gate-drill.sh <package>` plants the defect each gate
    exists to catch and checks it goes red.
```

---

## Why each rule, in one line

| # | rule | the measurement |
|---|---|---|
| 1–2 | domain entities, frozen, concentrated | the arm that followed this has **20 entities and 3 semantic types** in one `domain` module with **zero** unannotated functions; the arm that did not has **4 entities, 0 semantic types, 2 unannotated functions**, and its worst function is **CC 28** against the other's **12** |
| 3 | declarative over imperative | mean complexity per function **3.66 → 2.08**; functions over CC 10: **6 → 2**; over-budget functions: **4 → 0** |
| 4 | complete annotations, no `Any` at the boundary | `ty` found three real defects on first run in the enforced arm, including a name used before definition on a path no test covered. In the unstructured arm the same checker class would have been the only thing between a typo and production |
| 5 | modelled errors | 25 exception classes under one base in the enforced arm; a raised string in the other cannot be caught by kind |
| 6–7 | checker + lint as gates | `ruff` under the stated set: **138 findings** in the unstructured arm, **0** in the enforced one |
| 8 | complexity budget | the budget is what makes rule 3 enforceable instead of aspirational: it fails on exactly the 4 functions above |
| 9 | scoped dead-code gate | the recorded wiring defect (`is_feasible_pair` written, documented, never called) passed `ty`, `ruff` and `semgrep`; a dead-code gate is the only thing that sees it — and only at default confidence with tests excluded |
| 10 | drill | a gate nobody has seen fail is a gate that can be configured into a no-op; measured, `--min-confidence 80` silences the entire dead-code gate |

**The style question, answered by measurement rather than taste.** This standard is *object-oriented
with a functional core* — frozen entities plus pure functions over them — because that is what the
arms that scored best actually did. Enforced *functional* style throughout (a separate arm) cost
**+42% lines over the stdlib arm for the same guarantees and prevented zero defects**, so purity is
not what pays. What pays is the domain model and the gates. The one thing the functional arm did
better is worth copying anyway: its 31 frozen dataclasses are the cleanest data shapes in the
repository.

**Not in this standard, deliberately:** dependency injection, repository/unit-of-work layers, service
classes, and any pattern whose cost is a class per verb. None of them appeared in the arms that
scored best, and a standard that cannot be violated cheaply is not followed by agents — it is
paraphrased by them.
