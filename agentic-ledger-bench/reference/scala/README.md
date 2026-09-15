# Scala 3 implementation

Scala 3.9.0 LTS, built with `scala-cli`. No runtime dependencies (the JSON writer and
reader are hand-written, which is itself one of the findings — see `docs/FINDINGS.md` §5).

## Running

```bash
cd reference/scala
scala-cli run . --quiet -- baseline ../../benchmark/data/statement.csv
scala-cli run . --quiet -- accounts ../../benchmark/data/accounts.json
scala-cli test .
```

`project.scala` pins the version and the single test-only dependency:

```scala
//> using scala 3.9.0
//> using test.dep org.scalameta::munit::1.3.6
//> using options -deprecation -feature
```

## Note on the unused-code flag

The interesting gap in this arm is that its build **never enabled the dead-code
warning**, so an unused private member would have compiled silently. To see the
difference, run:

```bash
scala-cli compile . --scala-option -Wunused:all --scala-option -Xfatal-warnings
```

That flags an unused **private** member as a compile error. It does **not** flag an
unused **public** method — a public-but-uncalled function still compiles clean even
with `-Xfatal-warnings`, which is why Python's `vulture` (not `ruff`) is the actual
counterpart for that defect class. See `docs/FINDINGS.md` §11.

## Sandbox note

If you run this where `$HOME` is read-only, point every cache into a writable
directory, or `coursier` and `scala-cli` fail with `Read-only file system`:

```bash
COURSIER_CACHE=/writable/coursier \
XDG_CACHE_HOME=/writable/xdg-cache \
XDG_DATA_HOME=/writable/xdg-data \
SCALA_CLI_HOME=/writable/scala-cli-home \
  scala-cli run . --quiet -- baseline ../../benchmark/data/statement.csv
```

`COURSIER_CACHE` alone is not enough — `scala-cli` also writes under
`~/.cache/scalacli`, which is what `XDG_CACHE_HOME` redirects.
