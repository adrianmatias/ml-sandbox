package ledger

@main def ledger(args: String*): Unit =
  val result = Cli.run(args.toList)
  if result.stdout.nonEmpty then System.out.print(result.stdout)
  if result.stderr.nonEmpty then System.err.print(result.stderr)
  System.out.flush()
  System.err.flush()
  sys.exit(result.exitCode)
