---
description: Show timegraph memory stats for this project (episode/fact counts, last ingest)
allowed-tools: Bash(timegraph:*)
---

Run the timegraph stats subcommand and report the output verbatim to the user:

!`timegraph stats`

If the command fails because backends are down, surface the error and remind the user to run `timegraph init`.
