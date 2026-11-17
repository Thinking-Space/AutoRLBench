#!/usr/bin/env bash
# Stop multiple eval jobs — defective version.
# BUG: cctl job stop without --yes requires interactive confirmation.
# In a non-TTY shell, stdin is empty; the prompt reads nothing and aborts.
# All stop commands silently fail: jobs remain Running.

set -ex

JOB_IDS="${JOB_IDS:-tasks/1637987 tasks/1637996 tasks/1638055}"

for jid in $JOB_IDS; do
  echo "Stopping job: $jid"
  # BUG: missing --yes flag
  # This will read empty stdin, print "stop task aborted; confirmation did not match",
  # and return ok:false. The job is NOT stopped.
  tools/cctl.sh job stop "$jid"  # <-- fix: add --yes
done
