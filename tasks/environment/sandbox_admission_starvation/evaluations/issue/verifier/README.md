# Issue verifier

The issue tier reads `$AUTORLBENCH_RESULTS/agent_answer.md` and compares it with the hidden `oracle.md` using verdict / root-cause / evidence weights 0.2 / 0.6 / 0.2. The pass mark is 0.8. The verifier does not treat a training reward, a single GPU metric, or a raw environment error as a root cause without a code path.
