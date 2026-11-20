# Experiment records

The training console records cover steps 1 through 60. VeRL restores the preceding checkpoint at each segment boundary. Each segment includes the resolved configuration and the console output.

The held-out files record the same 30 AIME24 problems at six checkpoints, with eight answers per problem and a 32,768-token response limit. Each file contains the generated answers, how generation ended, and the grader's decisions.

Machine-specific artifact paths in the records were shortened. The run settings, answers, grades and console metrics are preserved.
