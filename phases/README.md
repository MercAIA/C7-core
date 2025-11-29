# C7-core
# C7 Phases – Design & Evolution Log

This folder contains the raw phase notes for the C7 architecture.

- These .md files are not cleaned-up documentation.
- They are the actual step-by-step thinking process:
  - early sketches of GRSC
  - observer–decoherence loops
  - experiments with different update rules
  - paths that were abandoned or merged
  - the road that finally led to the AudioBrain implementation.

Nothing here should be deleted or rewritten retroactively.
New phases should be added as new files, not by changing old ones.

In the rest of this repository:

- src/c7_core/ is the *current* minimal integrated core,
  extracted and distilled from these phases.
- bindings/ contains concrete implementations such as AudioBrain,
  which sit on top of the core (they are not the core itself).

If you want to understand how C7 really emerged, start here.
