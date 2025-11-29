C7 Core — Minimal Architecture Package

This package contains the core abstractions extracted from the C7
design phases (phases/ folder). It does NOT include modality-specific
implementations such as AudioBrain. Those are placed under c7_bindings/.

Modules provided in c7_core:
- GRSCState        : Internal G–R–S–C vector representation
- Observer         : Modality-agnostic observation interface
- DecoherenceEngine: State update dynamics
- MemoryBuffer     : Short-term transition memory
- Config           : Central configuration values

Version: 0.1.0
