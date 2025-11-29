"""
Global configuration for the C7 Core v0.1.0
"""

# Number of past steps kept in memory buffer
MEMORY_BUFFER_SIZE = 32

# Default initial GRSC values (all in [0, 1])
INITIAL_G = 0.5
INITIAL_R = 0.5
INITIAL_S = 0.5
INITIAL_C = 0.5

# Simple learning / update rates used by the DecoherenceEngine
G_UPDATE_RATE = 0.10
R_UPDATE_RATE = 0.08
S_UPDATE_RATE = 0.06
C_UPDATE_RATE = 0.12
