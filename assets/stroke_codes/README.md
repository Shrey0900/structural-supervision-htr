## 32-bit stroke-direction codes (S=8)

Each character is assigned a manually designed canonical stroke-direction sequence.
We encode at most S=8 steps. Each step is one of {U,D,L,R} and maps to a 4-way one-hot.
Concatenating 8 steps gives 32 bits. Missing steps are padded with zeros.

File: alphabet_binary_32.csv
Columns:
- alphabet: single character (case-sensitive)
- bits: 32-bit string (0/1)
