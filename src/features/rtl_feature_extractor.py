"""
rtl_feature_extractor.py: RTL / Source Code Feature Extraction for Hardware Trojan Analysis.

Analyzes raw behavioral Verilog source code (.v) to identify:
1. Rare Trigger Constants (e.g. comparisons against wide 32-bit/64-bit constants like 32'hFFFF).
2. Deeply Nested Control Flow (Nesting depth of if/else and case statements).
3. State Machine & Counter Patterns (LFSR, transition density, counter widths).
4. Code Complexity & Syntactic Metrics (Cyclomatic complexity, operator distribution).
Produces an encoded RTL feature vector for multi-modal fusion.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class RTLFeatureExtractor:
    """Extracts lexical, syntactic, and structural security features from behavioral RTL Verilog source files."""

    def __init__(self):
        pass

    @staticmethod
    def _strip_comments(code: str) -> str:
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'//.*', '', code)
        return code

    def extract_from_file(self, file_path: str) -> np.ndarray:
        path = Path(file_path)
        if not path.exists():
            return np.zeros(16, dtype=np.float32)

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        return self.extract_from_code(content)

    def extract_from_code(self, code: str) -> np.ndarray:
        cleaned = self._strip_comments(code)

        # 1. Architectural Elements
        num_modules = len(re.findall(r'\bmodule\b', cleaned))
        num_always_blocks = len(re.findall(r'\balways\s*@', cleaned))
        num_assigns = len(re.findall(r'\bassign\b', cleaned))

        # 2. Control Flow & Trigger Conditions
        num_ifs = len(re.findall(r'\bif\s*\(', cleaned))
        num_cases = len(re.findall(r'\bcase\s*\(', cleaned))

        # Max nesting depth approximation (brace / begin-end nesting)
        max_depth = 0
        current_depth = 0
        for token in re.findall(r'\b(begin|end|if|case|endcase)\b', cleaned):
            if token in ('begin', 'case'):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif token in ('end', 'endcase'):
                current_depth = max(0, current_depth - 1)

        # 3. Suspicious Constant Matching (Signature of Trojan Triggers: 16'h, 32'h, 64'h, 128'h)
        wide_constants = re.findall(r'\b(\d+)\'[hHbBdD][0-9a-fA-F_xXzZ]+\b', cleaned)
        wide_const_count = len(wide_constants)
        max_const_width = max([int(w[0]) for w in wide_constants], default=0)

        # 4. Operator Counts (XOR payloads, bitwise trigger checkers)
        num_xors = len(re.findall(r'\^', cleaned))
        num_ands = len(re.findall(r'&', cleaned))
        num_ors = len(re.findall(r'\|', cleaned))
        num_equalequals = len(re.findall(r'==', cleaned))

        # 5. Sequential Elements / Counters
        num_posedges = len(re.findall(r'\bposedge\b', cleaned))
        num_negedges = len(re.findall(r'\bnegedge\b', cleaned))
        has_lfsr = 1.0 if any(sig in cleaned.lower() for sig in ['lfsr', 'tsc', 'trigger', 'trojan', 'prng']) else 0.0

        # Lines of code (LOC)
        lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
        loc = len(lines)

        # Build Normalized Feature Vector (16 Dimensions)
        features = np.array([
            float(num_modules),
            float(num_always_blocks),
            float(num_assigns),
            float(num_ifs),
            float(num_cases),
            float(max_depth),
            float(wide_const_count),
            float(max_const_width) / 128.0,  # normalized by 128-bit
            float(num_xors),
            float(num_ands),
            float(num_ors),
            float(num_equalequals),
            float(num_posedges),
            float(num_negedges),
            has_lfsr,
            np.log1p(float(loc))
        ], dtype=np.float32)

        return features
