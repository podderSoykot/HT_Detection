"""
netlist_parser.py: Robust Gate-Level Verilog Netlist Parser for Hardware Trojan Analysis.

Extracts module hierarchy, primary inputs/outputs, logic gates, flip-flops,
standard cells, and continuous assignments from Verilog netlists.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class GateInstance:
    """Represents a logic gate or standard cell instance."""
    inst_name: str
    gate_type: str                  # normalized gate type, e.g., 'AND', 'OR', 'XOR', 'NOT', 'DFF', etc.
    raw_type: str                   # original cell type, e.g., 'AND2_X1', 'sky130_fd_sc_hd__and2_1'
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    is_sequential: bool = False
    is_trojan: bool = False         # ground-truth label for Trojan localization


@dataclass
class CircuitNetlist:
    """Structured representation of a parsed gate-level netlist."""
    module_name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    wires: Set[str] = field(default_factory=set)
    gates: List[GateInstance] = field(default_factory=list)
    file_path: Optional[str] = None


class NetlistParser:
    """Parses structural Verilog netlists into structured GateInstance objects."""

    # Standard Verilog primitives
    PRIMITIVE_MAP = {
        'and': ('AND', False),
        'nand': ('NAND', False),
        'or': ('OR', False),
        'nor': ('NOR', False),
        'xor': ('XOR', False),
        'xnor': ('XNOR', False),
        'not': ('NOT', False),
        'buf': ('BUF', False),
        'dff': ('DFF', True),
        'latch': ('LATCH', True),
    }

    def __init__(self):
        pass

    @staticmethod
    def _strip_comments(code: str) -> str:
        """Removes single-line and multi-line comments."""
        # Multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Single-line comments
        code = re.sub(r'//.*', '', code)
        return code

    @staticmethod
    def _clean_token(token: str) -> str:
        """Cleans escaped identifiers and whitespace."""
        token = token.strip()
        if token.startswith('\\'):
            token = token[1:].strip()
        return token

    def _normalize_cell_type(self, cell_type: str) -> Tuple[str, bool]:
        """Maps standard cell / ASIC library names to common primitive gate types."""
        lower = cell_type.lower()
        if lower in self.PRIMITIVE_MAP:
            return self.PRIMITIVE_MAP[lower]

        # Sequential elements
        if any(seq in lower for seq in ['dff', 'flop', 'reg', 'sdff', 'edff', 'latch']):
            return ('DFF', True)

        # Combinational standard cells
        if 'xnor' in lower:
            return ('XNOR', False)
        if 'xor' in lower:
            return ('XOR', False)
        if 'nand' in lower:
            return ('NAND', False)
        if 'nor' in lower:
            return ('NOR', False)
        if 'and' in lower:
            return ('AND', False)
        if 'or' in lower:
            return ('OR', False)
        if any(inv in lower for inv in ['not', 'inv']):
            return ('NOT', False)
        if any(buf in lower for buf in ['buf', 'clknbuf']):
            return ('BUF', False)
        if 'mux' in lower:
            return ('MUX', False)
        if 'aoi' in lower:
            return ('AOI', False)
        if 'oai' in lower:
            return ('OAI', False)

        return (cell_type.upper(), False)

    def parse_file(self, file_path: str) -> CircuitNetlist:
        """Parses a Verilog netlist file."""
        path = Path(file_path)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        netlist = self.parse_string(content)
        netlist.file_path = str(path.resolve())
        return netlist

    def parse_string(self, content: str) -> CircuitNetlist:
        """Parses Verilog code from string."""
        cleaned = self._strip_comments(content)

        # Extract module header
        module_match = re.search(r'module\s+([a-zA-Z0-9_\\]+)\s*\((.*?)\);', cleaned, re.DOTALL)
        if not module_match:
            module_name = "unknown"
            port_list = []
        else:
            module_name = self._clean_token(module_match.group(1))
            raw_ports = module_match.group(2)
            port_list = [self._clean_token(p) for p in raw_ports.split(',') if self._clean_token(p)]

        netlist = CircuitNetlist(module_name=module_name)

        # Parse inputs, outputs, wires
        for match in re.finditer(r'\b(input|output|inout|wire|reg)\s+(?:\[.*?\]\s+)?([^;]+);', cleaned):
            kind = match.group(1)
            raw_names = match.group(2)
            names = [self._clean_token(n) for n in raw_names.split(',') if self._clean_token(n)]
            if kind in ('input', 'inout'):
                netlist.inputs.extend(names)
            elif kind == 'output':
                netlist.outputs.extend(names)
            elif kind in ('wire', 'reg'):
                netlist.wires.update(names)

        # Ensure unique inputs / outputs while preserving order
        netlist.inputs = list(dict.fromkeys(netlist.inputs))
        netlist.outputs = list(dict.fromkeys(netlist.outputs))

        # Split remaining body by statements ending in semicolon
        # Remove module/endmodule lines
        body = re.sub(r'module\s+.*?;', '', cleaned, flags=re.DOTALL)
        body = re.sub(r'\bendmodule\b', '', body)

        statements = [stmt.strip() for stmt in body.split(';') if stmt.strip()]

        gate_counter = 0

        for stmt in statements:
            # Skip port and wire declaration statements
            if re.match(r'^(input|output|inout|wire|reg)\b', stmt):
                continue

            # Case 1: Continuous assignment: assign out = expression;
            assign_match = re.match(r'^assign\s+([^\s=]+)\s*=\s*(.+)$', stmt)
            if assign_match:
                out_wire = self._clean_token(assign_match.group(1))
                expr = assign_match.group(2).strip()
                gate_counter += 1
                inst_name = f"assign_gate_{gate_counter}"
                self._parse_assign_expression(inst_name, out_wire, expr, netlist)
                continue

            # Case 2: Named port mapping: CELL_TYPE inst_name (.PIN(wire), ...);
            named_inst = re.match(r'^([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_\\]+)\s*\((.*)\)$', stmt, re.DOTALL)
            if named_inst and named_inst.group(1).lower() not in ('module', 'always', 'initial'):
                raw_cell = named_inst.group(1)
                inst_name = self._clean_token(named_inst.group(2))
                port_mappings_str = named_inst.group(3)

                ports = re.findall(r'\.([a-zA-Z0-9_]+)\s*\(\s*([^\)]+)\s*\)', port_mappings_str)
                if ports:
                    inputs = []
                    outputs = []
                    norm_type, is_seq = self._normalize_cell_type(raw_cell)

                    for pin, wire_name in ports:
                        wire_name = self._clean_token(wire_name)
                        pin_upper = pin.upper()
                        # Common output pin designations: Y, Z, Q, QN, OUT, ZN
                        if pin_upper in ('Y', 'Z', 'Q', 'QN', 'OUT', 'ZN'):
                            outputs.append(wire_name)
                        else:
                            inputs.append(wire_name)

                    # If no obvious output pin found, use standard assumption (last pin is out or Q)
                    if not outputs and inputs:
                        outputs.append(inputs.pop())

                    # Check for Trojan annotations in instance or cell name
                    is_tr = any(t in inst_name.lower() or t in raw_cell.lower() for t in ['trojan', 'troj', 'tj', 'tsc', 'tr_'])

                    netlist.gates.append(GateInstance(
                        inst_name=inst_name,
                        gate_type=norm_type,
                        raw_type=raw_cell,
                        inputs=inputs,
                        outputs=outputs,
                        is_sequential=is_seq,
                        is_trojan=is_tr
                    ))
                    continue

            # Case 3: Positional primitive instantiation: gate_type [inst_name] (out, in1, in2, ...);
            pos_inst = re.match(r'^([a-zA-Z0-9_]+)(?:\s+([a-zA-Z0-9_\\]+))?\s*\((.*)\)$', stmt, re.DOTALL)
            if pos_inst:
                raw_type = pos_inst.group(1)
                inst_name = pos_inst.group(2)
                raw_args = pos_inst.group(3)

                norm_type, is_seq = self._normalize_cell_type(raw_type)
                gate_counter += 1
                if not inst_name:
                    inst_name = f"{norm_type.lower()}_{gate_counter}"
                else:
                    inst_name = self._clean_token(inst_name)

                args = [self._clean_token(a) for a in raw_args.split(',') if self._clean_token(a)]
                if args:
                    # In standard Verilog primitives, first argument is the output
                    out_wire = args[0]
                    in_wires = args[1:]
                    is_tr = any(t in inst_name.lower() or t in raw_type.lower() for t in ['trojan', 'troj', 'tj', 'tsc', 'tr_'])

                    netlist.gates.append(GateInstance(
                        inst_name=inst_name,
                        gate_type=norm_type,
                        raw_type=raw_type,
                        inputs=in_wires,
                        outputs=[out_wire],
                        is_sequential=is_seq,
                        is_trojan=is_tr
                    ))

        return netlist

    def _parse_assign_expression(self, inst_name: str, out_wire: str, expr: str, netlist: CircuitNetlist):
        """Decomposes boolean assign expression into gate primitives."""
        # Simple inversion: ~wire
        if expr.startswith('~') and not any(op in expr[1:] for op in ['&', '|', '^']):
            in_wire = self._clean_token(expr[1:])
            netlist.gates.append(GateInstance(
                inst_name=inst_name,
                gate_type='NOT',
                raw_type='assign_not',
                inputs=[in_wire],
                outputs=[out_wire],
                is_sequential=False
            ))
            return

        # Binary operations: a & b, a | b, a ^ b
        op_map = {
            '&': 'AND',
            '|': 'OR',
            '^': 'XOR',
        }
        for op, gate_type in op_map.items():
            if op in expr:
                parts = expr.split(op)
                inputs = []
                for p in parts:
                    token = self._clean_token(p)
                    # If negated operand like ~wire, we can track as input
                    if token.startswith('~'):
                        token = self._clean_token(token[1:])
                    inputs.append(token)

                netlist.gates.append(GateInstance(
                    inst_name=inst_name,
                    gate_type=gate_type,
                    raw_type=f'assign_{gate_type.lower()}',
                    inputs=inputs,
                    outputs=[out_wire],
                    is_sequential=False
                ))
                return

        # Direct assignment: out = in (Buffer)
        in_wire = self._clean_token(expr)
        netlist.gates.append(GateInstance(
            inst_name=inst_name,
            gate_type='BUF',
            raw_type='assign_buf',
            inputs=[in_wire],
            outputs=[out_wire],
            is_sequential=False
        ))
