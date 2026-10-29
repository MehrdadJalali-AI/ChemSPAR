"""The package's algorithm functions must be the paper's functions (paper/src/chemspar_v2 = tag paper-v1).

Syntax trees are compared after removing docstrings, type annotations and the function names, so only renames
of the function itself and typing are allowed; any change in logic, constants or order of operations fails.
"""
import ast
import inspect
import textwrap

import pytest
from conftest import ROOT

PAPER = ROOT / "paper" / "src" / "chemspar_v2"
pytestmark = pytest.mark.skipif(not PAPER.exists(), reason="paper code not present")

# (package module, function) -> (paper file, function)
PAIRS = {
    ("chemspar._keys", "canonical"): ("periodic.py", "canonical"),
    ("chemspar.graph", "build_instance_graph"): ("periodic.py", "build_instance_graph"),
    ("chemspar.taxonomy", "crystalnn_instances"): ("periodic.py", "crystalnn_instances"),
    ("chemspar.taxonomy", "annotate_instances"): ("periodic.py", "annotate_instances"),
    ("chemspar.taxonomy", "radius_bonded"): ("bonding.py", "radius_bonded"),
    ("chemspar.taxonomy", "classify_edge"): ("bonding.py", "classify_edge"),
    ("chemspar.taxonomy", "is_metal"): ("elements.py", "is_metal"),
    ("chemspar.taxonomy", "role"): ("elements.py", "role"),
    ("chemspar.taxonomy", "crystalnn_pairs"): ("crystalnn.py", "crystalnn_pairs"),
    ("chemspar.orderings", "instance_gravity"): ("periodic.py", "instance_gravity"),
    ("chemspar.orderings", "order_by"): ("periodic.py", "order_by"),
    ("chemspar.orderings", "bhs_node_gravity"): ("views.py", "_bhs_node_gravity"),
    ("chemspar.gate", "_check"): ("periodic.py", "_check"),
    ("chemspar.gate", "prune_instances"): ("periodic.py", "prune_instances"),
}
# Allowed, value-preserving substitutions in the package source (package -> paper):
#   DEFAULT_CUTOFF (= 4.5)  -> 4.5
#   bhs_weights()           -> BHS_WEIGHTS  (same file, bundled copy; its hash is tested in test_api.py)
# order_random differs only in the name of its third parameter (qmof_id -> structure_id); checked separately.
# export.instances_to_data is a deliberately generalised copy of train.instances_to_data (optional y, plain Data);
# it is checked by the PyG round-trip test instead.
SUBSTITUTIONS = (("DEFAULT_CUTOFF", "4.5"), ("bhs_weights()", "BHS_WEIGHTS"))


class _Strip(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.name, node.returns, node.decorator_list = "f", None, []
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            node.body = node.body[1:] or [ast.Pass()]
        return node

    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_AnnAssign(self, node):  # `x: T = v` -> `x = v`
        return ast.Assign(targets=[node.target], value=node.value, lineno=node.lineno)


def _norm(src: str) -> str:
    tree = _Strip().visit(ast.parse(textwrap.dedent(src)))
    # `a: T = []; b: U = []` (package) vs `a, b = [], []` (paper): compare after expanding tuple assignments
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            body = []
            for st in n.body:
                if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Tuple) and isinstance(st.value, ast.Tuple):
                    body += [ast.Assign(targets=[t], value=v, lineno=0) for t, v in zip(st.targets[0].elts, st.value.elts)]
                else:
                    body.append(st)
            n.body = body
            out.append(ast.dump(n, annotate_fields=False))
            break
    return out[0]


def _paper_source(fname: str, func: str) -> str:
    tree = ast.parse((PAPER / fname).resolve().read_text())
    src = (PAPER / fname).resolve().read_text().splitlines()
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == func:
            return "\n".join(src[n.lineno - 1 - len(n.decorator_list):n.end_lineno])
    raise LookupError(func)


@pytest.mark.parametrize("pkg,paper", PAIRS.items(), ids=lambda x: ".".join(x) if isinstance(x, tuple) else x)
def test_function_identical_to_paper(pkg, paper):
    import importlib
    mod, name = pkg
    obj = getattr(importlib.import_module(mod), name)
    obj = getattr(obj, "__wrapped__", obj)
    src = inspect.getsource(obj)
    for a, b in SUBSTITUTIONS:
        src = src.replace(a, b)
    assert _norm(src) == _norm(_paper_source(*paper))


def test_order_random_identical_up_to_parameter_name():
    from chemspar.orderings import order_random
    a = _norm(inspect.getsource(order_random).replace("structure_id", "qmof_id"))
    assert a == _norm(_paper_source("periodic.py", "order_random"))
