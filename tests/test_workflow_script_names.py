"""Guard against undefined names in the Snakemake workflow scripts.

These scripts are the one part of the codebase no test executes end to end: they need a
``snakemake`` object, real WAR files and a plotting backend, so the suite imports them at
most as modules. Import succeeds whether or not a function body references a name that was
never bound, and the failure then surfaces only during a pipeline run, after the expensive
stages have already completed.

A real linter would cover this (pyflakes F821), but none is installed and pre-commit runs
no lint hook, so this test stands in for one over the directory where the gap actually
bites. It is deliberately narrow: one rule, one directory, no dependencies.
"""

import ast
import builtins
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).parent.parent / "workflow" / "scripts"
SCRIPTS = sorted(SCRIPT_DIR.glob("*.py"))

# Injected by Snakemake into the script's globals at run time; never bound in the source.
SNAKEMAKE_GLOBALS = {"snakemake"}


def _module_bindings(tree):
    """Every name bound at module level: imports, assignments, defs, classes."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_target_names(target))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            names.update(_target_names(node.target))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Try, ast.If, ast.With)):
            # Conditional imports and try/except ImportError blocks are common here.
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        names.add(alias.asname or alias.name.split(".")[0])
                elif isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        names.update(_target_names(target))
    return names


def _target_names(target):
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        out = set()
        for element in target.elts:
            out |= _target_names(element)
        return out
    return set()


def _bound_within(node):
    """Names bound anywhere inside a function: params, assignments, imports, nested defs.

    Intentionally over-approximates. A name bound on only one branch still counts as bound,
    because the goal is catching names bound on NO branch, not flow analysis.
    """
    names = set()
    args = node.args
    for group in (args.posonlyargs, args.args, args.kwonlyargs):
        names.update(a.arg for a in group)
    for extra in (args.vararg, args.kwarg):
        if extra:
            names.add(extra.arg)

    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            names.add(sub.id)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            for alias in sub.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(sub.name)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            names.add(sub.name)
        elif isinstance(sub, (ast.Global, ast.Nonlocal)):
            names.update(sub.names)
        elif isinstance(sub, ast.arg):
            names.add(sub.arg)
    return names


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_undefined_names_in_functions(script):
    """Every name a function body loads must be bound somewhere it can see.

    The bug this exists for: three plot calls in ``generate_ep_figures.create_ep_plots``
    read ``plot_order``, which was a local of ``main()``. Import-time was fine and the
    whole suite stayed green, but every EP figure would have raised ``NameError`` on the
    first plot of a real run.
    """
    tree = ast.parse(script.read_text())
    module_level = _module_bindings(tree) | set(dir(builtins)) | SNAKEMAKE_GLOBALS

    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        visible = module_level | _bound_within(node)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if sub.id not in visible:
                    problems.append(f"{script.name}:{sub.lineno} {node.name}() uses undefined {sub.id!r}")

    assert not problems, "undefined name(s):\n  " + "\n  ".join(sorted(set(problems)))
