#!/usr/bin/env python3
"""Run a tree's workflow/scripts/generate_wars.py for one animal, without Snakemake.

For before and after comparisons of WAR generation on real data, outside results/. The script
gets the params, output, log, threads and resources the war_generation rule would give it,
built as workflow/Snakefile builds them: the config from load_dataset_config, the samples from
expand_animals_config, the animal's (folder, id, key) entry and channel subset, and the rule's
config param (analysis_config(config) where the tree has it, else the whole config, as before
PR 2). The temp directory moves under --out, so no zarr cache lands in the configured one.

    PYTHONPATH=TREE/src python scripts/diagnostics/generate_war_without_snakemake.py \\
        --tree TREE --dataset iqsec2 --animal IQSEC2_80 --out DIR --threads 4 --mem-mb 80000

Writes DIR/wars/<animal>/war.parquet and war.json, DIR/fdsars/<animal>/, and the logs under
DIR/logs/war_generation/. Compare two runs with scripts/compare_wars.py.
"""

import argparse
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace


class _Named(list):
    """A list whose items are also attributes, as Snakemake's log and wildcards objects are."""

    def __init__(self, **items):
        super().__init__(items.values())
        self._names = list(items)
        for name, value in items.items():
            setattr(self, name, value)

    def items(self):
        return [(name, getattr(self, name)) for name in self._names]

    def keys(self):
        return list(self._names)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", required=True, type=Path, help="the worktree whose code and configs run")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--animal", required=True, help="the animal id as the samples config writes it")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mem-mb", type=int, default=80000)
    args = ap.parse_args(argv)

    tree, out = args.tree.resolve(), args.out.resolve()
    os.chdir(tree)  # dataset configs and extract_func paths are relative to the tree

    import neurodent
    from neurodent.core.utils import slugify
    from neurodent.workflow.utils import expand_animals_config, load_dataset_config, resolve_samples_config

    if not Path(neurodent.__file__).resolve().is_relative_to(tree):
        raise SystemExit(f"neurodent is imported from {neurodent.__file__}, not {tree}; set PYTHONPATH")
    try:
        from neurodent.workflow.utils import analysis_config
    except ImportError:  # before PR 2, every rule passed the whole config
        analysis_config = None

    config = load_dataset_config(args.dataset)
    config["temp_directory"] = str(out / "tmp")
    samples = expand_animals_config(resolve_samples_config(config))
    if args.animal not in [entry["id"] for entry in samples["animals"]]:
        raise SystemExit(f"{args.animal} is not an animal of {args.dataset}")
    slug = slugify(args.animal, allow_unicode=True)
    subset = samples.get("_animal_channel_subsets", {}).get(args.animal)
    for folder in (out / "tmp", out / "wars" / slug, out / "logs" / "war_generation"):
        folder.mkdir(parents=True, exist_ok=True)

    snakemake = SimpleNamespace(
        params=SimpleNamespace(
            animal_folders=[("", args.animal, "")],
            animal_id=args.animal,
            is_split_recording=subset is not None,
            channel_subset=subset,
            config=analysis_config(config) if analysis_config else config,
            samples_config=samples,
        ),
        output=SimpleNamespace(
            war_parquet=str(out / "wars" / slug / "war.parquet"),
            war_json=str(out / "wars" / slug / "war.json"),
            fdsar_dir=str(out / "fdsars" / slug),
        ),
        log=_Named(
            stdout=str(out / "logs" / "war_generation" / f"{slug}.out"),
            stderr=str(out / "logs" / "war_generation" / f"{slug}.err"),
        ),
        threads=args.threads,
        resources=SimpleNamespace(mem_mb=args.mem_mb, attempt=1, time="12h", nodes=1),
        config=config,
        rule="war_generation",
        wildcards=_Named(animal=slug),
    )
    spec = importlib.util.spec_from_file_location("generate_wars", tree / "workflow" / "scripts" / "generate_wars.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # its main() runs only under __main__
    module.snakemake = snakemake
    module.main()


if __name__ == "__main__":
    main()
