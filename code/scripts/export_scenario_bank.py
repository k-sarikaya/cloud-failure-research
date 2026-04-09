from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running as a script from arbitrary working directories.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.edge_env_v2 import generate_paired_scenarios


def adj_to_edgelist(adj: list[list[int]]) -> np.ndarray:
    edges = []
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            if i < j:
                edges.append((i, j))
    return np.asarray(edges, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-scenarios', type=int, default=20)
    ap.add_argument('--n-nodes', type=int, default=100)
    ap.add_argument('--topo-type', type=str, default='ba')
    ap.add_argument('--ba-m', type=int, default=3)
    ap.add_argument('--max-steps', type=int, default=1440)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    pkg_root = Path(__file__).resolve().parents[2]
    art = pkg_root / 'artifacts'
    art.mkdir(exist_ok=True)

    out = Path(args.out) if args.out else art / f"scenario_bank_{args.topo_type}_n{args.n_nodes}_k{args.num_scenarios}_seed{args.seed}.npz"
    manifest = out.with_suffix('.json')

    scenarios = generate_paired_scenarios(
        seed=args.seed,
        num_scenarios=args.num_scenarios,
        n_nodes=args.n_nodes,
        topo_type=args.topo_type,
        ba_m=args.ba_m,
        max_steps=args.max_steps,
        capacity_range=(50.0, 500.0),
        init_util_range=(0.75, 0.90),
    )

    edge_offsets = [0]
    edges_all: list[np.ndarray] = []
    caps = []
    loads = []
    uniforms = []
    for sc in scenarios:
        edges = adj_to_edgelist(sc.adj)
        edges_all.append(edges)
        edge_offsets.append(edge_offsets[-1] + edges.shape[0])
        caps.append(sc.capacities)
        loads.append(sc.loads)
        uniforms.append(sc.fail_uniforms)

    edges_concat = np.concatenate(edges_all, axis=0) if edges_all else np.zeros((0, 2), dtype=np.int32)
    edge_offsets = np.asarray(edge_offsets, dtype=np.int64)
    caps = np.stack(caps, axis=0).astype(np.float32)
    loads = np.stack(loads, axis=0).astype(np.float32)
    uniforms = np.stack(uniforms, axis=0).astype(np.float32)

    np.savez_compressed(
        out,
        edges=edges_concat,
        edge_offsets=edge_offsets,
        capacities=caps,
        loads=loads,
        fail_uniforms=uniforms,
    )

    manifest.write_text(
        json.dumps(
            {
                'seed': int(args.seed),
                'num_scenarios': int(args.num_scenarios),
                'n_nodes': int(args.n_nodes),
                'topo_type': str(args.topo_type),
                'ba_m': int(args.ba_m),
                'max_steps': int(args.max_steps),
                'npz': str(out),
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    print(f"[ok] wrote {out}")
    print(f"[ok] wrote {manifest}")


if __name__ == '__main__':
    main()
