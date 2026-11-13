"""Balanced-Partition Edge Dismantling and a standalone plotting demo."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from fractions import Fraction
import heapq
import math
from pathlib import Path
import random

import networkx as nx
import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class Config:
    seeding: str = "random"
    level1_cap: int | None = None
    refine_passes: int | None = None  # None: positive-gain moves to convergence.
    eigen_space_tolerance: float = 1e-10
    eigen_residual_tolerance: float = 1e-8
    spectral_order_tolerance: float = 1e-10


def edge(u, v):
    return (u, v) if u < v else (v, u)


def capacity(gamma, n):
    try:
        value = Decimal(str(gamma))
    except InvalidOperation:
        raise ValueError("gamma must be a number in (0, 1]") from None
    if not value.is_finite() or not 0 < value <= 1:
        raise ValueError("gamma must be finite and in (0, 1]")
    cap = int((value * n).to_integral_value(rounding=ROUND_FLOOR))
    if cap < 1:
        raise ValueError(f"gamma={gamma} gives capacity 0 for N={n}")
    return cap


def _canonical_blocks(blocks):
    return sorted((set(b) for b in blocks if b), key=lambda b: tuple(sorted(b)))


def _owners(blocks):
    return {u: i for i, b in enumerate(blocks) for u in b}


def _check(graph, cap, config):
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("BPED expects an undirected simple graph without self-loops")
    if not graph:
        raise ValueError("BPED expects at least one node")
    if not isinstance(cap, int) or cap < 1:
        raise ValueError("capacity must be a positive integer")
    if config.seeding not in ("fps", "random", "degree"):
        raise ValueError("seeding must be fps, random or degree")
    if config.level1_cap is not None and (
        not isinstance(config.level1_cap, int) or not 1 <= config.level1_cap <= cap
    ):
        raise ValueError("level1_cap must be an integer in [1, capacity]")
    if config.refine_passes is not None and (
        not isinstance(config.refine_passes, int) or config.refine_passes < 0
    ):
        raise ValueError("refine_passes must be nonnegative or None")
    for name in ("eigen_space_tolerance", "eigen_residual_tolerance", "spectral_order_tolerance"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def grow_blocks(graph, limit, seeding="random", seed=0):
    """Grow connected blocks using degree(v) - 2*edges(v,current block).

    By default, each seed is uniform among the remaining nodes.
    FPS distances use the original graph and all earlier seeds. Unreachable
    nodes have infinite distance. The first FPS seed is uniformly random,
    controlled by seed; later seeds maximize nearest distance, with ID ties.
    """
    if limit < 1 or seeding not in ("fps", "random", "degree"):
        raise ValueError("invalid growth limit or seeding rule")
    remaining = set(graph)
    distance = dict.fromkeys(graph, math.inf)
    rng = random.Random(seed)
    blocks = []
    while remaining:
        if seeding == "random":
            start = rng.choice(sorted(remaining))
        elif seeding == "degree":
            start = min(remaining, key=lambda u: (-graph.degree(u), u))
        elif not blocks:
            start = rng.choice(sorted(remaining))
        else:
            start = min(remaining, key=lambda u: (-distance[u], u))
        if seeding == "fps":
            for u, d in nx.single_source_shortest_path_length(graph, start).items():
                distance[u] = min(distance[u], d)
        block = {start}
        remaining.remove(start)
        inside = Counter()
        frontier = []

        def expose(u):
            for v in sorted(graph.neighbors(u)):
                if v in remaining:
                    inside[v] += 1
                    heapq.heappush(frontier, (graph.degree(v) - 2 * inside[v], v, inside[v]))

        expose(start)
        while len(block) < limit and frontier:
            _, u, count = heapq.heappop(frontier)
            if u not in remaining or inside[u] != count:
                continue
            remaining.remove(u)
            block.add(u)
            expose(u)
        blocks.append(block)
    return _canonical_blocks(blocks)


def merge_blocks(graph, blocks, cap):
    """Dynamically merge the heaviest FEASIBLE seam; skip oversized pairs."""
    groups = {min(b): set(b) for b in blocks}
    owner = {u: min(b) for b in blocks for u in b}
    weights = Counter(edge(owner[u], owner[v]) for u, v in graph.edges() if owner[u] != owner[v])
    seams = {b: {} for b in groups}
    heap = []
    for (a, b), w in weights.items():
        seams[a][b] = seams[b][a] = w
        heapq.heappush(heap, (-w, a, b))
    while heap:
        negative_w, a, b = heapq.heappop(heap)
        if a not in groups or b not in groups or seams[a].get(b) != -negative_w:
            continue
        if len(groups[a]) + len(groups[b]) > cap:
            continue
        groups[a].update(groups.pop(b))
        seams[a].pop(b)
        for neighbor, w in list(seams[b].items()):
            if neighbor == a:
                continue
            combined = seams[a].get(neighbor, 0) + w
            seams[a][neighbor] = seams[neighbor][a] = combined
            seams[neighbor].pop(b)
        del seams[b]
        for neighbor, w in seams[a].items():
            x, y = edge(a, neighbor)
            heapq.heappush(heap, (-w, x, y))
    return _canonical_blocks(groups.values())


def refine_blocks(graph, blocks, cap, passes=None):
    """Accept only single-node moves that strictly reduce the boundary count."""
    groups = {i: set(b) for i, b in enumerate(blocks)}
    owner = _owners(blocks)
    iteration = 0
    while passes is None or iteration < passes:
        moved = False
        for u in sorted(graph):
            source = owner[u]
            count = Counter(owner[v] for v in graph.neighbors(u))
            targets = [b for b in count if b != source and len(groups[b]) < cap
                       and count[b] > count.get(source, 0)]
            if not targets:
                continue
            target = min(targets, key=lambda b: (-count[b], b))
            groups[source].remove(u)
            groups[target].add(u)
            owner[u] = target
            moved = True
        iteration += 1
        if not moved:
            break
    return _canonical_blocks(groups.values())


def split_connected(graph, blocks):
    """Split induced disconnected blocks without changing the boundary set."""
    return _canonical_blocks(piece for b in blocks for piece in nx.connected_components(graph.subgraph(b)))


def partition(graph, cap, config=None, seed=0):
    config = config or Config()
    _check(graph, cap, config)
    components = _canonical_blocks(nx.connected_components(graph))
    if max(map(len, components)) <= cap:
        return components
    limit = max(1, cap // 2) if config.level1_cap is None else config.level1_cap
    blocks = grow_blocks(graph, limit, config.seeding, seed)
    blocks = merge_blocks(graph, blocks, cap)
    blocks = refine_blocks(graph, blocks, cap, config.refine_passes)
    blocks = split_connected(graph, blocks)
    # Refinement and splitting can create new capacity-feasible seams.
    # A final dynamic merge preserves connectivity and exhausts those seams.
    blocks = merge_blocks(graph, blocks, cap)
    if sum(map(len, blocks)) != len(graph) or set(_owners(blocks)) != set(graph):
        raise RuntimeError("final blocks do not partition the input nodes")
    if any(len(b) > cap for b in blocks):
        raise RuntimeError("final block exceeds capacity")
    return blocks


def quotient_graph(graph, blocks):
    owner = _owners(blocks)
    quotient = nx.Graph()
    quotient.add_nodes_from(range(len(blocks)))
    mass = {i: len(b) for i, b in enumerate(blocks)}
    bundles = defaultdict(list)
    for u, v in sorted(edge(u, v) for u, v in graph.edges()):
        a, b = owner[u], owner[v]
        if a != b:
            bundles[edge(a, b)].append((u, v))
    for (a, b), originals in sorted(bundles.items()):
        quotient.add_edge(a, b, weight=len(originals))
    return quotient, mass, dict(bundles)


def _fiedler_order(graph, mass, config):
    """Mass-weighted Fiedler ordering, with deterministic repeated-eigenvalue ties."""
    nodes = sorted(graph)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="weight", dtype=float)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    m = np.array([mass[u] for u in nodes], dtype=float)
    values, vectors = eigh(laplacian, np.diag(m), type=1, driver="gvd", check_finite=True)
    value = float(values[1])
    if not np.isfinite(values).all() or value < -config.eigen_residual_tolerance:
        raise RuntimeError("invalid generalized spectrum")
    indices = [i for i in range(1, len(nodes))
               if abs(values[i] - value) <= config.eigen_space_tolerance * max(1.0, abs(value))]
    space = np.sqrt(m)[:, None] * vectors[:, indices]
    for anchor in range(len(nodes)):
        projection = space @ space[anchor, :]
        norm = float(np.linalg.norm(projection))
        if norm > config.eigen_space_tolerance:
            vector = projection / (norm * np.sqrt(m))
            break
    else:
        raise RuntimeError("cannot select a nonzero Fiedler vector")
    rayleigh = float(vector @ laplacian @ vector / np.dot(m * vector, vector))
    residual = float(np.linalg.norm(laplacian @ vector - rayleigh * m * vector))
    scale = max(float(np.linalg.norm(laplacian, ord=np.inf) * np.linalg.norm(vector)),
                float(abs(rayleigh) * np.linalg.norm(m * vector)), np.finfo(float).tiny)
    mean_error = float(abs(np.dot(m, vector)) / math.sqrt(float(m.sum())))
    if (not np.isfinite(vector).all() or residual / scale > config.eigen_residual_tolerance
            or mean_error > config.eigen_residual_tolerance):
        raise RuntimeError("Fiedler vector failed residual or weighted-mean validation")
    raw = sorted(zip(nodes, map(float, vector)), key=lambda item: (item[1], item[0]))
    order = []
    start = 0
    while start < len(raw):
        end = start + 1
        while end < len(raw) and raw[end][1] - raw[start][1] <= config.spectral_order_tolerance:
            end += 1
        order.extend(sorted(u for u, _ in raw[start:end]))
        start = end
    return order


def _sweep(graph, mass, order):
    """Minimize cut * (1/left_mass + 1/right_mass) over spectral prefixes.

    Mass counts original nodes, not quotient vertices. Exact ties keep the
    earliest prefix; the minimum is among these candidates, not all cuts.
    """
    total = sum(mass[u] for u in order)
    prefix = set()
    cut = left_mass = 0
    best = None
    for i, u in enumerate(order[:-1], 1):
        for v, data in graph[u].items():
            cut += -data["weight"] if v in prefix else data["weight"]
        prefix.add(u)
        left_mass += mass[u]
        score = Fraction(cut * total, left_mass * (total - left_mass))
        if best is None or score < best[0]:
            best = score, i
    split = best[1]
    return set(order[:split]), set(order[split:])


def spectral_schedule(graph, blocks, config=None):
    """Cut the most massive connected quotient component using RatioCut first.

    Use mass-weighted Fiedler ordering and minimize RatioCut over its prefixes.
    Original edges are emitted seam by seam. Within each cut, seams use
    increasing bundle weight, with lexicographic block-pair ties. Edges within
    each seam use lexicographic order. Each cut side is split into actual
    components.
    """
    config = config or Config()
    quotient, mass, bundles = quotient_graph(graph, blocks)
    queue = []

    def push_components(nodes):
        for component in nx.connected_components(quotient.subgraph(nodes)):
            ids = tuple(sorted(component))
            heapq.heappush(queue, (-sum(mass[u] for u in ids), ids))

    push_components(quotient.nodes)
    result = []
    while queue:
        _, component = heapq.heappop(queue)
        subgraph = quotient.subgraph(component)
        if not subgraph.number_of_edges():
            continue
        order = _fiedler_order(subgraph, mass, config)
        left, right = _sweep(subgraph, mass, order)
        cross = sorted(
            (edge(u, v) for u, v in subgraph.edges() if (u in left) != (v in left)),
            key=lambda pair: (len(bundles[pair]), pair),
        )
        for pair in cross:
            result.extend(bundles[pair])
        push_components(left)
        push_components(right)
    expected = {e for originals in bundles.values() for e in originals}
    if len(result) != len(expected) or set(result) != expected:
        raise RuntimeError("schedule must emit every final boundary edge exactly once")
    return result


def build_order(graph, cap, seed=0, config=None):
    """Generate a native boundary sequence; never mutate the input graph."""
    config = config or Config()
    blocks = partition(graph, cap, config, seed)
    return spectral_schedule(graph, blocks, config)


def lcc_curve(graph, order):
    """Exact LCC(t), including t=0, by reverse union-find; memory only."""
    removed = {edge(u, v) for u, v in order}
    if len(removed) != len(order) or any(not graph.has_edge(*e) for e in removed):
        raise ValueError("removal sequence has duplicate or nonexistent edges")
    parent = {u: u for u in graph}
    size = dict.fromkeys(graph, 1)
    largest = int(bool(graph))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        nonlocal largest
        a, b = find(u), find(v)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
        largest = max(largest, size[a])

    for u, v in graph.edges():
        if edge(u, v) not in removed:
            union(u, v)
    curve = [largest]
    for u, v in reversed(order):
        union(u, v)
        curve.append(largest)
    curve.reverse()
    return curve


def _threshold_step(curve, cap):
    if cap < 1:
        raise ValueError("capacity must be positive")
    tc = next((t for t, largest in enumerate(curve) if largest <= cap), None)
    if tc is None:
        raise ValueError("removal sequence does not reach capacity")
    return tc


def evaluate(graph, order, cap):
    """Return qc and threshold R, measured after each individual edge deletion.

    tc = first t with LCC(t) <= cap; qc = tc/M;
    R = sum(LCC(t), t=1..tc)/(N*M). No t=0 or residual tail contribution.
    """
    tc = _threshold_step(curve := lcc_curve(graph, order), cap)
    n, m = len(graph), graph.number_of_edges()
    if not m:
        return 0.0, 0.0
    return tc / m, sum(curve[1:tc + 1]) / (n * m)


def complete_order(graph, order, cap, config=None):
    """Keep the BPED prefix to the threshold, then bisect residual components."""
    tc = _threshold_step(lcc_curve(graph, order), cap)
    prefix = list(order[:tc])
    residual = graph.copy()
    residual.remove_edges_from(prefix)
    # Singleton blocks let the same spectral schedule operate on original nodes.
    singletons = [{node} for node in sorted(residual)]
    return prefix + spectral_schedule(residual, singletons, config), tc


def load_graph(path):
    """Read an integer edge list and keep the largest connected component."""
    graph = nx.Graph()
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if len(fields) not in (1, 2):
                raise ValueError(f"{path}:{number}: expected one or two integer columns")
            try:
                nodes = [int(field) for field in fields]
            except ValueError:
                raise ValueError(f"{path}:{number}: expected integer node labels") from None
            graph.add_nodes_from(nodes)
            if len(nodes) == 2 and nodes[0] != nodes[1]:
                graph.add_edge(*nodes)
    if not graph.number_of_edges():
        raise ValueError(f"{path}: the demo needs at least one edge")
    largest = max(nx.connected_components(graph), key=lambda nodes: (len(nodes), -min(nodes)))
    return graph.subgraph(largest)


def plot_curve(output, dataset, n, m, gamma, seed, curve, tc):
    """Write the full dismantling curve and its threshold metrics to PDF."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MultipleLocator

    q = np.arange(m + 1) / m
    s = np.asarray(curve) / n
    qc = q[tc]
    r = sum(curve[1:tc + 1]) / (n * m)
    r_full = sum(curve[1:]) / (n * m)
    red = "#B62631"
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.15, top=0.89)
    ax.set_title(dataset, fontsize=12, fontweight="bold")
    ax.step(q, s, where="post", color=red, linewidth=1.45, label="BPED")
    ax.axhline(float(gamma), color="#767676", linestyle="--", linewidth=0.85)
    ax.plot(
        qc, s[tc], marker="o", linestyle="none", markersize=4.6,
        markerfacecolor=red, markeredgecolor="#272727", markeredgewidth=0.55,
        clip_on=False, zorder=4, label=rf"Threshold ($\gamma={gamma}$)",
    )
    ax.set(xlim=(0, 1), ylim=(0, 1.035), xlabel=r"$q$", ylabel=r"$s(q)$")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.tick_params(direction="in", top=True, right=True, labelsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.text(
        0.97, 0.76,
        "\n".join((
            rf"$q_c = {qc:.5f}$",
            rf"Threshold $R = {r:.5f}$",
            rf"$R_{{\mathrm{{full}}}} = {r_full:.5f}$",
            f"Seed {seed}",
        )),
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42}):
        fig.savefig(output, format="pdf")
    plt.close(fig)
    return qc, r, r_full


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edgelist", nargs="?", type=Path,
                        default=here / "power-US-Grid.edgelist",
                        help="integer edge list (default: bundled Power dataset)")
    parser.add_argument("--gamma", default="0.01",
                        help="dismantling threshold (default: 0.01)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for block growth (default: 0)")
    parser.add_argument("--output", type=Path,
                        help="output PDF path (default: output/pdf/<dataset>_bped.pdf beside the script)")
    args = parser.parse_args()
    output = args.output or here / "output" / "pdf" / f"{args.edgelist.stem}_bped.pdf"
    if output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf")
    if output.resolve() == args.edgelist.resolve():
        parser.error("--output must be different from the input path")

    try:
        graph = load_graph(args.edgelist)
        n, m = len(graph), graph.number_of_edges()
        cap = capacity(args.gamma, n)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    native = build_order(graph, cap, seed=args.seed)
    order, tc = complete_order(graph, native, cap)
    if len(order) != m:
        raise RuntimeError("the full trajectory must remove all original edges")
    curve = lcc_curve(graph, order)
    qc, r, r_full = plot_curve(
        output, args.edgelist.stem, n, m, str(Decimal(args.gamma)), args.seed, curve, tc,
    )
    print(f"{args.edgelist.stem}: N={n}, M={m}, c={cap}, threshold after {tc} removals")
    print(f"qc={qc:.8f}  R_threshold={r:.8f}  R_full={r_full:.8f}")
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
