from compiler.ir import IRGraph


def analyze_graph(graph: IRGraph):
    return {
        "node_count": len(graph.nodes),
        "peak_memory_bytes": graph.compute_peak_memory(),
    }

def percent_reduction(before, after):
    if before == 0:
        return 0.0

    return ((before - after) / before) * 100

def compare_graphs(before: IRGraph, after: IRGraph):
    before_stats = analyze_graph(before)
    after_stats = analyze_graph(after)

    return {
        "before": before_stats,
        "after": after_stats,

        "node_reduction_percent": percent_reduction(
            before_stats["node_count"],
            after_stats["node_count"],
        ),

        "memory_reduction_percent": percent_reduction(
            before_stats["peak_memory_bytes"],
            after_stats["peak_memory_bytes"],
        ),
    }