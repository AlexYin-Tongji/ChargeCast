"""Topology processing utilities."""

from __future__ import annotations

from collections import deque
from typing import Dict, List

import pandas as pd


class TopologyProcessor:
    """Process graph structure from adjacency matrix."""

    def __init__(self, adj_df: pd.DataFrame):
        self.adj_df = adj_df
        self.graph = self._build_graph()
        self.nodes = self._get_nodes()
        self.exogenous_nodes = self._get_exogenous_nodes()
        self.endogenous_nodes = self._get_endogenous_nodes()
        self.topological_order = self._topological_sort()

    def _build_graph(self) -> Dict[str, List[str]]:
        graph = {}
        source_nodes = [str(x) for x in self.adj_df.index.tolist()]
        target_nodes = [str(x) for x in self.adj_df.columns.tolist()]

        for i, source in enumerate(source_nodes):
            graph[source] = []
            for j, target in enumerate(target_nodes):
                value = self.adj_df.iloc[i, j]
                if value > 0:
                    graph[source].append(target)
        return graph

    def _get_nodes(self) -> List[str]:
        source_nodes = [str(x) for x in self.adj_df.index.tolist()]
        target_nodes = [str(x) for x in self.adj_df.columns.tolist()]
        return list(dict.fromkeys(source_nodes + target_nodes))

    def _compute_in_degree(self) -> Dict[str, int]:
        in_degree = {}
        for target in self.adj_df.columns:
            in_degree[str(target)] = int((self.adj_df[target] > 0).sum())
        for node in self.nodes:
            if node not in in_degree:
                in_degree[node] = 0
        return in_degree

    def _get_exogenous_nodes(self) -> List[str]:
        in_degree = self._compute_in_degree()
        return [node for node in self.nodes if in_degree.get(node, 0) == 0]

    def _get_endogenous_nodes(self) -> List[str]:
        exogenous_set = set(self.exogenous_nodes)
        return [n for n in self.nodes if n not in exogenous_set]

    def _compute_distance_to_exogenous(self) -> Dict[str, float]:
        distances: Dict[str, float] = {}
        queue = deque()

        for ex_node in self.exogenous_nodes:
            distances[ex_node] = 0
            queue.append(ex_node)

        reverse_graph: Dict[str, List[str]] = {}
        for source, targets in self.graph.items():
            for target in targets:
                reverse_graph.setdefault(target, []).append(source)

        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            for upstream in reverse_graph.get(current, []):
                if upstream not in distances:
                    distances[upstream] = current_dist + 1
                    queue.append(upstream)

        for node in self.nodes:
            if node not in distances:
                distances[node] = float("inf")

        return distances

    def _topological_sort(self) -> List[str]:
        distances = self._compute_distance_to_exogenous()
        sorted_nodes = sorted(
            self.endogenous_nodes,
            key=lambda x: (distances.get(x, float("inf")), str(x)),
        )
        return self.exogenous_nodes + sorted_nodes

    def get_upstream_nodes(self, node: str) -> List[str]:
        upstream = []
        node = str(node)

        if node not in self.adj_df.columns:
            return upstream

        target_nodes = [str(x) for x in self.adj_df.columns.tolist()]
        target_idx = target_nodes.index(node)

        source_nodes = [str(x) for x in self.adj_df.index.tolist()]
        for i, source in enumerate(source_nodes):
            value = self.adj_df.iloc[i, target_idx]
            if value > 0:
                upstream.append(source)

        return upstream

    def get_downstream_nodes(self, node: str) -> List[str]:
        return self.graph.get(str(node), [])

    def get_out_degree(self, node: str) -> int:
        return len(self.get_downstream_nodes(node))

    def get_in_degree(self, node: str) -> int:
        return self._compute_in_degree().get(str(node), 0)

    def get_topological_order(self) -> List[str]:
        return self.topological_order

    def is_exogenous(self, node: str) -> bool:
        return str(node) in set(self.exogenous_nodes)


def demo():
    data = {
        "S001": [0, 0, 1, 0],
        "S002": [0, 0, 1, 1],
        "S003": [0, 0, 0, 1],
    }
    adj_df = pd.DataFrame(data, index=["S001", "S002", "S003", "S004"])
    topo = TopologyProcessor(adj_df)

    print("all nodes:", topo.nodes)
    print("exogenous:", topo.exogenous_nodes)
    print("endogenous:", topo.endogenous_nodes)
    print("topological order:", topo.get_topological_order())


if __name__ == "__main__":
    demo()
