import networkx as nx
from typing import Optional

def networkx_to_adjlist(g: nx.Graph) -> list[list[int]]:
    adjlist: list[list[int]] = []
    for n in g.nodes:
        adjlist.append(list(g.neighbors(n)))
    return adjlist