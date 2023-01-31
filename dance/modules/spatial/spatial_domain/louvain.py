"""Reimplementation of Louvain.

Extended from https://github.com/taynaud/python-louvain

Reference
----------
Blondel, V. D., et al. "Fast Unfolding of Community Hierarchies in Large Networks, 1–6 (2008)." arXiv:0803.0476.

"""

import array
import numbers
import warnings

import networkx as nx
import numpy as np
import scanpy as sc

from dance import logger
from dance.modules.base import BaseClusteringMethod
from dance.transforms import AnnDataTransform, CellPCA, Compose, FilterGenesMatch, SetConfig
from dance.transforms.graph import NeighborGraph
from dance.typing import LogLevel

PASS_MAX = -1
MIN = 0.0000001


class Status:
    """To handle several data in one struct.

    Could be replaced by named tuple, but don't want to depend on python
    2.6

    """
    node2com = {}
    total_weight = 0
    internals = {}
    degrees = {}
    gdegrees = {}

    def __init__(self):
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.loops = dict([])

    def __str__(self):
        return ("node2com : " + str(self.node2com) + " degrees : " + str(self.degrees) + " internals : " +
                str(self.internals) + " total_weight : " + str(self.total_weight))

    def copy(self):
        """Perform a deep copy of status."""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight

    def init(self, graph, weight, part=None):
        """Initialize the status of a graph with every node in one community."""
        count = 0
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.total_weight = graph.size(weight=weight)
        if part is None:
            for node in graph.nodes():
                self.node2com[node] = count
                deg = float(graph.degree(node, weight=weight))
                if deg < 0:
                    error = "Bad node degree ({})".format(deg)
                    raise ValueError(error)
                self.degrees[count] = deg
                self.gdegrees[node] = deg
                edge_data = graph.get_edge_data(node, node, default={weight: 0})
                self.loops[node] = float(edge_data.get(weight, 1))
                self.internals[count] = self.loops[node]
                count += 1
        else:
            for node in graph.nodes():
                com = part[node]
                self.node2com[node] = com
                deg = float(graph.degree(node, weight=weight))
                self.degrees[com] = self.degrees.get(com, 0) + deg
                self.gdegrees[node] = deg
                inc = 0.
                for neighbor, data in graph[node].items():
                    edge_weight = data.get(weight, 1)
                    if edge_weight <= 0:
                        error = "Bad graph type ({})".format(type(graph))
                        raise ValueError(error)
                    if part[neighbor] == com:
                        if neighbor == node:
                            inc += float(edge_weight)
                        else:
                            inc += float(edge_weight) / 2.
                self.internals[com] = self.internals.get(com, 0) + inc


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


def partition_at_level(dendrogram, level):
    """Return the partition of the nodes at the given level.

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1.
    The higher the level is, the bigger are the communities

    Parameters
    ----------
    dendrogram : list of dict
       a list of partitions, ie dictionaries where keys of the i+1 are the
       values of the i.
    level : int
       the level which belongs to [0..len(dendrogram)-1]

    Returns
    -------
    partition : dictionary
       A dictionary where keys are the nodes and the values are the set it
       belongs to

    Raises
    ------
    KeyError
       If the dendrogram is not well formed or the level is too high

    See Also
    --------
    best_partition : which directly combines partition_at_level and
    generate_dendrogram : to obtain the partition of highest modularity

    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendrogram = generate_dendrogram(G)
    >>> for level in range(len(dendrogram) - 1) :
    >>>     print("partition at level", level, "is", partition_at_level(dendrogram, level))  # NOQA

    """
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition


def modularity(partition, graph, weight="weight"):
    """Compute the modularity of a partition of a graph.

    Parameters
    ----------
    partition : dict
       the partition of the nodes, i.e a dictionary where keys are their nodes
       and values the communities
    graph : networkx.Graph
       the networkx graph which is decomposed
    weight : str
        the key in graph to use as weight. Default to "weight"


    Returns
    -------
    modularity : float
       The modularity

    Raises
    ------
    KeyError
       If the partition is not a partition of all graph nodes
    ValueError
        If the graph has no link
    TypeError
        If graph is not a networkx.Graph

    References
    ----------
    .. 1. Newman, M.E.J. & Girvan, M. Finding and evaluating community
    structure in networks. Physical Review E 69, 26113(2004).

    Examples
    --------
    >>> import community as community_louvain
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(100, 0.01)
    >>> partition = community_louvain.best_partition(G)
    >>> modularity(partition, G)

    """
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = graph.size(weight=weight)
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
        for neighbor, data in graph[node].items():
            edge_weight = data.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) - \
               (deg.get(com, 0.) / (2. * links)) ** 2
    return res


def best_partition(graph, partition=None, weight="weight", resolution=1., randomize=None, random_state=None):
    """Compute the partition of the graph nodes which maximises the modularity (or
    try..) using the Louvain heuristices.

    This is the partition of highest modularity, i.e. the highest partition
    of the dendrogram generated by the Louvain algorithm.

    Parameters
    ----------
    graph : networkx.Graph
       the networkx graph which is decomposed
    partition : dict
       the algorithm will start using this partition of the nodes.
       It's a dictionary where keys are their nodes and values the communities
    weight : str
        the key in graph to use as weight. Default to "weight"
    resolution :  double
        Will change the size of the communities, default to 1.
        represents the time described in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona
    randomize : boolean
        Will randomize the node evaluation order and the community evaluation
        order to get different partitions at each call
    random_state : int, RandomState instance or None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by :func:`numpy.random`.

    Returns
    -------
    partition : dictionary
       The partition, with communities numbered from 0 to number of communities

    Raises
    ------
    NetworkXError
       If the graph is not undirected.

    See Also
    --------
    generate_dendrogram : to obtain all the decompositions levels

    Notes
    -----
    Uses Louvain algorithm

    References
    ----------
    .. 1. Blondel, V.D. et al. Fast unfolding of communities in
    large networks. J. Stat. Mech 10008, 1-12(2008).

    Examples
    --------
    >>> # basic usage
    >>> import community as community_louvain
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(100, 0.01)
    >>> partition = community_louvain.best_partition(G)

    >>> # display a graph with its communities:
    >>> # as Erdos-Renyi graphs don't have true community structure,
    >>> # instead load the karate club graph
    >>> import community as community_louvain
    >>> import matplotlib.cm as cm
    >>> import matplotlib.pyplot as plt
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> # compute the best partition
    >>> partition = community_louvain.best_partition(G)

    >>> # draw the graph
    >>> pos = nx.spring_layout(G)
    >>> # color the nodes according to their partition
    >>> cmap = cm.get_cmap("viridis", max(partition.values()) + 1)
    >>> nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
    >>>                        cmap=cmap, node_color=list(partition.values()))
    >>> nx.draw_networkx_edges(G, pos, alpha=0.5)
    >>> plt.show()

    """
    dendo = generate_dendrogram(graph, partition, weight, resolution, randomize, random_state)
    return partition_at_level(dendo, len(dendo) - 1)


class Louvain(BaseClusteringMethod):
    """Louvain classBaseClassificationMethod.

    Parameters
    ----------
    resolution
        Resolution parameter.

    """

    def __init__(self, resolution: float = 1):
        self.resolution = resolution

    @staticmethod
    def preprocessing_pipeline(dim: int = 50, n_neighbors: int = 17, log_level: LogLevel = "INFO"):
        return Compose(
            FilterGenesMatch(prefixes=["ERCC", "MT-"]),
            AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
            AnnDataTransform(sc.pp.log1p),
            CellPCA(n_components=dim),
            NeighborGraph(n_neighbors=n_neighbors),
            SetConfig({
                "feature_channel": "NeighborGraph",
                "feature_channel_type": "obsp",
                "label_channel": "label",
                "label_channel_type": "obs"
            }),
            log_level=log_level,
        )

    def fit(self, adj, partition=None, weight="weight", randomize=None, random_state=None):
        """Fit function for model training.

        Parameters
        ----------
        adj :
            adjacent matrix.
        partition : dict
            a dictionary where keys are graph nodes and values the part the node
            belongs to
        weight : str
            the key in graph to use as weight. Default to "weight"
        randomize : boolean
            Will randomize the node evaluation order and the community evaluation
            order to get different partitions at each call
        random_state : int, RandomState instance or None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by :func:`numpy.random`.

        """
        # convert adata,adj into networkx
        logger.info("Converting adjacency matrix to networkx graph...")
        if (adj - adj.T).sum() != 0:
            ValueError("louvain use no direction graph, but the input is not")
        g = nx.from_numpy_array(adj)
        logger.info("Conversion done. Start fitting...")
        self.dendo = generate_dendrogram(g, partition, weight, self.resolution, randomize, random_state)
        logger.info("Fitting done.")

    def predict(self, x=None):
        """Prediction function.

        Parameters
        ----------
        x
            Not used. For compatibility with :func:`dance.modules.base.BaseMethod.fit_score`,  which calls :meth:`fit`
            with ``x``.

        """
        pred_dict = partition_at_level(self.dendo, len(self.dendo) - 1)
        pred = np.array(list(map(pred_dict.get, sorted(pred_dict))))
        return pred


def generate_dendrogram(graph, part_init=None, weight="weight", resolution=1., randomize=None, random_state=None):
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    # Properly handle random state, eventually remove old `randomize` parameter
    # NOTE: when `randomize` is removed, delete code up to random_state = ...
    if randomize is not None:
        warnings.warn("The `randomize` parameter will be deprecated in future "
                      "versions. Use `random_state` instead.", DeprecationWarning)
        # If shouldn't randomize, we set a fixed seed to get deterministic results
        if randomize is False:
            random_state = 0

    # We don't know what to do if both `randomize` and `random_state` are defined
    if randomize and random_state is not None:
        raise ValueError("`randomize` and `random_state` cannot be used at the "
                         "same time")

    random_state = check_random_state(random_state)

    # special case, when there is no link
    # the best partition is everyone in its community
    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]

    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part_init)
    status_list = list()
    _one_level(current_graph, status, weight, resolution, random_state)
    new_mod = _modularity(status, resolution)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    current_graph = induced_graph(partition, current_graph, weight)
    status.init(current_graph, weight)

    while True:
        _one_level(current_graph, status, weight, resolution, random_state)
        new_mod = _modularity(status, resolution)
        if new_mod - mod < MIN:
            break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        current_graph = induced_graph(partition, current_graph, weight)
        status.init(current_graph, weight)
    return status_list[:]


def induced_graph(partition, graph, weight="weight"):
    """Produce the graph where nodes are the communities.

    there is a link of weight w between communities if the sum of the weights
    of the links between their elements is w

    Parameters
    ----------
    partition : dict
       a dictionary where keys are graph nodes and  values the part the node
       belongs to
    graph : networkx.Graph
        the initial graph
    weight : str, optional
        the key in graph to use as weight. Default to "weight"


    Returns
    -------
    g : networkx.Graph
       a networkx graph where nodes are the parts

    Examples
    --------
    >>> n = 5
    >>> g = nx.complete_graph(2*n)
    >>> part = dict([])
    >>> for node in g.nodes() :
    >>>     part[node] = node % 2
    >>> ind = induced_graph(part, g)
    >>> goal = nx.Graph()
    >>> goal.add_weighted_edges_from([(0,1,n*n),(0,0,n*(n-1)/2), (1, 1, n*(n-1)/2)])  # NOQA
    >>> nx.is_isomorphic(ind, goal)
    True

    """
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())

    for node1, node2, data in graph.edges(data=True):
        edge_weight = data.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})

    return ret


def __renumber(dictionary):
    """Renumber the values of the dictionary from 0 to n."""
    values = set(dictionary.values())
    target = set(range(len(values)))

    if values == target:
        # no renumbering necessary
        ret = dictionary.copy()
    else:
        # add the values that won't be renumbered
        renumbering = dict(zip(target.intersection(values), target.intersection(values)))
        # add the values that will be renumbered
        renumbering.update(dict(zip(values.difference(target), target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret


def load_binary(data):
    """Load binary graph as used by the cpp implementation of this algorithm."""
    data = open(data, "rb")

    reader = array.array("I")
    reader.fromfile(data, 1)
    num_nodes = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_nodes)
    cum_deg = reader.tolist()
    num_links = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_links)
    links = reader.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0

    for index in range(num_nodes):
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg

    return graph


def _one_level(graph, status, weight_key, resolution, random_state):
    """Compute one level of communities."""
    modified = True
    nb_pass_done = 0
    cur_mod = _modularity(status, resolution)
    new_mod = cur_mod

    while modified and nb_pass_done != PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in _randomize(graph.nodes(), random_state):
            com_node = status.node2com[node]
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
            neigh_communities = _neighcom(node, graph, status, weight_key)
            remove_cost = - neigh_communities.get(com_node, 0) + \
                resolution * (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
            _remove(node, com_node, neigh_communities.get(com_node, 0.), status)
            best_com = com_node
            best_increase = 0
            for com, dnc in _randomize(neigh_communities.items(), random_state):
                incr = remove_cost + dnc - \
                       resolution * status.degrees.get(com, 0.) * degc_totw
                if incr > best_increase:
                    best_increase = incr
                    best_com = com
            _insert(node, best_com, neigh_communities.get(best_com, 0.), status)
            if best_com != com_node:
                modified = True
        new_mod = _modularity(status, resolution)
        if new_mod - cur_mod < MIN:
            break


def _neighcom(node, graph, status, weight_key):
    """Compute the communities in the neighborhood of node in the graph given with the
    decomposition node2com."""
    weights = {}
    for neighbor, data in graph[node].items():
        if neighbor != node:
            edge_weight = data.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights


def _remove(node, com, weight, status):
    """Remove node from community com and modify status."""
    status.degrees[com] = (status.degrees.get(com, 0.) - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) - weight - status.loops.get(node, 0.))
    status.node2com[node] = -1


def _insert(node, com, weight, status):
    """Insert node into community and modify status."""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) + status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) + weight + status.loops.get(node, 0.))


def _modularity(status, resolution):
    """Fast compute the modularity of the partition of the graph using status
    precomputed."""
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree * resolution / links - ((degree / (2. * links))**2)
    return result


def _randomize(items, random_state):
    """Returns a List containing a random permutation of items."""
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items
