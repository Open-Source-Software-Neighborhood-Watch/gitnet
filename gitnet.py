"""Create and analyze graph from csv file. Intended for use with Github contributors and projects but will work with
any arbitrary nodes and edges"""

import collections
import itertools
import os
import time
import argparse
import json
import networkx as nx
from pathlib import Path
import plotly.graph_objects as go


def form_graph(
    input_file,
    node_index,
    edge_index,
    recency_size=0,
    recency_data="dat/bioconda_repo_info.json",
):
    """Create a networkx graph from an input csv file.

    Reads csv file with one connection per line (such as github contributor and the project they worked on) and returns
    graph linking those elements
    Example input format:

    project,contributor,other_info,more_info
    project,contributor2,other_info,more_info
    project2,contributor,other_info,more_info
    project2,contributor3,other_info,more_info

    Args:
        input_file - a csv file that contains information for graph, such as github contributors and projects
        node_index - csv index to be used for nodes
        edge_index - csv index to be used for edges
        recency_size - int to trim repositories by x most recent

    Returns:
        G - a networkx graph
    """

    if not recency_size == 0:
        recency_list = get_recent_repos(recency_size, recency_data)

    edges = collections.defaultdict(list)

    G = nx.Graph()

    # Read import file from GitGeo:
    with open(input_file, "r") as f:

        # Skip first line
        lines = f.readlines()[1:]

        # Remove repos if they're not in the list of most recent repos
        if not recency_size == 0:
            lines_to_remove = []
            for line in lines:
                if not line.strip().split(",")[0] in recency_list:
                    lines_to_remove.append(line)
                else:
                    print(line)

            lines = [i for i in lines if i not in lines_to_remove]

        for line in lines:
            info = line.strip().split(",")
            edges[info[edge_index]].append(info[node_index])

    for i in edges:
        edge_combinations = list(itertools.combinations(edges[i], 2))
        if edges[i] in [
            "ChunhuaLiLab_aPRBind",
            "LeidenRNA_SRHS",
            "BorgwardtLab_MoProEmbeddings",
            "Helena-todd_TInGa",
            "marbl_Winnowmap",
        ]:
            print(edges[i])
        if not edge_combinations:
            # Empty list, add lone node
            G.add_node(edges[i][0])
        else:
            for edge in edge_combinations:
                if G.has_edge(edge[0], edge[1]):
                    G[edge[0]][edge[1]]["weight"] += 1
                else:
                    G.add_edge(edge[0], edge[1], weight=1)

    list_difference = []
    for item in recency_list:
        if item not in G.nodes():
            list_difference.append(item)

    return G


def analyze(graph, quick_analysis=False):
    """Analyzes an input networkx graph.

    Calculations done:
    node_num - total number of nodes in graph
    edge_num - total number of edges in graph
    avg_degree - average of number of edges that are incident to all nodes
    density - ratio of the number of edges with respect to the maximum possible edges
    assortativity - the tendency for nodes of high degree in a graph to be connected to high degree nodes
    average_clustering - a measure of the degree to which nodes in a graph tend to cluster together
    page_rank - results of Google PageRank algorithm

    Args:
        graph - networkx graph with weight="weight"
        quick_analysis - bool that removes calculations not needed for visualization

    Returns:
        results - a dictionary with all finished calculations
    """

    results = {}

    if not quick_analysis:
        results["assortativity"] = nx.degree_assortativity_coefficient(
            graph, weight="weight"
        )
        results["average_clustering"] = nx.algorithms.cluster.average_clustering(
            graph, weight="weight"
        )
        degree = dict(graph.degree(weight="weight"))
        results["avg_degree"] = sum(degree.values()) / len(degree)
        results["density"] = nx.density(graph)
        results["edge_num"] = len(graph.edges())
        results["node_num"] = len(graph.nodes)

        # Hyperlink-Induced Topic Search (HITS) algorithm
        results["hits"] = nx.algorithms.link_analysis.hits_alg.hits(graph)

    # Google PageRank algorithm
    page_ranks = nx.algorithms.link_analysis.pagerank_alg.pagerank(graph)

    # Sort dict based on rank
    results["page_rank"] = {
        key: value
        for key, value in sorted(
            page_ranks.items(), reverse=True, key=lambda item: item[1]
        )
    }

    return results


def write_graph_results(filename, results):
    """Write dictionary results to json file.

    Args:
        filename - filename base for output file. Can be basename or full path (path will be trimmed)
        results - dictionary of analysis results

    Returns:
        null
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]

    filename = Path.cwd() / "results"
    filename.mkdir(exist_ok=True)
    filename /= basename + "_" + timestamp + ".json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=4)


def append_country_information(input_file, node_index, results):
    """Get country from GitGeo csv file and add to rank results, as a percentage of country distribution.

    Args:
        input_file - input file. Should be same as in original form_graph call
        node_index - index of nodes in input_file to parse
        results - dictionary of analysis results

    Returns:
        results - modified results with added country data
    """

    node_locations = collections.defaultdict(dict)

    # Read import file from GitGeo:
    with open(input_file, "r") as f:

        # Skip first line
        next(f)

        for line in f:
            info = line.strip().split(",")

            location = info[-1]
            node = info[node_index]
            if location in node_locations[node]:
                # Location already recorded for this node - increment
                node_locations[node][location] += 1
            else:
                node_locations[node][location] = 1

    # Change locations to percentage
    for node, locations in node_locations.items():
        s = sum(locations.values())
        for key, value in locations.items():
            percent = value * 100.0 / s
            node_locations[node][key] = percent

    # Add locations to rank
    for node in results["page_rank"]:
        results["page_rank"][node] = {
            **{"rank": results["page_rank"][node]},
            **node_locations[node],
        }
        node_locations[node]["Rank"] = results["page_rank"][node]

    return results


def visualize_network(graph):
    """Visualize a networkx network graph.

    Args:
        graph - networkx graph

    Returns:
        null
    """
    pos_ = nx.spring_layout(graph)

    # For each edge, make an edge_trace, append to list
    edge_trace = []
    for edge in graph.edges():

        if graph.get_edge_data(edge[0], edge[1])["weight"] > 0:
            char_1 = edge[0]
            char_2 = edge[1]
        x0, y0 = pos_[char_1]
        x1, y1 = pos_[char_2]
        text = (
            char_1
            + "--"
            + char_2
            + ": "
            + str(graph.get_edge_data(edge[0], edge[1])["weight"])
        )

        trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(
                width=0.3 * graph.get_edge_data(edge[0], edge[1])["weight"] ** 1.75,
                color="cornflowerblue",
            ),
            hoverinfo="text",
            text=([text]),
            mode="lines",
        )
        edge_trace.append(trace)

    # Make a node trace
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        textposition="top center",
        textfont_size=10,
        mode="markers+text",
        hoverinfo="none",
        marker=dict(color=[], size=[], line=None),
    )
    # For each node in graph, get the position and size and add to the node_trace
    for node in graph.nodes():
        x, y = pos_[node]
        node_trace["x"] += tuple([x])
        node_trace["y"] += tuple([y])
        node_trace["marker"]["color"] += tuple(["cornflowerblue"])
        # node_trace["marker"]["size"] += tuple([5 * graph.nodes()[node]["size"]])
        node_trace["text"] += tuple(["<b>" + node + "</b>"])

    # Customize layout
    layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",  # transparent background
        plot_bgcolor="rgba(0,0,0,0)",  # transparent 2nd background
        xaxis={"showgrid": False, "zeroline": False},  # no gridlines
        yaxis={"showgrid": False, "zeroline": False},  # no gridlines
    )
    # Create figure
    fig = go.Figure(layout=layout)
    # Add all edge traces
    for trace in edge_trace:
        fig.add_trace(trace)
    # Add node trace
    fig.add_trace(node_trace)
    # Remove legend
    fig.update_layout(showlegend=False)
    # Remove tick labels
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    # Show figure
    fig.show()


def summarize_countries(results, prune_num, country=None):
    """Creates table with nodes and their associated country.

    Args:
        results - dictionary of analysis results
        prune_num - number of rows to show
        country - show only nodes from specified country. For repos, shows only countries with majority of contributors from that country

    Returns:
        null
    """

    rankings = []
    countries = []
    page_rank = results["page_rank"]
    i = 0

    for rank in page_rank:
        if i == prune_num:
            break
        top_country = max(page_rank[rank], key=page_rank[rank].get)
        if country:
            if country.upper() == top_country.upper():
                rankings.append(rank)
                countries.append(top_country)
                i += 1
        else:
            rankings.append(rank)
            countries.append(top_country)
            i += 1

    # TODO: Read header values from input file
    with open("csv_output.csv", "w") as f:
        f.write("Node,Country\n")
        for i in range(0, len(rankings)):
            f.write(f"{rankings[i]},{countries[i]}\n")

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["Node", "Country"]),
                cells=dict(values=[rankings, countries]),
            )
        ]
    )
    fig.show()


def prune_graph(graph, results, prune_num):
    """Prunes graph to specific size, based on connectivity ratings

    Args:
        graph - networkx graph
        results - dictionary of analysis results
        prune_num - number of nodes to prune to

    Returns:
        pruned_graph - new networkx graph that has been pruned to prune_num size
    """

    top_rankings = {}
    pruned_graph = graph.copy()
    nodes = graph.nodes()

    i = 0
    for (k, v) in results["page_rank"].items():
        if k not in graph.nodes():
            continue
        if i == prune_num:
            break
        top_rankings[k] = v
        i += 1

    for node in nodes:
        if node not in top_rankings:
            pruned_graph.remove_node(node)

    return pruned_graph


def prune_by_country(graph, results, country):
    """Prunes graph to only include results from specific countries

    Args:
        graph - networkx graph
        results - dictionary of analysis results
        country - country to sort by

    Returns:
        country_graph - new networkx graph that has been pruned include only specified country
    """

    country_results = {}
    country_graph = graph.copy()
    nodes = graph.nodes()

    for i, (k, v) in enumerate(results["page_rank"].items()):
        if country.upper() in (nation.upper() for nation in v):
            country_results[k] = v

    for node in nodes:
        if node not in country_results:
            country_graph.remove_node(node)

    return country_graph


def get_recent_repos(recency_size, filepath):

    repo_stems = []

    # open and load json from file
    with open(filepath, "r") as f:
        repos = json.load(f)

    # Remove None from list
    repos = [i for i in repos if i]

    # Sort by date
    sorted_repos = sorted(repos, key=lambda d: d["created_at"], reverse=True)

    # Trim by recency size
    trimmed_repos = sorted_repos[:recency_size]

    # Return list in GitGeo format {group}_{repo}
    for repo in trimmed_repos:
        repo_name = repo["full_name"].replace("/", "_")
        repo_stems.append(repo_name)

    return repo_stems


def parse_command_line_arguments():
    """Parse command line arguments with argparse."""
    parser = argparse.ArgumentParser(
        description="Convert csv file into nodes and edges, such as github contributors and projects",
        epilog="For help with this program, contact John Speed at jmeyers@iqt.org.",
    )
    parser.add_argument(
        "--node-index",
        default=1,
        type=int,
        help="Index of nodes in csv file",
    )
    parser.add_argument(
        "--edge-index",
        default=0,
        type=int,
        help="Index of edges in csv file",
    )
    parser.add_argument(
        "--filepath",
        help="Filepath to csv file for analysis",
    )
    parser.add_argument(
        "--country",
        help="Include only results from specific country",
    )
    parser.add_argument(
        "--prune-size",
        default=100,
        type=int,
        help="Prune results based on top PageRank score",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Base file name for analysis results (timestamp will be added)",
    )
    parser.add_argument(
        "--quick-analysis",
        action="store_true",
        help="Whether to do a full graph analysis or just enough for visualization",
    )
    parser.add_argument(
        "--recency-size",
        default=0,
        type=int,
        help="Trim input by x most recent repositories. 0 for no trim",
    )
    parser.add_argument(
        "--recency-data",
        default="dat/bioconda_repo_info.json",
        help="For use with recency-size. Specifies path to jsonified GitHub repository api calls, including creation date.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_arguments()

    # Form graph
    graph = form_graph(
        args.filepath,
        args.node_index,
        args.edge_index,
        args.recency_size,
        args.recency_data,
    )

    # Analyze graph
    results = analyze(graph, args.quick_analysis)
    locational_results = append_country_information(
        args.filepath, args.node_index, results
    )

    # Save results
    # filename = args.filename or args.filepath
    # write_graph_results(filename, locational_results)

    # Prune graph
    if args.country:
        pruned_graph = prune_by_country(graph, results, args.country)
        if not args.quick_analysis:
            # Analyze country results
            country_results = analyze(pruned_graph, args.quick_analysis)
            country_locational_results = append_country_information(
                args.filepath, args.node_index, country_results
            )
            # Save county results
            filename = args.filename or args.filepath
            write_graph_results(
                f"{args.country}_{filename}", country_locational_results
            )

        pruned_graph = prune_graph(pruned_graph, results, args.prune_size)
    else:
        pruned_graph = prune_graph(graph, results, args.prune_size)

    # Visualize Results
    visualize_network(pruned_graph)
    summarize_countries(results, args.prune_size, args.country)
