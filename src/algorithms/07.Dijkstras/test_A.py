from dijkstras_algorithm import dijkstrasAlgorithm


# %%
def test_case_A():
    # the graph
    graph = {}
    # add node and edges
    graph["start"] = {}
    graph["start"]["A"] = 5
    graph["start"]["C"] = 2

    graph["A"] = {}
    graph["A"]["B"] = 4
    graph["A"]["D"] = 2

    graph["B"] = {}
    graph["B"]["fin"] = 3
    graph["B"]["D"] = 6

    graph["C"] = {}
    graph["C"]["A"] = 8
    graph["C"]["D"] = 7

    graph["D"] = {}
    graph["D"]["fin"] = 1

    graph["fin"] = {}

    print(graph)

    # the costs table
    infinity = float("inf")
    costs = {}
    costs["A"] = graph["start"]["A"]
    costs["B"] = graph["start"].get("B", infinity)
    costs["C"] = infinity
    costs["D"] = infinity
    costs["fin"] = infinity

    # the parents table
    parents = {}
    parents["A"] = "start"
    parents["B"] = "A"
    parents["C"] = "start"
    parents["D"] = "B"
    parents["fin"] = None

    # Run the dijkstras algorithm
    costs, parents, node_path = dijkstrasAlgorithm(graph, costs, parents)
    assert node_path == ["start", "A", "D", "fin"]


if __name__ == "__main__":
    test_case_A()
    print("PASS")
