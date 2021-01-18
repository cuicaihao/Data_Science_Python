# the graph
graph = {}
graph["start"] = {}
graph["start"]["a"] = 6
graph["start"]["b"] = 2

graph["a"] = {}
graph["a"]["fin"] = 1

graph["b"] = {}
graph["b"]["a"] = 3
graph["b"]["fin"] = 5

graph["fin"] = {}

print(graph)

# the costs table
infinity = float("inf")
costs = {}
costs["a"] = 6
costs["b"] = 2
costs["fin"] = infinity

# the parents table
parents = {}
parents["a"] = "start"
parents["b"] = "start"
parents["fin"] = None


def find_lowest_cost_node(costs, processed):
    lowest_cost = float("inf")
    lowest_cost_node = None
    # Go through each node.
    for node in costs:
        cost = costs[node]
        # If it's the lowest cost so far and hasn't been processed yet...
        if cost < lowest_cost and node not in processed:
            # ... set it as the new lowest-cost node.
            lowest_cost = cost
            lowest_cost_node = node
    return lowest_cost_node


def generate_shortest_path(parents):
    node_path = ["fin"]
    for i in range(len(parents)):
        node_name = node_path[-1]
        node_name = parents[node_name]
        node_path.append(node_name)
        if node_name == "start":
            break
    node_path.reverse()
    print("The shortest node path from start to end is : ", node_path)
    return node_path


def dijkstrasAlgorithm(graph, costs, parents):
    processed = []
    # Find the lowest-cost node that you haven't processed yet.
    node = find_lowest_cost_node(costs, processed)
    # If you've processed all the nodes, this while loop is done.
    while node is not None:
        cost = costs[node]
        # Go through all the neighbors of this node.
        neighbors = graph[node]
        for n in neighbors.keys():
            new_cost = cost + neighbors[n]
            # If it's cheaper to get to this neighbor by going through this node...
            if costs[n] > new_cost:
                # ... update the cost for this node.
                costs[n] = new_cost
                # This node becomes the new parent for this neighbor.
                parents[n] = node
        # Mark the node as processed.
        processed.append(node)
        # Find the next node to process, and loop.
        node = find_lowest_cost_node(costs, processed)

    print("Cost from the start to each node:")
    print(costs)
    print(parents)
    node_path = generate_shortest_path(parents)
    return costs, parents, node_path
    # print(len(parents))


dijkstrasAlgorithm(graph, costs, parents)
