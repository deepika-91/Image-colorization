graph = {
  '5' : ['3','7'],
  '3' : ['2', '4'],
  '7' : ['8'],
  '2' : [],
  '4' : ['8'],
  '8' : []
}

visited= []
queue=[]

def bfs(visited, graph, node):
    visited.append(node)
    queue.append(node)

    while queue:
        s = queue.pop(0)
        print(s, end=" ")

        for neighbour in graph[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)


print("Following is the Breadth-First Search")
bfs(visited, graph, '5')

"""DFS(Depth-First-Search)"""

graph = {
  '5' : ['3','7'],
  '3' : ['2', '4'],
  '7' : ['8'],
  '2' : [],
  '4' : ['8'],
  '8' : []
}

visited= set()#set to tract visited nodes of graph


def dfs(visited, graph, node):
   if node not in visited:
    print(node)
    visited.add(node)
    for neighbour in graph[node]:
      dfs(visited,graph,neighbour)


print("Following is the Depth-First Search")
dfs(visited, graph, '5')

"""IDFS(Iterative-Deepening-First-Search)"""

from collections import defaultdict

n = int(input("Enter No of Nodes:"))
e = int(input("Enter No of Edge:"))

graph = defaultdict(list)

for i in range(e):
    i,j = map(int,input().split())
    graph[i].append(j)

def dfs(v,goal,limit):
    if v == goal:
        return 1;

    for i in graph[v]:
        if limit-1 >= 0:
            if dfs(i,goal,limit-1) !=  -1:
                return 1
    return -1


goal = int(input("Enter Goal:"))
limit = int(input("Enter Limit:"))

res = dfs(0,goal,limit)

if res == -1:
    print("not found")
else:
    print("found within depth limit")

"""A Star Algortihm

"""

Graph_nodes = {       #dictionary for graph creation
    'A': [('B', 6), ('F', 3)],
    'B': [('C', 3), ('D', 2)],
    'C': [('D', 1), ('E', 5)],
    'D': [('C', 1), ('E', 8)],
    'E': [('I', 5), ('J', 5)],
    'F': [('G', 1), ('H', 7)],
    'G': [('I', 3)],
    'H': [('I', 2)],
    'I': [('E', 5), ('J', 3)],

}


def get_neighbors(v):            #returns neighbour node v= Node A
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None


def h(n):     #heuristic values
    H_dist = {
        'A': 10,
        'B': 8,
        'C': 5,
        'D': 7,
        'E': 3,
        'F': 6,
        'G': 5,
        'H': 3,
        'I': 1,
        'J': 0
    }
    return H_dist[n]


def aStarAlgo(start_node, stop_node):  #working process of alg
    open_set = set(start_node)
    closed_set = set()
    g = {}          #distance Path
    parents = {}
    g[start_node] = 0
    parents[start_node] = start_node

    while len(open_set) > 0:   #for traversing a graph
        n = None

        for v in open_set:
            if n == None or g[v] + h(v) < g[n] + h(n):
                n = v

        if n == stop_node or Graph_nodes[n] == None:
            pass
        else:
            for (m, weight) in get_neighbors(n):
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight

                else:
                    if g[m] > g[n] + weight: # to find Optimistic path
                        g[m] = g[n] + weight
                        parents[m] = n
                        if m in closed_set:
                            closed_set.remove(m) #to get optimistic path
                            open_set.add(m)

        if n == None:
            print('Path does not exist!')
            return None
        if n == stop_node:
            path = []

            while parents[n] != n:    # parent of B should not be B
                path.append(n)
                n = parents[n]

            path.append(start_node)    #add start node to path

            path.reverse()             # Applying reverse fun to get path from root node

            print('Path found: {}'.format(path))
            return path
        open_set.remove(n)    #after explore nodes will remove from opened set
        closed_set.add(n)     #after explore nodes will add to closed set

    print('Path does not exist!')
    return None


aStarAlgo('A', 'J')

# Map Coloring Problem for Australia using Backtracking Algorithm

# Variables (regions) and their neighbors
regions = {
    'WA': ['NT', 'SA'],
    'NT': ['WA', 'SA', 'Q'],
    'Q': ['NT', 'SA', 'NSW'],
    'NSW': ['Q', 'V', 'SA'],
    'V': ['NSW', 'SA'],
    'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
    'T': []  # Tasmania is not connected to any other region
}

# Colors available for each region
colors = ['Red', 'Green', 'Blue']

# Assign colors to regions such that adjacent regions don't share the same color
def is_safe(region, color, coloring):
    for neighbor in regions[region]:
        if coloring.get(neighbor) == color:
            return False
    return True

def solve_map_coloring(region, coloring):
    if region is None:
        return True

    for color in colors:
        if is_safe(region, color, coloring):
            coloring[region] = color

            # Proceed to the next region
            next_region = get_uncolored_region(coloring)
            if solve_map_coloring(next_region, coloring):
                return True

            # If the coloring doesn't lead to a solution, backtrack
            coloring[region] = None

    return False

def get_uncolored_region(coloring):
    for region in regions:
        if coloring.get(region) is None:
            return region
    return None

# Solve the map coloring problem
if __name__ == "__main__":
    coloring = {region: None for region in regions}
    if solve_map_coloring(get_uncolored_region(coloring), coloring):
        print("Solution Found!")
        for region, color in coloring.items():
            print(f"{region}: {color}")
    else:
        print("No solution exists.")