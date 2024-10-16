from collections import defaultdict


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def DLS(self, src, target, maxDepth, path):
        path.append(src)

        if src == target:
            return True

        if maxDepth <= 0:
            path.pop()  # Backtrack if not found at this level
            return False

        for i in self.graph[src]:
            if self.DLS(i, target, maxDepth - 1, path):
                return True

        path.pop()  # Backtrack if target not found in any adjacent node
        return False

    def IDDFS(self, src, target, maxDepth):
        path = []

        for i in range(maxDepth):
            if self.DLS(src, target, i, path):
                return path  # Return path if target is found

        return None  # Return None if target is not reachable within maxDepth


g = Graph(7)
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 3)
g.addEdge(1, 4)
g.addEdge(2, 5)
g.addEdge(2, 6)

target = 6
maxDepth = 3
src = 0

path = g.IDDFS(src, target, maxDepth)
if path:
    print(f"Target is reachable from source within max depth. Path: {path}")
else:
    print("Target is NOT reachable from source within max depth")