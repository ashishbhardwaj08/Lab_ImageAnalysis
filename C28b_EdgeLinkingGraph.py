import cv2
import numpy as np

# -----------------------------
# DFS function
# -----------------------------
def dfs(edges, visited, x, y, component):
    stack = [(x, y)]
    visited[x, y] = 1

    while stack:
        cx, cy = stack.pop()
        component.append((cx, cy))

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = cx + dx, cy + dy

                if 0 <= nx < edges.shape[0] and 0 <= ny < edges.shape[1]:
                    if edges[nx, ny] == 255 and not visited[nx, ny]:
                        visited[nx, ny] = 1
                        stack.append((nx, ny))

# -----------------------------
# Edge Linking Function
# -----------------------------
def link_edges(edges):
    visited = np.zeros_like(edges, dtype=np.uint8)
    h, w = edges.shape
    components = []

    for i in range(h):
        for j in range(w):
            if edges[i, j] == 255 and not visited[i, j]:
                component = []
                dfs(edges, visited, i, j, component)
                components.append(component)

    return components

# -----------------------------
# MAIN PROGRAM
# -----------------------------
img = cv2.imread("Image5.png")   # <-- change path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1: Edge detection
edges = cv2.Canny(gray, 100, 200)

# Step 2: Edge linking
components = link_edges(edges)

# Step 3: Draw linked edges in random colors
output = np.zeros_like(img)

for comp in components:
    color = np.random.randint(0, 255, size=3).tolist()
    for (x, y) in comp:
        output[x, y] = color

# -----------------------------
# Display results
# -----------------------------
cv2.imshow("Original", img)
cv2.imshow("Edges", edges)
cv2.imshow("Linked Edges (Components)", output)

cv2.waitKey(0)
cv2.destroyAllWindows()