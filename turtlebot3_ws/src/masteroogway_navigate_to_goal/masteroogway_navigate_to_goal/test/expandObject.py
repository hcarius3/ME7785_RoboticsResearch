import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString

def expand_object(edges, safetyDistance):
    """
    Expands an object (polygon or line) outward with squared edges.
    
    :param edges: List of (x, y) tuples representing the object vertices.
    :param safetyDistance: Minimum distance to expand outward.
    :return: Shapely geometry of the expanded object.
    """
    if len(edges) == 2:  # If it's just a line segment
        (x1, y1), (x2, y2) = edges
        
        # Compute direction vector
        dx, dy = x2 - x1, y2 - y1
        length = (dx**2 + dy**2) ** 0.5
        ux, uy = dx / length, dy / length  # Unit direction vector
        
        # Extend line endpoints
        x1_ext, y1_ext = x1 - ux * safetyDistance, y1 - uy * safetyDistance
        x2_ext, y2_ext = x2 + ux * safetyDistance, y2 + uy * safetyDistance

        geometry = LineString([(x1_ext, y1_ext), (x2_ext, y2_ext)])
    else:  # Otherwise, assume it's a polygon
        geometry = Polygon(edges)
    
    expanded_geometry = geometry.buffer(safetyDistance, cap_style=3, join_style=2, mitre_limit=1.1)
    # expanded_geometry = geometry.buffer(safetyDistance, cap_style=3, join_style=2)
    return expanded_geometry

def visualize_expansion(original_edges, expanded_shape):
    """
    Visualizes the original shape and its expanded version.
    """
    fig, ax = plt.subplots()
    
    # Plot original shape
    if len(original_edges) == 2:
        x, y = zip(*original_edges)
        ax.plot(x, y, 'bo-', label="Original Line", linewidth=2)
    else:
        orig_polygon = Polygon(original_edges)
        x, y = orig_polygon.exterior.xy
        ax.plot(x, y, 'bo-', label="Original Shape", linewidth=2)

    # Plot expanded shape
    x, y = expanded_shape.exterior.xy
    ax.fill(x, y, color='red', alpha=0.5, label="Expanded Area")

    ax.legend()
    ax.set_aspect('equal')
    plt.show()

# Example usage
# edges = [(0, 0), (4, 0), (4, 3), (0, 3)]  # Rectangle
edges = [(0.895, 1.51), (0.995, 1.43), (0.77, 1.19)]  # Triangle
# edges = [(0, 0), (2, 1), (2, 4)]  # Triangle
# edges = [(0, 0), (4, 1)]  # Try this for a two-point object (line segment)
safetyDistance = 0.15

expanded_shape = expand_object(edges, safetyDistance)
expanded_coords = list(expanded_shape.exterior.coords)
print("Expanded Shape Coordinates:", expanded_coords)
print("Number of Points:", len(expanded_coords))
visualize_expansion(edges, expanded_shape)
