import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

# --- 1. Basic Structures ---

@dataclass
class AABB:
    id: int
    min_point: np.ndarray  # [x, y, z]
    max_point: np.ndarray  # [x, y, z]
    
    @property
    def center(self):
        """Calculates the geometric center of the AABB."""
        return (self.min_point + self.max_point) / 2.0
    
    @property
    def volume(self):
        """Calculates the volume of the AABB."""
        dims = np.maximum(0, self.max_point - self.min_point)
        return np.prod(dims)

    def intersects(self, other: 'AABB') -> bool:
        """Checks if this AABB intersects with another AABB."""
        return np.all(self.max_point >= other.min_point) and \
               np.all(self.min_point <= other.max_point)

    def intersection_aabb(self, other: 'AABB') -> 'AABB':
        """Return the AABB representing the intersection volume."""
        if not self.intersects(other):
            return None
        min_p = np.maximum(self.min_point, other.min_point)
        max_p = np.minimum(self.max_point, other.max_point)
        return AABB(id=-1, min_point=min_p, max_point=max_p)

# --- 2. Broad Phase: Spatial Hashing ---

class SpatialHash:
    """
    A spatial partitioning system that hashes 3D space into grid cells.
    Used for Broad Phase collision detection to filter candidates.
    """
    def __init__(self, cell_size: float):
        """
        Initialize the spatial hash.
        :param cell_size: The size of each grid cell. Should be roughly the size of the objects.
        """
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int, int], List[int]] = {}

    def _get_cell_coords(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Converts a 3D point into integer grid cell coordinates."""
        return tuple(np.floor(point / self.cell_size).astype(int))

    def insert(self, obj: AABB):
        """
        Inserts an AABB into the spatial hash.
        The object is added to every cell it overlaps with.
        """
        min_cell = self._get_cell_coords(obj.min_point)
        max_cell = self._get_cell_coords(obj.max_point)

        # Iterate over all cells this object touches
        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    cell_key = (x, y, z)
                    if cell_key not in self.grid:
                        self.grid[cell_key] = []
                    self.grid[cell_key].append(obj.id)

    def get_candidates(self) -> Set[Tuple[int, int]]:
        """
        Returns unique pairs of potential colliding object IDs.
        Only objects sharing at least one grid cell are returned.
        """
        candidates = set()
        for objects_in_cell in self.grid.values():
            if len(objects_in_cell) < 2:
                continue
            # Generate pairs (i, j) where i < j
            for i in range(len(objects_in_cell)):
                for j in range(i + 1, len(objects_in_cell)):
                    id_a, id_b = objects_in_cell[i], objects_in_cell[j]
                    if id_a > id_b:
                        id_a, id_b = id_b, id_a
                    candidates.add((id_a, id_b))
        return candidates

# --- 3. Narrow Phase: Voxel Intersection Sampling ---

def analyze_interference_voxelized(obj_a: AABB, obj_b: AABB, voxel_size: float = 0.5):
    """
    Simulates finding the 'shape' of interference by sampling the intersection AABB.
    
    :param obj_a: First object AABB
    :param obj_b: Second object AABB
    :param voxel_size: The resolution of the voxel grid
    :return: Tuple of (approximate_volume, list_of_voxel_points)
    
    In a real mesh scenario, we would check if voxel center is inside Mesh A AND Mesh B.
    Here, simplified to just AABB check (which is trivially true inside the intersection AABB).
    """
    inter_aabb = obj_a.intersection_aabb(obj_b)
    if inter_aabb is None:
        return 0, []

    # Create grid points within the intersection AABB
    # We add a small epsilon to avoid boundary floating point issues
    x_range = np.arange(inter_aabb.min_point[0] + voxel_size/2, inter_aabb.max_point[0], voxel_size)
    y_range = np.arange(inter_aabb.min_point[1] + voxel_size/2, inter_aabb.max_point[1], voxel_size)
    z_range = np.arange(inter_aabb.min_point[2] + voxel_size/2, inter_aabb.max_point[2], voxel_size)

    if len(x_range) == 0 or len(y_range) == 0 or len(z_range) == 0:
        return 0, []

    # Create 3D meshgrid of points
    xv, yv, zv = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    voxel_centers = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)

    # In a real scenario, this is where we check: Mask = IsInside(MeshA) & IsInside(MeshB)
    # For AABB prototype, all points inside 'inter_aabb' are virtually colliding.
    active_voxels = voxel_centers # All valid
    
    approx_volume = len(active_voxels) * (voxel_size ** 3)
    return approx_volume, active_voxels

# --- 4. Scene Generation & Main Loop ---

def generate_scene(num_objects=50, world_size=100.0) -> List[AABB]:
    """
    Generates a random scene with scattered AABB objects.
    Also adds two manually placed overlapping objects for verification.
    """
    objects = []
    rng = np.random.default_rng(42) # Fixed seed for reproducibility

    for i in range(num_objects):
        # Random position
        pos = rng.uniform(0, world_size, 3)
        # Random size (min 1, max 10)
        size = rng.uniform(1, 10, 3)
        
        objects.append(AABB(
            id=i, 
            min_point=pos, 
            max_point=pos + size
        ))
    
    # Manually add a guaranteed collision pair for verification
    objects.append(AABB(id=998, min_point=np.array([10, 10, 10]), max_point=np.array([20, 20, 20])))
    objects.append(AABB(id=999, min_point=np.array([15, 10, 10]), max_point=np.array([25, 20, 20])))
    # Intersection volume should be (20-15) * (20-10) * (20-10) = 5 * 10 * 10 = 500
    
    return objects

def main():
    """
    Main execution pipeline:
    1. Generate Scene
    2. Run Broad Phase (Spatial Hash) to get candidates
    3. Run Narrow Phase (Voxelized Check) on candidates
    4. Report results
    """
    print("--- 1. Generating Scene ---")
    objects = generate_scene(num_objects=500, world_size=200.0)
    obj_map = {obj.id: obj for obj in objects}
    print(f"Generated {len(objects)} objects.")

    print("\n--- 2. Broad Phase (Spatial Hash) ---")
    start_time = time.time()
    
    # Cell size heuristic: roughly average object size * 2
    spatial = SpatialHash(cell_size=20.0)
    for obj in objects:
        spatial.insert(obj)
        
    candidates = spatial.get_candidates()
    broad_time = time.time() - start_time
    print(f"Broad Key Construction time: {broad_time:.4f}s")
    print(f"Found {len(candidates)} candidate pairs.")

    print("\n--- 3. Narrow Phase (AABB Check + Voxelization) ---")
    start_time = time.time()
    
    confirmed_collisions = []
    
    for id_a, id_b in candidates:
        obj_a = obj_map[id_a]
        obj_b = obj_map[id_b]
        
        # Double check actual AABB intersection (Spatial hash is coarse)
        if obj_a.intersects(obj_b):
            # Perform "Expensive" voxel analysis
            vol, voxels = analyze_interference_voxelized(obj_a, obj_b, voxel_size=1.0)
            if vol > 0:
                confirmed_collisions.append((id_a, id_b, vol, len(voxels)))

    narrow_time = time.time() - start_time
    print(f"Narrow Phase time: {narrow_time:.4f}s")
    print(f"Confirmed {len(confirmed_collisions)} collisions.")

    print("\n--- 4. Report ---")
    print(f"{'Obj A':<6} | {'Obj B':<6} | {'Inter Vol':<10} | {'Voxel Count':<10}")
    print("-" * 45)
    for c in confirmed_collisions[:10]: # Print first 10
        print(f"{c[0]:<6} | {c[1]:<6} | {c[2]:<10.2f} | {c[3]:<10}")
    
    if len(confirmed_collisions) > 10:
        print(f"... and {len(confirmed_collisions) - 10} more.")

    # Verify our manual case (998 vs 999)
    manual_case = next((x for x in confirmed_collisions if x[0] == 998 and x[1] == 999), None)
    if manual_case:
        print(f"\n[Verification] Manual pair (998, 999) found! Volume: {manual_case[2]:.2f} (Expected ~500.0)")
    else:
        print("\n[Verification] FAILED: Manual pair (998, 999) NOT found.")

if __name__ == "__main__":
    main()
