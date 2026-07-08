from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaac_neuromeka.terrain.mesh_terrain_importer import MeshTerrainImporter


@configclass
class MeshTerrainImporterCfg(TerrainImporterCfg):
    class_type = MeshTerrainImporter
    terrain_type = "usd"

    # relative path to the dir containing the terrain mesh obj file
    obj_dir: str = "isaac_neuromeka/assets/terrain_meshes/demo0"

    # origin sampling parameters
    node_density: float = 1.0  # number of nodes per m^2
    gravity_alignment_threshold: float = 0.785  # radians
    flat_patch_radius: float | list[float] = [0.5, 0.75]  # radius of flat patches to be considered
    flat_height_range: tuple[float, float] = (-0.1, 0.1)  # height range for flat patches
