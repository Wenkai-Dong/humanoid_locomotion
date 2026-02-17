from dataclasses import dataclass
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg


import humanoid_locomotion.tasks.velocity.ame_1.terrains as ame_1_terrains


@configclass
@dataclass
class MeshConcentricBeamsTerrainCfg(SubTerrainBaseCfg):
    """
    回字形横梁地形配置
    """
    function = ame_1_terrains.mesh_concentric_beams_terrain

    platform_width: float = 2.0  # 中心出生点平台的宽度
    beam_thickness: float = 3.0  # 地形厚度（Z轴）

    # --- 课程参数 ---
    # 难度 0 -> 1 变化时：
    step_height_range: tuple[float, float] = (0.0, 0.3)  # 高度噪声幅度
    beam_width_range: tuple[float, float] = (1.0, 0.2)  # 横梁宽度：从宽变窄
    gap_width_range: tuple[float, float] = (0.1, 0.4)  # 缝隙宽度：从窄变宽


@configclass
class MeshPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a pyramid stair mesh terrain."""

    function = ame_1_terrains.pyramid_stairs_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """

    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""

    step_width: float = MISSING
    """The width of the steps (in m)."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.
    """


@configclass
class MeshInvertedPyramidStairsTerrainCfg(MeshPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stair mesh terrain.

    Note:
        This is the same as :class:`MeshPyramidStairsTerrainCfg` except that the steps are inverted.
    """

    function = ame_1_terrains.inverted_pyramid_stairs_terrain


@configclass
class HfTerrainBaseCfg(SubTerrainBaseCfg):
    """The base configuration for height field terrains."""

    border_width: float = 0.0
    """The width of the border/padding around the terrain (in m). Defaults to 0.0.

    The border width is subtracted from the :obj:`size` of the terrain. If non-zero, it must be
    greater than or equal to the :obj:`horizontal scale`.
    """

    horizontal_scale: float = 0.1
    """The discretization of the terrain along the x and y axes (in m). Defaults to 0.1."""

    vertical_scale: float = 0.005
    """The discretization of the terrain along the z axis (in m). Defaults to 0.005."""

    slope_threshold: float | None = None
    """The slope threshold above which surfaces are made vertical. Defaults to None,
    in which case no correction is applied."""


@configclass
class HfSingleColumnStonesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a stepping stones height field terrain."""

    function = ame_1_terrains.single_column_stones_terrain

    stone_height_max: float = MISSING
    """The maximum height of the stones (in m)."""

    stone_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the stones (in m)."""

    stone_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum distance between stones (in m)."""

    holes_depth: float = -10.0
    """The depth of the holes (negative obstacles). Defaults to -10.0."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class HfNarrowPalletsTerrainCfg(HfSingleColumnStonesTerrainCfg):
    """Configuration for a narrow pallets height field terrain."""

    function = ame_1_terrains.narrow_pallets_terrain

    stone_length_range: tuple[float, float] = MISSING
    """The minimum and maximum length of the stones (in m)."""