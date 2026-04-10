from __future__ import annotations
from shapely.geometry import box
import numpy as np
import trimesh
from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from typing import TYPE_CHECKING
from random import randint
import scipy.interpolate as interpolate

if TYPE_CHECKING:
    from . import ame_1_terrains_cfg


def mesh_concentric_beams_terrain(
        difficulty: float, cfg: ame_1_terrains_cfg.MeshConcentricBeamsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """
    生成同心回字形横梁地形。
    修复点：
    1. 自动根据当前宽度和间距计算能放下多少圈。
    2. 保证中心平台始终存在。
    3. 自动填充外围安全地面。
    """
    # 1. 解析课程难度参数
    gap = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    width = cfg.beam_width_range[0] + difficulty * (cfg.beam_width_range[1] - cfg.beam_width_range[0])
    noise_mag = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # 2. 基础常量计算
    # 地形半长（正方形的一半）
    terrain_half_size = min(cfg.size[0], cfg.size[1]) / 2
    # 安全边距：最外圈保留 1.0 米平地，防止机器人从最后一关掉出地图
    safety_margin = 1.0

    # 3. 核心修复：显式计算能放下多少个回字形 (num_rings)
    # 可用半径 = 总半径 - 安全边距 - 中心平台半径
    available_radius = terrain_half_size - safety_margin - (cfg.platform_width / 2)
    # 单个回字形占用的径向距离 = 缝隙 + 梁宽
    unit_thickness = gap + width

    # 向下取整，算出最多能放几圈
    if unit_thickness > 0.001 and available_radius > 0:
        num_rings = int(available_radius / unit_thickness)
    else:
        num_rings = 0

    # 初始化网格列表
    meshes_list = list()

    # ==========================================
    # 4. 生成中心平台 (Spawn Point) - 保证始终存在
    # ==========================================
    # 为了防止 Z-fighting 或方便行走，中心平台高度设为平地高度
    # 假设平地高度为 -half_thickness (表面在 0)
    # 或者为了与横梁对齐，我们把所有物体基准设为 Z=0 表面

    center_pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * cfg.beam_thickness)
    center_mesh = trimesh.creation.box(
        (cfg.platform_width, cfg.platform_width, cfg.beam_thickness),
        trimesh.transformations.translation_matrix(center_pos)
    )
    meshes_list.append(center_mesh)

    # ==========================================
    # 5. 循环生成回字形横梁
    # ==========================================
    # 初始内边长 = 中心平台 + 两侧第一个缝隙
    current_inner_side = cfg.platform_width + 2 * gap

    for i in range(num_rings):
        # 当前环的外边长 = 内边长 + 两侧梁宽
        current_outer_side = current_inner_side + 2 * width

        # 使用 Shapely 制作回字形截面 (大正方形 减去 小正方形)
        outer_rect = box(-current_outer_side / 2, -current_outer_side / 2, current_outer_side / 2,
                         current_outer_side / 2)
        inner_rect = box(-current_inner_side / 2, -current_inner_side / 2, current_inner_side / 2,
                         current_inner_side / 2)
        beam_2d = outer_rect.difference(inner_rect)

        # 计算随机高度 (相对于 Z=0 平面的偏移)
        z_noise = np.random.uniform(-noise_mag, noise_mag)

        # 挤出并平移
        # 注意：box生成是以中心为基准的，extrude是从0向上挤出的
        # 我们希望 Mesh 的顶面位于 z_noise
        beam_mesh = trimesh.creation.extrude_polygon(beam_2d, height=cfg.beam_thickness)

        # 平移到地形中心 + 高度调整
        # extrude 出来的物体底面在 Z=0，顶面在 Z=thickness
        # 我们要让顶面变成 z_noise，所以整体向下移 thickness，再加 z_noise
        beam_pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], z_noise - cfg.beam_thickness)
        beam_mesh.apply_transform(trimesh.transformations.translation_matrix(beam_pos))

        meshes_list.append(beam_mesh)

        # 更新下一圈的内边长 = 当前外边长 + 两侧缝隙
        current_inner_side = current_outer_side + 2 * gap

    # ==========================================
    # 6. 生成外围安全地面 (Safety Border)
    # ==========================================
    # 我们用一个巨大的地形块矩形，减去中间已经被占用的区域（最后一圈横梁 + 最后的缝隙）
    # 中间被挖空的区域边长：
    hole_side = current_inner_side  # 注意：循环结束后 current_inner_side 已经加上了最后的 gap

    total_rect = box(-cfg.size[0] / 2, -cfg.size[1] / 2, cfg.size[0] / 2, cfg.size[1] / 2)
    # 限制挖空区域不要超过地形总大小 (虽然逻辑上不应该发生)
    hole_side = min(hole_side, min(cfg.size[0], cfg.size[1]))
    hole_rect = box(-hole_side / 2, -hole_side / 2, hole_side / 2, hole_side / 2)

    border_2d = total_rect.difference(hole_rect)

    if not border_2d.is_empty:
        # 地面高度设为 0 (标准平地)
        border_pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -cfg.beam_thickness)  # 顶面在 0
        border_mesh = trimesh.creation.extrude_polygon(border_2d, height=cfg.beam_thickness)
        border_mesh.apply_transform(trimesh.transformations.translation_matrix(border_pos))
        meshes_list.append(border_mesh)

    # ==========================================
    # 7. 设置地形原点
    # ==========================================
    # 机器人出生在中心平台上方 0.1m 处
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0 + 0.1])

    return meshes_list, origin


def pyramid_stairs_terrain(
    difficulty: float, cfg: ame_1_terrains_cfg.MeshPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`cfg.platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.

    .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))

    # initialize list of meshes
    meshes_list = list()

    # generate the border if needed
    if cfg.border_width > 0.0:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * cfg.step_width, terrain_size[1] - 2 * k * cfg.step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] + k * step_height / 2.0
        box_offset = (k + 0.5) * cfg.step_width
        # -- dimensions
        box_height = (k + 2) * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], cfg.step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        if cfg.holes:
            box_dims = (cfg.step_width, box_size[1], box_height)
        else:
            box_dims = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]

    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        (num_steps + 2) * step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] + num_steps * step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    # origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])

    return meshes_list, origin


def inverted_pyramid_stairs_terrain(
    difficulty: float, cfg: ame_1_terrains_cfg.MeshInvertedPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a inverted pyramid stair pattern.

    The terrain is an inverted pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`cfg.platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.

    .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))
    # total height of the terrain
    total_height = (num_steps + 1) * step_height

    # initialize list of meshes
    meshes_list = list()

    # generate the border if needed
    if cfg.border_width > 0.0:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * step_height]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders
    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * cfg.step_width, terrain_size[1] - 2 * k * cfg.step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
        box_offset = (k + 0.5) * cfg.step_width
        # -- dimensions
        box_height = total_height - (k + 1) * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], cfg.step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        if cfg.holes:
            box_dims = (cfg.step_width, box_size[1], box_height)
        else:
            box_dims = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]
    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    # origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height])

    return meshes_list, origin


@height_field_to_mesh
def single_column_stones_terrain(difficulty: float, cfg: ame_1_terrains_cfg.HfSteppingStonesTerrainCfg) -> np.ndarray:
    """Generate a terrain with a stepping stones pattern.

    The terrain is a stepping stones pattern which trims to a flat platform at the center of the terrain.

    .. image:: ../../_static/terrains/height_field/stepping_stones_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- stones
    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    # -- holes
    holes_depth = int(cfg.holes_depth / cfg.vertical_scale)
    # -- platform
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    # create range of heights
    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)

    # create a terrain with a flat platform at the center
    hf_raw = np.full((width_pixels, length_pixels), holes_depth)

    # 计算中心位置
    mid_x = width_pixels // 2
    mid_y = length_pixels // 2
    # 竖线在 X 轴的位置
    fixed_x_start = max(0, mid_x - stone_width // 2)
    fixed_x_end = min(width_pixels, fixed_x_start + stone_width)
    # 横线在 Y 轴的位置
    fixed_y_start = max(0, mid_y - stone_width // 2)
    fixed_y_end = min(length_pixels, fixed_y_start + stone_width)


    # add the stones
    start_x, start_y = 0, 0
    # -- if the terrain is longer than it is wide then fill the terrain column by column
    while start_y < length_pixels:
        # ensure that stone stops along y-axis
        stop_y = min(length_pixels, start_y + stone_width)
        # fill first stone
        hf_raw[fixed_x_start:fixed_x_end, start_y:stop_y] = np.random.choice(stone_height_range)
        # update x-position
        start_y += stone_width + stone_distance
    while start_x < width_pixels:
        # ensure that stone stops along x-axis
        stop_x = min(width_pixels, start_x + stone_width)
        # fill first stone
        hf_raw[start_x:stop_x, fixed_y_start:fixed_y_end] = np.random.choice(stone_height_range)
        # update x-position
        start_x += stone_width + stone_distance
    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)

@height_field_to_mesh
def narrow_pallets_terrain(difficulty: float, cfg: ame_1_terrains_cfg.HfNarrowPalletsTerrainCfg) -> np.ndarray:
    """Generate a terrain with a stepping stones pattern.

    The terrain is a stepping stones pattern which trims to a flat platform at the center of the terrain.

    .. image:: ../../_static/terrains/height_field/narrow_pallets_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    # 解析长度
    stone_length = cfg.stone_length_range[1] - difficulty * (cfg.stone_length_range[1] - cfg.stone_length_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- stones
    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_length = int(stone_length / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    # -- holes
    holes_depth = int(cfg.holes_depth / cfg.vertical_scale)
    # -- platform
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    # create range of heights
    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)

    # create a terrain with a flat platform at the center
    hf_raw = np.full((width_pixels, length_pixels), holes_depth)

    # 计算中心位置
    mid_x = width_pixels // 2
    mid_y = length_pixels // 2
    # 竖线在 X 轴的位置
    fixed_x_start = max(0, mid_x - stone_width // 2)
    fixed_x_end = min(width_pixels, fixed_x_start + stone_width)
    # 横线在 Y 轴的位置
    fixed_y_start = max(0, mid_y - stone_width // 2)
    fixed_y_end = min(length_pixels, fixed_y_start + stone_width)


    # add the stones
    start_x, start_y = 0, 0
    # -- if the terrain is longer than it is wide then fill the terrain column by column
    while start_y < length_pixels:
        # ensure that stone stops along y-axis
        stop_y = min(length_pixels, start_y + stone_length)
        # fill first stone
        hf_raw[fixed_x_start:fixed_x_end, start_y:stop_y] = np.random.choice(stone_height_range)
        # update x-position
        start_y += stone_length + stone_distance
    while start_x < width_pixels:
        # ensure that stone stops along x-axis
        stop_x = min(width_pixels, start_x + stone_length)
        # fill first stone
        hf_raw[start_x:stop_x, fixed_y_start:fixed_y_end] = np.random.choice(stone_height_range)
        # update x-position
        start_x += stone_length + stone_distance
    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)

@height_field_to_mesh
def random_uniform_difficulty_terrain(difficulty: float, cfg: ame_1_terrains_cfg.HfRandomUniformDifficultyTerrainCfg) -> np.ndarray:
    """Generate a terrain with height sampled uniformly from a specified range.

    .. image:: ../../_static/terrains/height_field/random_uniform_terrain.jpg
       :width: 40%
       :align: center

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the downsampled scale is smaller than the horizontal scale.
    """
    # check parameters
    # -- horizontal scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    return np.rint(z_upsampled * difficulty).astype(np.int16)

@height_field_to_mesh
def stones_bridge_terrain(difficulty: float, cfg: ame_1_terrains_cfg.HfStonesBridgeTerrainCfg) -> np.array:
    """Generate a terrain with stones bridge pattern.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_length = cfg.stone_length_range[1] - difficulty * (cfg.stone_length_range[1] - cfg.stone_length_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
            cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )
    stone_lateral_distance = cfg.stone_lateral_distance_range[0] + difficulty * (
        cfg.stone_lateral_distance_range[1] - cfg.stone_lateral_distance_range[0]
    )

    # switch parameters to discrete units
    # --terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # --stones
    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_lateral_distance = int(stone_lateral_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_length = int(stone_length / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    # --holes
    holes_depth = int(cfg.holes_depth / cfg.vertical_scale)
    # -- platform
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    # create range of heights
    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)

    # create a terrain with a flat platform at one side
    hf_raw = np.full((width_pixels, length_pixels), holes_depth)

    # add the stones
    start_x = stone_distance
    while start_x < width_pixels:
        # ensure that stones stops along x-axis
        stop_x = min(width_pixels, start_x + stone_width)
        # randomly sample x-position
        start_y = (length_pixels - stone_length) // 2 + np.random.choice([-stone_lateral_distance, stone_lateral_distance])
        stop_y = start_y + stone_length
        hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
        # update y-position
        start_x = stop_x + stone_distance
    start_y = stone_distance
    while start_y < length_pixels:
        # ensure that stones stops along y-axis
        stop_y = min(length_pixels, start_y + stone_width)
        # randomly sample x-position
        start_x = (width_pixels - stone_length) // 2 + np.random.choice([-stone_lateral_distance, stone_lateral_distance])
        stop_x = start_x + stone_length
        hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
        # update y-position
        start_y = stop_y + stone_distance

    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def double_column_stakes_terrain(
    difficulty: float, cfg: ame_1_terrains_cfg.HfDoubleColumnStakesTerrainCfg
) -> np.ndarray:
    """Generate a double-column stake heightfield extending along x/y directions."""

    # Interpolate parameters by difficulty
    stake_side = cfg.stake_side_range[1] - difficulty * (
        cfg.stake_side_range[1] - cfg.stake_side_range[0]
    )
    stake_gap = cfg.stake_gap_range[0] + difficulty * (
        cfg.stake_gap_range[1] - cfg.stake_gap_range[0]
    )
    column_gap = cfg.column_gap_range[0] + difficulty * (
        cfg.column_gap_range[1] - cfg.column_gap_range[0]
    )

    # Discretized grid parameters
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    stake_side_px = max(1, int(stake_side / cfg.horizontal_scale))
    stake_gap_px = max(0, int(stake_gap / cfg.horizontal_scale))
    column_gap_px = max(0, int(column_gap / cfg.horizontal_scale))
    column_jitter_px = max(0, int(cfg.column_jitter / cfg.horizontal_scale))

    stake_height_max_px = max(0, int(cfg.stake_height_max / cfg.vertical_scale))
    holes_depth_px = int(cfg.holes_depth / cfg.vertical_scale)

    platform_width_px = max(1, int(cfg.platform_width / cfg.horizontal_scale))

    hf_raw = np.full((width_pixels, length_pixels), holes_depth_px, dtype=float)
    half_lower = stake_side_px // 2
    half_upper = stake_side_px - half_lower
    center_offset_px = stake_side_px + column_gap_px

    center_x = width_pixels // 2
    center_y = length_pixels // 2

    rng = np.random.default_rng()
    stake_height_values = (
        np.arange(-stake_height_max_px, stake_height_max_px + 1)
        if stake_height_max_px > 0
        else np.array([0], dtype=int)
    )

    def paint_square(cx: int, cy: int, value: int) -> None:
        if cx < 0 or cx >= width_pixels or cy < 0 or cy >= length_pixels:
            return
        x1 = max(0, cx - half_lower)
        x2 = min(width_pixels, cx + half_upper)
        y1 = max(0, cy - half_lower)
        y2 = min(length_pixels, cy + half_upper)
        hf_raw[x1:x2, y1:y2] = value

    def place_column_pair(primary_pos: int, along_x: bool) -> None:
        if along_x:
            axis_limit_low = half_lower
            axis_limit_high = length_pixels - half_upper
            base_offset = max(center_offset_px // 2, half_lower)
            for sign in (-1, 1):
                jitter = (
                    rng.integers(-column_jitter_px, column_jitter_px + 1)
                    if column_jitter_px > 0
                    else 0
                )
                cy = int(np.clip(center_y + sign * base_offset + jitter, axis_limit_low, axis_limit_high))
                height_value = int(rng.choice(stake_height_values))
                paint_square(primary_pos, cy, height_value)
        else:
            axis_limit_low = half_lower
            axis_limit_high = width_pixels - half_upper
            base_offset = max(center_offset_px // 2, half_lower)
            for sign in (-1, 1):
                jitter = (
                    rng.integers(-column_jitter_px, column_jitter_px + 1)
                    if column_jitter_px > 0
                    else 0
                )
                cx = int(np.clip(center_x + sign * base_offset + jitter, axis_limit_low, axis_limit_high))
                height_value = int(rng.choice(stake_height_values))
                paint_square(cx, primary_pos, height_value)

    def extend_from_center(along_x: bool, direction: int) -> None:
        if along_x:
            start = center_x + direction * (half_upper + stake_gap_px + stake_side_px)
            step = (stake_gap_px + stake_side_px) * direction
            while 0 <= start < width_pixels:
                if not (half_lower <= start <= width_pixels - half_upper):
                    break
                place_column_pair(int(start), along_x=True)
                start += step
        else:
            start = center_y + direction * (half_upper + stake_gap_px + stake_side_px)
            step = (stake_gap_px + stake_side_px) * direction
            while 0 <= start < length_pixels:
                if not (half_lower <= start <= length_pixels - half_upper):
                    break
                place_column_pair(int(start), along_x=False)
                start += step

    def extend_from_edge(along_x: bool) -> None:
        start = 0
        step = stake_gap_px + stake_side_px
        while 0 <= start < width_pixels:
            place_column_pair(int(start), along_x)
            start += step


    # Extend along +x/-x
    # extend_from_center(along_x=True, direction=1)
    # extend_from_center(along_x=True, direction=-1)
    extend_from_edge(along_x=True)

    # Extend along +y/-y
    # extend_from_center(along_x=False, direction=1)
    # extend_from_center(along_x=False, direction=-1)
    extend_from_edge(along_x=False)

    # add the platform in the center
    x1 = (width_pixels - platform_width_px) // 2
    x2 = (width_pixels + platform_width_px) // 2
    y1 = (length_pixels - platform_width_px) // 2
    y2 = (length_pixels + platform_width_px) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def concentric_gap_terrain(difficulty: float, cfg: ame_1_terrains_cfg.HfConcentricGapTerrainCfg) -> np.ndarray:
    """
    Generate concentric gap terrain with a center platform.
    Gap width is difficulty-dependent and gap depth is fixed.
    """
    # Gap depth in pixels
    gap_depth = int(2.0 / cfg.vertical_scale)
    # Gap width varies with difficulty
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    gap_width = int(gap_width / cfg.horizontal_scale)
    # Ground width varies with difficulty (narrower for harder terrains)
    ground_width = cfg.ground_width_range[0] + (1.0 - difficulty) * (cfg.ground_width_range[1] - cfg.ground_width_range[0])
    ground_width = int(ground_width / cfg.horizontal_scale)
    # Ground height
    ground_height_max = int(cfg.ground_height_max / cfg.vertical_scale)
    # Terrain dimensions
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # Platform width
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels))
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    is_gap = True
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        if is_gap:
            # Fill gap ring
            hf_raw[start_x:stop_x, start_y:stop_y] = -gap_depth
            start_x += gap_width
            stop_x -= gap_width
            start_y += gap_width
            stop_y -= gap_width
        else:
            # Fill ground ring
            hf_raw[start_x:stop_x, start_y:stop_y] = randint(-ground_height_max, ground_height_max)
            start_x += ground_width
            stop_x -= ground_width
            start_y += ground_width
            stop_y -= ground_width
        is_gap = not is_gap
    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def alternate_column_stakes_terrain(
    difficulty: float, cfg: ame_1_terrains_cfg.HfDoubleColumnStakesTerrainCfg
) -> np.ndarray:
    """Generate alternating double-column stake terrain along x/y directions."""

    # Interpolate parameters by difficulty
    stake_side = cfg.stake_side_range[1] - difficulty * (
        cfg.stake_side_range[1] - cfg.stake_side_range[0]
    )
    stake_gap = cfg.stake_gap_range[0] + difficulty * (
        cfg.stake_gap_range[1] - cfg.stake_gap_range[0]
    )
    column_gap = cfg.column_gap_range[1] - difficulty * (
        cfg.column_gap_range[1] - cfg.column_gap_range[0]
    )

    # Discretized grid parameters
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    stake_side_px = max(1, int(stake_side / cfg.horizontal_scale))
    stake_gap_px = max(0, int(stake_gap / cfg.horizontal_scale))
    column_gap_px = max(0, int(column_gap / cfg.horizontal_scale))
    column_jitter_px = max(0, int(cfg.column_jitter / cfg.horizontal_scale))

    stake_height_max_px = max(0, int(cfg.stake_height_max / cfg.vertical_scale))
    holes_depth_px = int(cfg.holes_depth / cfg.vertical_scale)

    platform_width_px = max(1, int(cfg.platform_width / cfg.horizontal_scale))

    hf_raw = np.full((width_pixels, length_pixels), holes_depth_px, dtype=float)
    half_lower = stake_side_px // 2
    half_upper = stake_side_px - half_lower

    # Build a deterministic RNG for this sub-terrain when cfg.seed is provided.
    # We mix in quantized difficulty so each tile can still look different while
    # remaining reproducible across runs.
    if getattr(cfg, "seed", None) is not None:
        difficulty_key = int(round(float(difficulty) * 1_000_000.0))
        local_seed = (int(cfg.seed) * 1_000_003 + difficulty_key) % (2**32)
        rng = np.random.default_rng(local_seed)
    else:
        rng = np.random.default_rng()
    stake_height_values = (
        np.arange(-stake_height_max_px, stake_height_max_px + 1)
        if stake_height_max_px > 0
        else np.array([0], dtype=int)
    )

    def paint_square(cx: int, cy: int, value: int) -> None:
        if cx < 0 or cx >= width_pixels or cy < 0 or cy >= length_pixels:
            return
        x1 = max(0, cx - half_lower)
        x2 = min(width_pixels, cx + half_upper)
        y1 = max(0, cy - half_lower)
        y2 = min(length_pixels, cy + half_upper)
        hf_raw[x1:x2, y1:y2] = value

    def place_alternate_columns(start_pos: int, along_x: bool) -> None:
        offset = column_gap_px // 2  # Alternating offset
        step = stake_gap_px + stake_side_px
        while start_pos < (width_pixels if along_x else length_pixels):
            jitter = (
                rng.integers(-column_jitter_px, column_jitter_px + 1)
                if column_jitter_px > 0
                else 0
            )
            height_value = int(rng.choice(stake_height_values))

            if along_x:
                cx = start_pos
                cy = (length_pixels // 2) + offset + jitter
                paint_square(cx, cy, height_value)
            else:
                cy = start_pos
                cx = (width_pixels // 2) + offset + jitter
                paint_square(cx, cy, height_value)

            # Flip offset for alternating pattern
            offset = -offset
            start_pos += step

    # Place alternating columns along x and y
    place_alternate_columns(0, along_x=True)
    place_alternate_columns(0, along_x=False)

    # add the platform in the center
    x1 = (width_pixels - platform_width_px) // 2
    x2 = (width_pixels + platform_width_px) // 2
    y1 = (length_pixels - platform_width_px) // 2
    y2 = (length_pixels + platform_width_px) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)