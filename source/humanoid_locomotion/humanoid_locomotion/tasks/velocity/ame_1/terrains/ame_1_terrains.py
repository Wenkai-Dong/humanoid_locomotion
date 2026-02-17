from __future__ import annotations
from shapely.geometry import box
import numpy as np
import trimesh
from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from typing import TYPE_CHECKING

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