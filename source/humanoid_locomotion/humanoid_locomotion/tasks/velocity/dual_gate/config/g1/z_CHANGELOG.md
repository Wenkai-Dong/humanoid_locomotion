# DualGate-Attention-G1 Experiment Changelog

> This file documents all configuration changes across different versions of the
> `DualGate-Attention-G1` environment, including terrain, MDP, rewards, and any
> other modifications.
>
> **File location reference:**
> - Environment config: `dual_gate/config/g1/attention_env_cfg.py`
> - Terrain configs: `dual_gate/terrains/config/attention*.py`
> - Custom height-field functions: `dual_gate/terrains/height_field/hf_terrains.py`
> - Custom trimesh functions: `dual_gate/terrains/trimesh/mesh_terrains.py`

---

## Version Details

### v0 — Baseline (`DualGate-Attention-G1-v0`)

- **Env config class:** `G1AttentionEnvCfg` → inherits `G1VelocityRoughEnvCfg`
- **Terrain config:** `ATTENTION_TERRAINS_CFG` (`terrains/config/attention.py`)
- **MDP / Rewards / Commands:** Inherited from `G1VelocityRoughEnvCfg`, no modifications.
- **Notes:** Baseline version. All sub-terrain types use default Isaac Lab generation
  functions. Obstacle heights are mostly fixed and do not scale with difficulty.
  `platform_width` is set to `2.0` across all terrain types.

---

### v1 — Height Curriculum + Difficulty Increase (`DualGate-Attention-G1-v1`)

- **Env config class:** `G1AttentionEnvCfgv1` → inherits `G1VelocityRoughEnvCfg`
- **Terrain config:** `ATTENTION_TERRAINS_CFGv1` (`terrains/config/attention_v1.py`)
- **MDP / Rewards / Commands:** No changes compared to v0.

#### Changes from v0

- **GridStones:** Replaced the default Isaac Lab `HfSteppingStonesTerrainCfg` function
  with the custom `hf_terrains.stepping_stones_height_terrain`. Changed `stone_height_max`
  from a fixed scalar to a tuple range so that stone height now scales with difficulty
  (height curriculum). Also adjusted `stone_width_range`, `stone_distance_range`, and
  reduced `platform_width`.
- **Pallets:** Changed `pallet_height` from a fixed value to a difficulty-scaled range
  (height curriculum). Also adjusted `pallet_width_range`, `pallet_distance_range`, and
  reduced `platform_width`.
- **Gaps:** Increased the minimum gap width.
- **Pits:** Reduced `platform_width`.
- **PitsInverted:** Reduced `platform_width`.
- **Beams:** Increased `bar_height_range` and reduced `platform_width`.
- **boxes:** Increased the maximum `grid_height_range`.
- **Rough / Stairs / StairsInverted:** No changes.
- **Global `TerrainGeneratorCfg`:** No changes.

---
---

### v2 — Trimesh Terrain Overhaul + New Sub-Terrains (`DualGate-Attention-G1-v2`)

- **Env config class:** `G1AttentionEnvCfgv2` → inherits `G1VelocityRoughEnvCfg`
- **Terrain config:** `ATTENTION_TERRAINS_CFGv2` (`terrains/config/attention_v2.py`)
- **MDP / Rewards / Commands:** No changes compared to v1.
- **Note:** `G1AttentionEnvCfgv2` has **no** registered `_PLAY` or `_EVAL` variant yet.

#### Changes from v1

- **GridStones:** Replaced the height-field based `HfSteppingStonesTerrainCfg` with the
  custom trimesh-based `MeshSteppingStonesTerrainCfg`. This switches the generation
  backend from height-field to trimesh, enabling per-stone rotation via the new
  `max_yx_angle` parameter (set to `False` here, i.e. no rotation). Reduced `proportion`
  and decreased the lower bound of `stone_width_range`. Kept `platform_width` at `1.0`.
- **GridStonesAngle (NEW):** Added a new sub-terrain type that is identical to GridStones
  but with `max_yx_angle=(0, 10)` enabled, which rotates individual stones by a random
  angle that scales with difficulty. This introduces tilted stepping stones as a new
  challenge.
- **Pallets → Pallets_mesh (renamed + backend change):** Replaced the height-field based
  `HfPalletsTerrainCfg` with the custom trimesh-based `MeshPalletsTerrainCfg`. The key
  name changed from `"Pallets"` to `"Pallets_mesh"`. Also decreased the lower bounds of
  `pallet_width_range` and `pallet_distance_range` to create tighter and narrower pallets
  at higher difficulty.
- **Gaps:** Increased the maximum `gap_width_range` upper bound.
- **Pits:** Increased the maximum `pit_depth_range` upper bound.
- **PitsInverted:** Increased the maximum `box_height_range` upper bound.
- **Rough / Stairs / StairsInverted / Beams / boxes:** No changes.
- **Global `TerrainGeneratorCfg`:** No changes.
- **Sub-terrain proportions redistributed:** With the addition of `GridStonesAngle`,
  GridStones proportion was reduced and the new type took its share. The total still sums
  to approximately 1.2 (same as v1).

---

## Summary Comparison Table (continued)

### Terrain Sub-Type: GridStones

| Version | Config Class | `function` | `stone_height_max` | `stone_width_range` | `stone_distance_range` | `platform_width` | `max_yx_angle` | `proportion` |
|:--------|:-------------|:-----------|:-------------------:|:-------------------:|:----------------------:|:-----------------:|:--------------:|:------------:|
| **v0** | `terrain_gen.HfSteppingStonesTerrainCfg` | *(default)* | `0.15` | `(0.31, 0.5)` | `(0.05, 0.3)` | `2.0` | N/A | `0.2` |
| **v1** | `terrain_gen.HfSteppingStonesTerrainCfg` | `hf_terrains.stepping_stones_height_terrain` | `(0, 0.15)` | `(0.3, 0.5)` | `(0.1, 0.3)` | `1.0` | N/A | `0.2` |
| **v2** | `dual_gate_terrain_gen.MeshSteppingStonesTerrainCfg` | *(class default: trimesh)* | `(0, 0.15)` | `(0.15, 0.5)` | `(0.1, 0.3)` | `1.0` | `False` | `0.1` |

### Terrain Sub-Type: GridStonesAngle *(new in v2)*

| Version | Config Class | `stone_height_max` | `stone_width_range` | `stone_distance_range` | `platform_width` | `max_yx_angle` | `proportion` |
|:--------|:-------------|:-------------------:|:-------------------:|:----------------------:|:-----------------:|:--------------:|:------------:|
| **v0** | — | — | — | — | — | — | — |
| **v1** | — | — | — | — | — | — | — |
| **v2** | `dual_gate_terrain_gen.MeshSteppingStonesTerrainCfg` | `(0, 0.15)` | `(0.15, 0.5)` | `(0.1, 0.3)` | `1.0` | `(0, 10)` | `0.1` |

### Terrain Sub-Type: Pallets / Pallets_mesh

| Version | Key Name | Config Class | `pallet_height` | `pallet_width_range` | `pallet_distance_range` | `platform_width` | `proportion` |
|:--------|:---------|:-------------|:---------------:|:--------------------:|:-----------------------:|:-----------------:|:------------:|
| **v0** | `Pallets` | `dual_gate_terrain_gen.HfPalletsTerrainCfg` | `(0.03, 0.03)` | `(0.31, 0.5)` | `(0.09, 0.5)` | `2.0` | `0.2` |
| **v1** | `Pallets` | `dual_gate_terrain_gen.HfPalletsTerrainCfg` | `(0.0, 0.2)` | `(0.3, 0.5)` | `(0.1, 0.5)` | `1.0` | `0.2` |
| **v2** | `Pallets_mesh` | `dual_gate_terrain_gen.MeshPalletsTerrainCfg` | `(0.0, 0.2)` | `(0.15, 0.5)` | `(0.05, 0.5)` | `1.0` | `0.2` |

### Terrain Sub-Type: Gaps

| Version | `gap_width_range` | `platform_width` | `proportion` |
|:--------|:-----------------:|:-----------------:|:------------:|
| **v0** | `(0.01, 0.6)` | `2.0` | `0.1` |
| **v1** | `(0.05, 0.6)` | — | — |
| **v2** | `(0.05, 0.7)` | — | — |

### Terrain Sub-Type: Pits

| Version | `pit_depth_range` | `platform_width` | `proportion` |
|:--------|:-----------------:|:-----------------:|:------------:|
| **v0** | `(0.05, 0.45)` | `2.0` | `0.1` |
| **v1** | — | `1.0` | — |
| **v2** | `(0.05, 0.55)` | — | — |

### Terrain Sub-Type: PitsInverted

| Version | `box_height_range` | `platform_width` | `proportion` |
|:--------|:------------------:|:-----------------:|:------------:|
| **v0** | `(0.05, 0.6)` | `2.0` | `0.1` |
| **v1** | — | `1.0` | — |
| **v2** | `(0.05, 0.7)` | — | — |

### Terrain Sub-Type: Beams

| Version | `bar_height_range` | `platform_width` | `proportion` |
|:--------|:------------------:|:-----------------:|:------------:|
| **v0** | `(5.0, 5.0)` | `2.0` | `0.1` |
| **v1** | `(10.0, 10.0)` | `1.0` | — |
| **v2** | — | — | — |

### Terrain Sub-Type: boxes

| Version | `grid_height_range` | `platform_width` | `proportion` |
|:--------|:-------------------:|:-----------------:|:------------:|
| **v0** | `(0.05, 0.2)` | `2.0` | `0.1` |
| **v1** | `(0.05, 0.3)` | — | — |
| **v2** | — | — | — |

### Environment Config (Non-Terrain)

| Version | Env Config Class | Terrain Config Used | MDP Changes | Reward Changes | Command Changes | PLAY/EVAL Variants |
|:--------|:-----------------|:--------------------|:-----------:|:--------------:|:---------------:|:------------------:|
| **v0** | `G1AttentionEnvCfg` | `ATTENTION_TERRAINS_CFG` | None | None | None | Yes |
| **v1** | `G1AttentionEnvCfgv1` | `ATTENTION_TERRAINS_CFGv1` | None | None | None | Yes |
| **v2** | `G1AttentionEnvCfgv2` | `ATTENTION_TERRAINS_CFGv2` | None | None | None | No |