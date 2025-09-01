#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fisher Map Direction Analyzer (Refactor)
- 与模拟器相同的可视化控制参数：--headless / --realtime / --steps / --world-size
- 计算与显示解耦；优雅、简洁
"""

import os
import math
import time
import argparse
import random
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import cv2

# 导入更新后的模拟器组件
from aag_slam_simulator import RobotCore, RobotRenderer


# ------------------------------ Math Utils ------------------------------ #

def angnorm_deg(a: float) -> float:
    """归一化到 [-180, 180)"""
    return (a + 180.0) % 360.0 - 180.0


def angdiff_deg(a: float, b: float) -> float:
    """环向角度差"""
    return abs(angnorm_deg(a - b))


# ---------------------------- Data Structures --------------------------- #

@dataclass
class FisherDirectionInfo:
    angle: float                 # 方向角度 [0, 360)
    strength: float              # 对应强度
    confidence: float            # 置信度 [0, 1]
    center: Tuple[int, int]      # 中心点（局部地图坐标）


# --------------------------- Core: Analyzer ----------------------------- #

class FisherMapAnalyzer:
    """
    用方向扇区积分扫描，找 Fisher map 的主/次方向（以机器人为中心的局部图）
    """
    def __init__(self,
                 threshold_ratio: float = 0.2,
                 min_points: int = 15,
                 fov_angle: float = 90.0,
                 angle_step: int = 5,
                 sector_width: float = 30.0,
                 min_radius_px: int = 5):
        self.threshold_ratio = float(threshold_ratio)
        self.min_points = int(min_points)
        self.fov_angle = float(fov_angle)
        self.angle_step = int(angle_step)
        self.sector_width = float(sector_width)   # 方向扇区总宽度（度）
        self.min_radius_px = int(min_radius_px)   # 忽略离中心太近的点（避免“站在原地”的噪声）

    def analyze(self, fmap: np.ndarray) -> Tuple[Optional[FisherDirectionInfo], Optional[FisherDirectionInfo]]:
        pts = self._extract_points(fmap)
        if len(pts) < self.min_points:
            return None, None

        center = (fmap.shape[1] // 2, fmap.shape[0] // 2)
        ranked = self._scan_directions(fmap, center)

        if not ranked:
            return None, None

        angle1, s1 = ranked[0]
        primary = FisherDirectionInfo(angle=float(angle1), strength=float(s1),
                                      confidence=1.0, center=center)

        # 找与主方向分离 > FOV 的次方向
        secondary = None
        sep_needed = self.fov_angle + 5.0
        for a2, s2 in ranked[1:]:
            if angdiff_deg(a2, angle1) >= sep_needed:
                conf = (s2 / s1) if s1 > 1e-9 else 0.0
                secondary = FisherDirectionInfo(angle=float(a2), strength=float(s2),
                                                confidence=float(conf), center=center)
                break
        return primary, secondary

    # ---- internals ---- #

    def _extract_points(self, fmap: np.ndarray) -> np.ndarray:
        vmax = float(np.max(fmap))
        if vmax <= 0.0:
            return np.empty((0, 2), dtype=np.int32)

        thr = vmax * self.threshold_ratio
        ys, xs = np.where(fmap > thr)
        if xs.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        # 简洁：仅用坐标（不做重复加权，以免人为放大块状区域）
        return np.stack([xs, ys], axis=1)

    def _scan_directions(self, fmap: np.ndarray, center: Tuple[int, int]) -> List[Tuple[float, float]]:
        h, w = fmap.shape
        cx, cy = center

        # 预计算网格
        y, x = np.mgrid[0:h, 0:w]
        dx = x - cx
        dy = y - cy
        ang = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        dist = np.hypot(dx, dy)

        max_r = min(w, h) // 2
        base_mask = (dist > self.min_radius_px) & (dist < max_r)
        values = fmap

        ranked: List[Tuple[float, float]] = []
        half = self.sector_width / 2.0

        for a in range(0, 360, self.angle_step):
            a0 = (a - half) % 360.0
            a1 = (a + half) % 360.0
            if a0 <= a1:
                m = (ang >= a0) & (ang <= a1)
            else:
                # 跨 0/360
                m = (ang >= a0) | (ang <= a1)

            mask = base_mask & m
            if not np.any(mask):
                ranked.append((float(a), 0.0))
                continue

            # 距离越近权重越大：1/(d+1)
            dsel = dist[mask]
            wsel = 1.0 / (dsel + 1.0)
            vsel = values[mask]
            strength = float(np.sum(vsel * wsel) / np.sum(wsel))
            ranked.append((float(a), strength))

        ranked.sort(key=lambda t: t[1], reverse=True)
        return ranked


# --------------------------- View: Visualization ------------------------ #

def visualize_directions(fmap: np.ndarray,
                         primary: Optional[FisherDirectionInfo],
                         secondary: Optional[FisherDirectionInfo],
                         show: bool,
                         win_name: str = "Fisher Directions",
                         out_size: int = 600) -> np.ndarray:
    vmax = float(np.max(fmap))
    norm = ((fmap / vmax) * 255.0).astype(np.uint8) if vmax > 0 else np.zeros_like(fmap, dtype=np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    disp = cv2.resize(color, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    scale = out_size / float(fmap.shape[0])  # 假设正方形

    def _draw_dir(info: FisherDirectionInfo, length_px: int, bgr: Tuple[int, int, int], thick: int):
        cx = int(info.center[0] * scale)
        cy = int(info.center[1] * scale)
        ex = int(cx + length_px * math.cos(math.radians(info.angle)))
        ey = int(cy + length_px * math.sin(math.radians(info.angle)))
        cv2.arrowedLine(disp, (cx, cy), (ex, ey), bgr, thick, tipLength=0.15)
        cv2.circle(disp, (cx, cy), max(2, int(3 * scale)), (255, 255, 255), -1)

    if primary:
        _draw_dir(primary, int(40 * scale), (255, 0, 0), 3)
        cv2.putText(disp, f"P {primary.angle:.0f}deg ({primary.strength:.2f})",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    if secondary:
        _draw_dir(secondary, int(30 * scale), (0, 255, 0), 2)
        cv2.putText(disp, f"S {secondary.angle:.0f}deg ({secondary.strength:.2f})",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if show:
        cv2.imshow(win_name, disp)
        cv2.waitKey(1)
    return disp


# -------------------------------  Main  -------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Fisher Map Direction Analyzer")
    # 与模拟器一致的核心参数
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no visualization)')
    parser.add_argument('--realtime', action='store_true', help='Run at real-time speed (otherwise run as fast as possible)')
    parser.add_argument('--steps', type=int, default=1000, help='Number of simulation steps to run (default: 1000)')
    parser.add_argument('--world-size', type=float, default=40.0, help='World size in meters (default: 40.0)')

    # 分析器可选参数（不影响可视化开关）
    parser.add_argument('--fov-angle', type=float, default=90.0, help='Analyzer FOV angle for direction separation')
    parser.add_argument('--angle-step', type=int, default=5, help='Directional scan angle step (deg)')
    parser.add_argument('--sector-width', type=float, default=30.0, help='Sector width (deg) around direction')
    parser.add_argument('--analyze-every', type=int, default=5, help='Analyze every N steps')
    parser.add_argument('--display-every', type=int, default=1, help='Display every N analyses')
    parser.add_argument('--save-every', type=int, default=10, help='Save image every N analyses')
    parser.add_argument('--save-dir', type=str, default='fisher_analysis_plots', help='Directory to save images')

    args = parser.parse_args()

    # 可视化开关与模拟器一致
    show_display = not args.headless
    os.makedirs(args.save_dir, exist_ok=True)

    print("Fisher Map Analyzer - Starting...")
    print(f"  - Headless mode : {args.headless}")
    print(f"  - Real-time     : {args.realtime}")
    print(f"  - Steps         : {args.steps}")
    print(f"  - World size    : {args.world_size} m")
    print(f"  - Save dir      : {args.save_dir}")

    # 创建模拟器核心和渲染器
    core = RobotCore(
        world_width=args.world_size,
        world_height=args.world_size,
        fov_angle=args.fov_angle,
        control_frequency=5.0
    )
    
    if show_display:
        renderer = RobotRenderer(core, render_mode="human")
    else:
        renderer = None

    analyzer = FisherMapAnalyzer(
        threshold_ratio=0.2,
        min_points=15,
        fov_angle=float(args.fov_angle),
        angle_step=int(args.angle_step),
        sector_width=float(args.sector_width),
        min_radius_px=5
    )

    # 重置环境
    core.reset(regenerate_map=True)
    print("Simulation running... Press Ctrl+C to stop")

    # 计时控制（与模拟器风格一致）
    start_wall = time.time()
    expected_sim_t = 0.0

    analysis_count = 0
    last_gaze_update = 0.0

    try:
        for step in range(args.steps):
            # 速度控制策略
            if step % 50 == 0:
                core.set_velocity(
                    float(np.random.uniform(-2.0, 3.0)),
                    float(np.random.uniform(-0.8, 0.8))
                )
            
            # 凝视控制策略
            if core.sim_time - last_gaze_update >= 1.5:
                gaze_angle = np.random.uniform(0.0, 360.0)
                core.set_gaze(gaze_angle)
                last_gaze_update = core.sim_time

            # 执行仿真步骤
            core.step()
            core.update_maps()

            # 进度提示
            if step % 5 == 0:
                state = core.state()
                pos = state['position']
                gza = state['gaze_angle']
                fmap_stats = core.fisher_map_stats()
                print(f"\rStep {step:4d} | Pos:[{pos[0]:6.1f},{pos[1]:6.1f}] | "
                      f"Gaze:{gza:6.1f}° | Fisher features:{int(fmap_stats['total_features'])}", end="")

            # 分析与可视化
            if step % args.analyze_every == 0:
                fmap = core.feature_map
                primary, secondary = analyzer.analyze(fmap)
                analysis_count += 1

                if primary:
                    print(f"\n[ANALYSIS] step={step:4d}  "
                          f"P={primary.angle:6.1f}°({primary.strength:.2f})  "
                          f"S={secondary.angle:6.1f}°({secondary.strength:.2f})"
                          if secondary else
                          f"\n[ANALYSIS] step={step:4d}  P={primary.angle:6.1f}°({primary.strength:.2f})  S=None")

                    do_show = show_display and (analysis_count % args.display_every == 0)
                    img = visualize_directions(fmap, primary, secondary, show=do_show)

                    if analysis_count % args.save_every == 0:
                        out_path = os.path.join(args.save_dir, f"fisher_analysis_step_{step:04d}.png")
                        cv2.imwrite(out_path, img)
                        print(f"[SAVE] {out_path}")
                else:
                    print(f"\n[ANALYSIS] step={step:4d}  insufficient Fisher data")

            # 渲染
            if renderer:
                renderer.render()

            # 实时节拍
            if args.realtime:
                expected_sim_t += core.dt
                now = time.time() - start_wall
                sleep_t = expected_sim_t - now
                if sleep_t > 0:
                    time.sleep(sleep_t)
            else:
                # 非实时：轻微睡眠便于 UI
                if show_display:
                    time.sleep(0.01)

            # UI 退出
            if show_display:
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q') or k == 27:
                    print("\nUser requested quit")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if show_display:
            cv2.destroyAllWindows()
        print("\nFisher Map Analyzer - Completed")


if __name__ == "__main__":
    main()
