import numpy as np
import random
import math
import cv2
import time
import argparse
from typing import List, Tuple, Dict, Any, Optional


# ------------------------------- Utils -------------------------------- #

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def angnorm_deg(a):
    # [-180,180)
    return (a + 180.0) % 360.0 - 180.0

def angdiff_deg(a, b):
    return abs(angnorm_deg(a - b))


# ----------------------------- Core (Compute) -------------------------- #

class RobotCore:
    """
    Headless core: physics, FOV geometry, Fisher map (robot-centered),
    collision/stuck detection, obstacle generation. No rendering side-effects.
    """
    def __init__(self,
                 world_width=40.0,
                 world_height=40.0,
                 pixel_per_meter=20,
                 grid_size=0.5,
                 fov_angle=90,
                 fov_distance=12.5,
                 robot_size=0.5,
                 feature_map_size=100,
                 feature_map_resolution=0.25,
                 control_frequency=5.0):
        # World
        self.world_width = float(world_width)
        self.world_height = float(world_height)
        self.pixel_per_meter = int(pixel_per_meter)
        self.grid_size = float(grid_size)
        self.fov_angle = float(fov_angle)
        self.fov_distance = float(fov_distance)
        self.robot_size = float(robot_size)

        # Feature map (robot-centered local view)
        self.feature_map_size = int(feature_map_size)
        self.feature_map_resolution = float(feature_map_resolution)
        self.feature_map = np.zeros((self.feature_map_size, self.feature_map_size), dtype=np.float32)
        self.global_feature_map_size = int(max(self.world_width, self.world_height) * 2 / self.feature_map_resolution)
        self.global_feature_map = np.zeros((self.global_feature_map_size, self.global_feature_map_size), dtype=np.float32)

        # Time & control
        self.dt = 0.1
        self.sim_time = 0.0
        self.control_frequency = float(control_frequency)
        self.control_period = 1.0 / self.control_frequency if self.control_frequency > 0 else 0.0
        self.last_control_update = 0.0

        # Robot state
        self.robot_pos = np.array([self.world_width / 2, self.world_height / 2], dtype=np.float64)
        self.robot_angle = 0.0           # deg, body heading
        self.gaze_angle = 0.0            # deg, neck/eye
        self._target_gaze_angle = 0.0
        self._active_gaze_control = False

        # Velocities
        self.velocity = 0.0              # m/s
        self.angular_velocity = 0.0      # rad/s
        self.max_linear_velocity = 3.0
        self.max_angular_velocity = 1.0
        self.velocity_sampling_enabled = True
        self.velocity_sample_interval = 5.0
        self.last_velocity_sample_time = 0.0

        # Obstacles (world meters)
        self.obstacles = []
        self._have_map = False

        # Diagnostics
        self.step_counter = 0
        self._collision_occurred = False
        self._stuck_counter = 0

        # Cached conversions
        self.width = int(self.world_width * self.pixel_per_meter)
        self.height = int(self.world_height * self.pixel_per_meter)

    # -------- Public API (no rendering) -------- #

    def reset(self, regenerate_map=True, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if regenerate_map or not self._have_map:
            self._gen_obstacles(num_obstacles=20)
            self._have_map = True

        self.robot_pos = self._find_safe_start()
        self.robot_angle = float(random.randint(0, 360))
        # self.gaze_angle = 0.
        self.gaze_angle = float(random.randint(0, 360))
        self._target_gaze_angle = 0.0
        self._active_gaze_control = False

        self.velocity = 0.0
        self.angular_velocity = 0.0

        self.sim_time = 0.0
        self.last_control_update = 0.0
        self.last_velocity_sample_time = 0.0

        self.feature_map.fill(0.0)
        self.global_feature_map.fill(0.0)

        self._collision_occurred = False
        self._stuck_counter = 0
        self.step_counter = 0

    def set_velocity(self, linear_vel: float, angular_vel: float):
        self.velocity = clamp(float(linear_vel), -5.0, 5.0)
        self.angular_velocity = clamp(float(angular_vel), -1.5, 1.5)

    def set_gaze(self, gaze_angle_deg: float):
        self._target_gaze_angle = gaze_angle_deg % 360.0
        self._active_gaze_control = True

    def step(self):
        """One simulation step: time, control update, kinematics, collisions."""
        prev_pos = self.robot_pos.copy()
        self._collision_occurred = False

        # time
        self.sim_time += self.dt

        # control update cadence
        if self.control_period > 0 and (self.sim_time - self.last_control_update) >= self.control_period:
            self._maybe_sample_velocity()
            self.last_control_update = self.sim_time
        else:
            # keep legacy internal sampler working if user wants it independent of control_frequency
            self._maybe_sample_velocity()

        # update pose
        self._update_robot()

        # collision proxy: near-zero displacement
        moved = np.linalg.norm(self.robot_pos - prev_pos)
        if moved < 0.01:
            self._collision_occurred = True
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0

        self.step_counter += 1

    def update_maps(self):
        """Update Fisher maps AFTER pose update. No rendering here."""
        self._apply_feature_decay()
        self._detect_and_add_features_to_global_map()
        self._extract_local_feature_map()

    # Pure compute: FOV sampling points (world meters)
    def compute_fov_points_world(self) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        max_range = self.fov_distance
        ray_step = max(0.05, self.grid_size * 0.5)
        sample_step = max(self.grid_size, 0.2)

        half = int(self.fov_angle // 2)
        for off in range(-half, half + 1, 3):
            ang = math.radians((self.gaze_angle + off) % 360.0)
            # march ray
            hit_dist = 0.0
            d = 0.0
            while d < max_range:
                d += ray_step
                wx = self.robot_pos[0] + math.cos(ang) * d
                wy = self.robot_pos[1] + math.sin(ang) * d
                if not self._inside_world(wx, wy) or self._point_in_obstacle(wx, wy):
                    hit_dist = d
                    break
                hit_dist = d
            # sample along visible segment
            s = 0.0
            while s <= hit_dist:
                wx = self.robot_pos[0] + math.cos(ang) * s
                wy = self.robot_pos[1] + math.sin(ang) * s
                if self._inside_world(wx, wy):
                    pts.append((wx, wy))
                s += sample_step
        return pts

    def fisher_map_stats(self) -> Dict[str, float]:
        flat = self.feature_map.ravel()
        nz = flat[flat > 0]
        if nz.size == 0:
            return {'mean_fisher': 0.0, 'total_features': 0.0, 'density': 0.0}
        return {
            'mean_fisher': float(nz.mean()),
            'total_features': float(nz.size),
            'density': float(nz.size) / float(flat.size)
        }

    def fov_fisher_stats(self) -> Dict[str, float]:
        pts = self.compute_fov_points_world()
        if not pts:
            return {'mean_fisher': 0.0, 'total_features': 0.0, 'density': 0.0}

        vals = []
        half = self.feature_map_size // 2
        res = self.feature_map_resolution
        for wx, wy in pts:
            dx = (wx - self.robot_pos[0]) / res
            dy = (wy - self.robot_pos[1]) / res
            lx = int(round(dx)) + half
            ly = int(round(dy)) + half
            if 0 <= lx < self.feature_map_size and 0 <= ly < self.feature_map_size:
                vals.append(self.feature_map[ly, lx])

        if not vals:
            return {'mean_fisher': 0.0, 'total_features': 0.0, 'density': 0.0}

        arr = np.asarray(vals, dtype=np.float32)
        nz = arr[arr > 0]
        mean_fisher = float(nz.mean()) if nz.size else 0.0
        density = float(nz.size) / float(arr.size) if arr.size else 0.0
        return {'mean_fisher': mean_fisher, 'total_features': float(nz.size), 'density': density}

    def state(self) -> Dict[str, Any]:
        return {
            'position': self.robot_pos.copy(),
            'angle': float(self.robot_angle),
            'gaze_angle': float(self.gaze_angle),
            'linear_velocity': float(self.velocity),
            'angular_velocity': float(self.angular_velocity),
            'step_counter': int(self.step_counter),
            'collision_occurred': bool(self._collision_occurred),
            'stuck_counter': int(self._stuck_counter),
            'sim_time': float(self.sim_time)
        }

    # -------- Internals (no rendering) -------- #

    def _inside_world(self, x, y) -> bool:
        return 0.0 <= x < self.world_width and 0.0 <= y < self.world_height

    def _gen_obstacles(self, num_obstacles=20):
        self.obstacles.clear()
        wall = 0.5
        # walls: rect (x, y, w, h) in meters
        self.obstacles += [
            ('rect', (0.0, 0.0, self.world_width, wall)),
            ('rect', (0.0, self.world_height - wall, self.world_width, wall)),
            ('rect', (0.0, 0.0, wall, self.world_height)),
            ('rect', (self.world_width - wall, 0.0, wall, self.world_height))
        ]
        # room-like blocks
        for _ in range(num_obstacles):
            x = random.uniform(2.5, self.world_width - 2.5)
            y = random.uniform(2.5, self.world_height - 2.5)
            w = random.uniform(2.0, 5.0)
            h = random.uniform(2.0, 5.0)
            if random.choice([True, False]):
                self.obstacles.append(('rect', (x, y, w, h)))
            else:
                self.obstacles.append(('rect', (x, y, h, w)))

    def _find_safe_start(self) -> np.ndarray:
        margin = self.robot_size + 0.5
        for _ in range(100):
            p = np.array([random.uniform(margin, self.world_width - margin),
                          random.uniform(margin, self.world_height - margin)], dtype=np.float64)
            if self._position_safe(p):
                return p
        # fallback search near center
        c = np.array([self.world_width / 2, self.world_height / 2], dtype=np.float64)
        if self._position_safe(c):
            return c
        for r in np.arange(0.5, 5.0, 0.5):
            for a in range(0, 360, 30):
                px = c[0] + r * math.cos(math.radians(a))
                py = c[1] + r * math.sin(math.radians(a))
                p = np.array([px, py], dtype=np.float64)
                if self._position_safe(p):
                    return p
        return np.array([clamp(c[0], margin, self.world_width - margin),
                         clamp(c[1], margin, self.world_height - margin)], dtype=np.float64)

    def _position_safe(self, pos: np.ndarray) -> bool:
        if (pos[0] < self.robot_size or pos[0] > self.world_width - self.robot_size or
            pos[1] < self.robot_size or pos[1] > self.world_height - self.robot_size):
            return False
        return not self._collide_at(pos)

    def _point_in_obstacle(self, x: float, y: float) -> bool:
        for kind, data in self.obstacles:
            if kind == 'rect':
                ox, oy, w, h = data
                if ox <= x <= ox + w and oy <= y <= oy + h:
                    return True
        return False

    def _collide_at(self, target_pos: np.ndarray) -> bool:
        rx, ry = target_pos[0], target_pos[1]
        rs = self.robot_size
        for kind, data in self.obstacles:
            if kind == 'rect':
                x, y, w, h = data
                cx = clamp(rx, x, x + w)
                cy = clamp(ry, y, y + h)
                if math.hypot(rx - cx, ry - cy) < rs:
                    return True
        return False

    def _update_robot(self):
        # angular
        new_angle = (self.robot_angle + math.degrees(self.angular_velocity * self.dt)) % 360.0
        # gaze (independent)
        if self._active_gaze_control:
            self.gaze_angle = self._target_gaze_angle % 360.0
            self._active_gaze_control = False

        # linear
        new_pos = self.robot_pos.copy()
        new_pos[0] += math.cos(math.radians(self.robot_angle)) * self.velocity * self.dt
        new_pos[1] += math.sin(math.radians(self.robot_angle)) * self.velocity * self.dt
        new_pos[0] = clamp(new_pos[0], self.robot_size, self.world_width - self.robot_size)
        new_pos[1] = clamp(new_pos[1], self.robot_size, self.world_height - self.robot_size)

        if self._collide_at(new_pos):
            self._handle_collision()
        else:
            self.robot_pos = new_pos
            self.robot_angle = new_angle

    def _handle_collision(self):
        # reverse with randomness
        self.velocity = clamp(-self.velocity * random.uniform(0.5, 1.0) + random.uniform(-0.5, 0.5),
                              -self.max_linear_velocity, self.max_linear_velocity)
        if abs(self.angular_velocity) < 0.1:
            self.angular_velocity = random.choice([-1, 1]) * random.uniform(0.3, 0.8)
        else:
            self.angular_velocity = clamp(-self.angular_velocity + random.uniform(-0.2, 0.2),
                                          -self.max_angular_velocity, self.max_angular_velocity)

    def _maybe_sample_velocity(self):
        if not self.velocity_sampling_enabled:
            return
        if (self.sim_time - self.last_velocity_sample_time) >= self.velocity_sample_interval:
            self.velocity = random.uniform(-self.max_linear_velocity, self.max_linear_velocity)
            self.angular_velocity = random.uniform(-self.max_angular_velocity, self.max_angular_velocity)
            self.last_velocity_sample_time = self.sim_time

    # ----- Fisher Map ----- #

    def _apply_feature_decay(self):
        # very slight decay with floor
        self.global_feature_map *= (1.0 - 5e-6)
        self.global_feature_map[self.global_feature_map < 0.1] = 0.0

    def _detect_and_add_features_to_global_map(self):
        max_dist = self.fov_distance
        step = 0.1
        half = int(self.fov_angle // 2)
        for off in range(-half, half + 1, 3):
            ang = math.radians((self.gaze_angle + off) % 360.0)
            steps = int(max_dist / step)
            for k in range(1, steps):
                d = k * step
                wx = self.robot_pos[0] + math.cos(ang) * d
                wy = self.robot_pos[1] + math.sin(ang) * d
                out = not self._inside_world(wx, wy)
                hit = self._point_in_obstacle(wx, wy)
                if out or hit:
                    fisher = self._fisher_at(wx, wy, d, ang)
                    self._add_global_feature(wx, wy, fisher)
                    break

    def _fisher_at(self, wx: float, wy: float, distance: float, ang_rad: float) -> float:
        # distance factor
        dist_factor = 1.0 / max(distance / 50.0, 0.1)
        dist_factor = min(dist_factor, 10.0)
        # angle factor wrt principal axes
        angle_deg = (math.degrees(ang_rad) % 360.0)
        min_dev = min(angdiff_deg(angle_deg, d) for d in (0.0, 90.0, 180.0, 270.0))
        ang_factor = max(math.cos(math.radians(min_dev)) ** 2, 0.1)
        # FOV center factor
        dev = angdiff_deg(angle_deg, self.gaze_angle % 360.0)
        fov_factor = max(math.exp(-dev / (self.fov_angle / 4.0)), 0.2)
        return float(clamp(dist_factor * ang_factor * fov_factor, 0.1, 10.0))

    def _add_global_feature(self, wx: float, wy: float, val: float):
        m = self.global_feature_map
        size = self.global_feature_map_size
        res = self.feature_map_resolution

        # world->global map index (centered to world)
        gx = int(wx / res + size // 2 - self.world_width // (2 * res))
        gy = int(wy / res + size // 2 - self.world_height // (2 * res))
        if 0 <= gx < size and 0 <= gy < size:
            if m[gy, gx] < val:
                m[gy, gx] = val
            nb = val * 0.4
            # 8-neighborhood
            x0, x1 = max(0, gx - 1), min(size, gx + 2)
            y0, y1 = max(0, gy - 1), min(size, gy + 2)
            patch = m[y0:y1, x0:x1]
            m[y0:y1, x0:x1] = np.maximum(patch, nb)

    def _extract_local_feature_map(self):
        """Robot-centered crop without Python loops."""
        m = self.global_feature_map
        size = self.global_feature_map_size
        res = self.feature_map_resolution
        half = self.feature_map_size // 2

        rx = int(self.robot_pos[0] / res + size // 2 - self.world_width // (2 * res))
        ry = int(self.robot_pos[1] / res + size // 2 - self.world_height // (2 * res))

        gx0, gy0 = rx - half, ry - half
        gx1, gy1 = gx0 + self.feature_map_size, gy0 + self.feature_map_size

        sx0 = max(0, -gx0)
        sy0 = max(0, -gy0)
        sx1 = self.feature_map_size - max(0, gx1 - size)
        sy1 = self.feature_map_size - max(0, gy1 - size)

        Gx0 = max(0, gx0)
        Gy0 = max(0, gy0)
        Gx1 = min(size, gx1)
        Gy1 = min(size, gy1)

        self.feature_map.fill(0.0)
        if sx0 < sx1 and sy0 < sy1:
            self.feature_map[sy0:sy1, sx0:sx1] = m[Gy0:Gy1, Gx0:Gx1]

        # slight blur
        self.feature_map = cv2.GaussianBlur(self.feature_map, (3, 3), 0.5)


# --------------------------- Renderer (View) --------------------------- #

class RobotRenderer:
    """
    Pure view: draw world, obstacles, FOV overlay, robot, and feature map window.
    No simulation logic, no statistics.
    """
    def __init__(self, core: RobotCore, render_mode: Optional[str] = "human"):
        self.core = core
        self.render_mode = render_mode
        # Persistent canvases
        self.world_img = np.ones((core.height, core.width, 3), dtype=np.uint8) * 255

    def render(self):
        if self.render_mode is None:
            return None

        self.world_img.fill(255)
        self._draw_obstacles()
        self._draw_fov_overlay()
        self._draw_robot()

        if self.render_mode == "human":
            self._show_windows()
        elif self.render_mode == "rgb_array":
            return self.world_img.copy()

    # ------ drawing helpers ------ #

    def _w2p(self, wx: float, wy: float) -> Tuple[int, int]:
        return int(wx * self.core.pixel_per_meter), int(wy * self.core.pixel_per_meter)

    def _draw_obstacles(self):
        ppm = self.core.pixel_per_meter
        for kind, data in self.core.obstacles:
            if kind == 'rect':
                x, y, w, h = data
                px, py = int(x * ppm), int(y * ppm)
                pw, ph = int(w * ppm), int(h * ppm)
                cv2.rectangle(self.world_img, (px, py), (px + pw, py + ph), (0, 0, 0), -1)

    def _draw_robot(self):
        ppm = self.core.pixel_per_meter
        cx, cy = self._w2p(self.core.robot_pos[0], self.core.robot_pos[1])
        r = int(self.core.robot_size * ppm)
        cv2.circle(self.world_img, (cx, cy), r, (0, 255, 0), -1)

        # body heading (red)
        L1 = int(18 * ppm / 20)
        ex = int(cx + math.cos(math.radians(self.core.robot_angle)) * L1)
        ey = int(cy + math.sin(math.radians(self.core.robot_angle)) * L1)
        cv2.arrowedLine(self.world_img, (cx, cy), (ex, ey), (0, 0, 255), 3)

        # gaze (blue)
        L2 = int(25 * ppm / 20)
        gx = int(cx + math.cos(math.radians(self.core.gaze_angle)) * L2)
        gy = int(cy + math.sin(math.radians(self.core.gaze_angle)) * L2)
        cv2.arrowedLine(self.world_img, (cx, cy), (gx, gy), (255, 0, 0), 2)

        cv2.putText(self.world_img, "R", (cx - 5, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_fov_overlay(self):
        ppm = self.core.pixel_per_meter
        cx, cy = self._w2p(self.core.robot_pos[0], self.core.robot_pos[1])

        fov_pts_pix = [(cx, cy)]
        half = int(self.core.fov_angle // 2)
        fov_pix_range = int(self.core.fov_distance * ppm)

        for off in range(-half, half + 1, 3):
            ang = math.radians((self.core.gaze_angle + off) % 360.0)
            max_pix = fov_pix_range
            # march in pixels (purely for drawing)
            for d in range(1, fov_pix_range, 3):
                wx = self.core.robot_pos[0] + math.cos(ang) * (d / ppm)
                wy = self.core.robot_pos[1] + math.sin(ang) * (d / ppm)
                if not self.core._inside_world(wx, wy) or self.core._point_in_obstacle(wx, wy):
                    max_pix = d
                    break
            ex = int(cx + math.cos(ang) * max_pix)
            ey = int(cy + math.sin(ang) * max_pix)
            ex = clamp(ex, 0, self.core.width - 1)
            ey = clamp(ey, 0, self.core.height - 1)
            fov_pts_pix.append((ex, ey))

        if len(fov_pts_pix) > 2:
            overlay = np.zeros_like(self.world_img)
            pts = np.array(fov_pts_pix, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (100, 150, 255))
            cv2.polylines(overlay, [pts], True, (50, 100, 200), 2)
            cv2.addWeighted(self.world_img, 0.6, overlay, 0.4, 0, self.world_img)

    def _show_windows(self):
        # world view
        cv2.imshow("Robot Simulation", self.world_img)

        # feature map view (robot-centered)
        fmap = self.core.feature_map
        vmax = float(np.max(fmap))
        if vmax > 0:
            norm = (fmap / vmax * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(fmap, dtype=np.uint8)
        heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        c = self.core.feature_map_size // 2
        cv2.circle(heat, (c, c), 3, (255, 255, 255), -1)
        cv2.circle(heat, (c, c), 4, (0, 0, 0), 1)

        L = 10
        ex = int(c + math.cos(math.radians(self.core.robot_angle)) * L)
        ey = int(c + math.sin(math.radians(self.core.robot_angle)) * L)
        cv2.arrowedLine(heat, (c, c), (ex, ey), (255, 255, 255), 2)

        gx = int(c + math.cos(math.radians(self.core.gaze_angle)) * L)
        gy = int(c + math.sin(math.radians(self.core.gaze_angle)) * L)
        cv2.arrowedLine(heat, (c, c), (gx, gy), (0, 255, 255), 2)

        view = cv2.resize(heat, (600, 600), interpolation=cv2.INTER_NEAREST)
        stats = self.core.fisher_map_stats()
        cv2.putText(view, "Fisher Information Map", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(view, "White: Robot, Yellow: Gaze", (10, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(view, f"Features: {int(stats['total_features'])}", (10, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(view, f"Avg Fisher: {stats['mean_fisher']:.2f}", (150, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imshow("Feature Map", view)


# -------------------------------- Main -------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot Simulator with Active Gaze Control')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no visualization)')
    parser.add_argument('--realtime', action='store_true', help='Run at real-time speed (otherwise run as fast as possible)')
    parser.add_argument('--steps', type=int, default=1000, help='Number of simulation steps to run (default: 1000)')
    parser.add_argument('--world-size', type=float, default=40.0, help='World size in meters (default: 40.0)')
    args = parser.parse_args()

    print("Starting Robot Simulator with Active Gaze Control...")
    print(f"  - Headless mode: {args.headless}")
    print(f"  - Real-time: {args.realtime}")
    print(f"  - Simulation steps: {args.steps}")
    print(f"  - World size: {args.world_size}m x {args.world_size}m")

    core = RobotCore(world_width=args.world_size, world_height=args.world_size)
    core.velocity_sampling_enabled = True
    core.reset(regenerate_map=True)

    if not args.headless:
        renderer = RobotRenderer(core, render_mode="human")
    else:
        renderer = None

    init = core.state()
    print(f"Robot initialized at position: [{init['position'][0]:.2f}, {init['position'][1]:.2f}] m")

    start_real = time.time()
    expected_sim_t = 0.0
    last_gaze_update = 0.0
    try:
        for step in range(args.steps):
            
            if core.sim_time - last_gaze_update >= 1.5:  # 每 1.5 秒随机一次
                desired = random.uniform(0.0, 360.0)
                core.set_gaze(desired)
                last_gaze_update = core.sim_time
                
            core.step()
            core.update_maps()

            if step % 50 == 0:
                st = core.state()
                f_all = core.fisher_map_stats()
                f_fov = core.fov_fisher_stats()
                print(f"Step {step:4d} (t={st['sim_time']:6.1f}s): "
                      f"Pos=[{st['position'][0]:6.2f},{st['position'][1]:6.2f}]m, "
                      f"Vel={st['linear_velocity']:5.2f}m/s, "
                      f"ω={st['angular_velocity']:5.2f}rad/s, "
                      f"Gaze={st['gaze_angle']:3.0f}°, "
                      f"Fisher={f_all['total_features']:4.0f}, "
                      f"FOV={f_fov['total_features']:3.0f}")

            if renderer:
                renderer.render()
                # allow UI to breathe
                cv2.waitKey(1)

            if args.realtime:
                expected_sim_t += core.dt
                now = time.time() - start_real
                sleep_t = expected_sim_t - now
                if sleep_t > 0:
                    time.sleep(sleep_t)
            else:
                # small delay for smoother UI
                if renderer:
                    time.sleep(0.01)

            if renderer:
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q') or k == 27:
                    print("User requested quit")
                    break

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        st = core.state()
        all_stats = core.fisher_map_stats()
        fov_stats = core.fov_fisher_stats()
        print("\nFinal Results:")
        print(f"  Simulation time: {st['sim_time']:.1f} seconds")
        print(f"  Real execution time: {time.time() - start_real:.1f} seconds")
        print(f"  Final position: [{st['position'][0]:.2f}, {st['position'][1]:.2f}] m")
        print(f"  Final angle: {st['angle']:.0f}°")
        print(f"  Final gaze: {st['gaze_angle']:.0f}°")
        print(f"  Total steps: {st['step_counter']}")
        print(f"  Fisher features discovered: {all_stats['total_features']:.0f}")
        print(f"  Average Fisher value: {all_stats['mean_fisher']:.3f}")
        print(f"  Map density: {all_stats['density']:.3f}")
        print(f"  Stuck counter: {st['stuck_counter']}")
        if not args.headless:
            cv2.destroyAllWindows()
        print("Simulation completed!")
