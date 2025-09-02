#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PPO for Active Gaze (Full Observation, Minimal Args, Linus-style)
- 全观测：主/次方向 + 局部 Fisher 统计 + FOV 统计（共 14 维）
- Arg 仅保留：--train/--test, --headless, --realtime, --model, --seed
- 其余参数在下方常量硬编码（直接改数值即可）
"""

import os
import cv2
import time
import math
import random
import argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ==== 你可以根据需要直接修改这些常量（无需命令行）==========================
# 环境常量
WORLD_SIZE   = 40.0     # m
FOV_ANGLE    = 90.0     # deg

# PPO/网络常量
ACTION_DIM   = 72       # 5° 一格
HIDDEN       = 256
LR           = 3e-4
GAMMA        = 0.95
LAMBDA       = 0.90
CLIP         = 0.20
VCOEF        = 0.50
ECOEF        = 0.01
PPO_EPOCHS   = 4
MAX_GRAD_NORM= 0.5

# 训练循环常量
EPISODES     = 300
MAX_STEPS    = 500
UPDATE_FREQ  = 2048
VEL_INTERVAL = 50

# 奖励权重/逻辑
PRIMARY_JUMP_DEG = 45.0   # 主方向“剧烈跳变”的阈值（度）
SECONDARY_BONUS  = 0.15   # 主方向跳变时，次方向择优的加成
W_ALIGN          = 1.00   # 对齐奖励权重
W_EXPLORE        = 0.30   # 探索奖励权重
W_SMOOTH         = 0.20   # 平滑惩罚权重（越大越平稳）
EXPL_CLIP        = 0.05   # 单步探索奖励的上限（避免过大）
# ===========================================================================

# 环境 / 可视化 / 分析器
from aag_slam_simulator import RobotCore, RobotRenderer
from aag_slam_fisher_analyzer import FisherMapAnalyzer, FisherDirectionInfo


# ------------------------- 小工具 ------------------------- #
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def angdiff(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)

def safe_norm(x: float, eps: float = 1e-6) -> float:
    return float(x) if x > eps else 0.0


# ------------------------- MLP 模型 ------------------------- #
class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)


# ------------------------- PPO Agent（离散角度, 全观测14维） ------------------------- #
class PPO:
    """
    动作：将 360° 等分为 ACTION_DIM 份。
    观测（固定 14 维）：
      [cosP, sinP, sP, confP, cosS, sinS, sS, confS, meanL, denL, totL, meanF, denF, totF]
    """
    OBS_DIM = 14

    def __init__(self):
        self.action_dim = ACTION_DIM
        self.angle_step = 360.0 / float(self.action_dim)

        # nets
        self.actor  = Actor(self.OBS_DIM, self.action_dim, HIDDEN)
        self.critic = Critic(self.OBS_DIM, HIDDEN)
        self.actor_old  = Actor(self.OBS_DIM, self.action_dim, HIDDEN)
        self.critic_old = Critic(self.OBS_DIM, HIDDEN)
        self._sync_old()

        # device/optim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for m in (self.actor, self.critic, self.actor_old, self.critic_old):
            m.to(self.device)
        self.opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=LR)

        # buffers
        self.reset_buf()

    def _sync_old(self):
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def reset_buf(self):
        self.S: List[np.ndarray] = []
        self.A: List[int] = []
        self.logpA: List[float] = []
        self.R: List[float] = []
        self.V: List[float] = []
        self.D: List[bool] = []

    # -------- 观测拼装（固定 14 维） -------- #
    @staticmethod
    def _enc_dir(info: Optional[FisherDirectionInfo], fmap: np.ndarray) -> List[float]:
        if info is None:
            return [0.0, 0.0, 0.0, 0.0]
        ang = math.radians(info.angle % 360.0)
        vmax = float(np.max(fmap)) + 1e-6
        s_norm = float(np.clip(info.strength / vmax, 0.0, 1.0))
        conf   = float(np.clip(info.confidence, 0.0, 1.0))
        return [math.cos(ang), math.sin(ang), s_norm, conf]

    def build_obs(self, core: RobotCore, analyzer: FisherMapAnalyzer) -> Tuple[np.ndarray, Optional[FisherDirectionInfo], Optional[FisherDirectionInfo]]:
        fmap = core.feature_map
        primary, secondary = analyzer.analyze(fmap)

        enc_p = self._enc_dir(primary, fmap)
        enc_s = self._enc_dir(secondary, fmap)

        st_local = core.fisher_map_stats()      # mean_fisher / total_features / density
        vmax = float(np.max(fmap)) + 1e-6
        ncell = float(fmap.size)
        local_mean_norm  = float(np.clip(st_local.get('mean_fisher', 0.0) / vmax, 0.0, 1.0))
        local_density    = float(np.clip(st_local.get('density', 0.0),        0.0, 1.0))
        local_total_norm = float(np.clip(st_local.get('total_features', 0.0) / ncell, 0.0, 1.0))

        st_fov = core.fov_fisher_stats()
        fov_mean_norm  = float(np.clip(st_fov.get('mean_fisher', 0.0) / vmax, 0.0, 1.0))
        fov_density    = float(np.clip(st_fov.get('density', 0.0),        0.0, 1.0))
        fov_total_norm = float(np.clip(st_fov.get('total_features', 0.0) / ncell, 0.0, 1.0))

        obs = np.array(enc_p + enc_s +
                       [local_mean_norm, local_density, local_total_norm,
                        fov_mean_norm,   fov_density,   fov_total_norm],
                       dtype=np.float32)
        return obs, primary, secondary

    # -------- 动作/价值 -------- #
    def act(self, obs: np.ndarray) -> Tuple[float, float, int]:
        x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = Categorical(logits=self.actor_old(x))
            a = dist.sample()
            logp = dist.log_prob(a).item()
        idx = int(a.item())
        gaze = (idx * self.angle_step) % 360.0
        return gaze, logp, idx

    def value(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.critic(torch.from_numpy(obs).float().unsqueeze(0).to(self.device)).item())

    def store(self, s, a_idx, logp, r, v, done):
        self.S.append(s); self.A.append(a_idx); self.logpA.append(logp)
        self.R.append(r); self.V.append(v); self.D.append(done)

    # -------- GAE/PPO 更新 -------- #
    def _gae(self, R, V, D):
        T = len(R); adv = np.zeros(T, np.float32); gae = 0.0
        for t in reversed(range(T)):
            nv = 0.0 if (t == T - 1 or D[t]) else V[t + 1]
            delta = R[t] + GAMMA * nv * (1.0 - float(D[t])) - V[t]
            gae = delta + GAMMA * LAMBDA * (1.0 - float(D[t])) * gae
            adv[t] = gae
        return adv

    def update(self):
        if not self.S: return
        self.D[-1] = True  # 末样本视作 done

        S = torch.from_numpy(np.asarray(self.S, np.float32)).to(self.device)
        A = torch.from_numpy(np.asarray(self.A, np.int64)).to(self.device)
        old_logp = torch.from_numpy(np.asarray(self.logpA, np.float32)).to(self.device)
        V_np = np.asarray(self.V, np.float32)
        R_np = np.asarray(self.R, np.float32)
        D_np = np.asarray(self.D, np.bool_)

        ADV_np = self._gae(R_np.tolist(), V_np.tolist(), D_np.tolist())
        ADV = torch.from_numpy((ADV_np - ADV_np.mean()) / (ADV_np.std() + 1e-8)).to(self.device)
        RET = torch.from_numpy(ADV_np + V_np).to(self.device)

        tot_pi = tot_v = tot_H = tot_KL = 0.0
        for _ in range(PPO_EPOCHS):
            dist = Categorical(logits=self.actor(S))
            new_logp = dist.log_prob(A)
            entropy = dist.entropy().mean()
            Vpred = self.critic(S)

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * ADV
            surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * ADV
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss = nn.MSELoss()(Vpred, RET)
            kl = (old_logp - new_logp).mean()

            loss = pi_loss + VCOEF * v_loss - ECOEF * entropy
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),  MAX_GRAD_NORM)
            nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
            self.opt.step()

            tot_pi += pi_loss.item(); tot_v += v_loss.item()
            tot_H += entropy.item();  tot_KL += kl.item()

        print(f"\nPPO Update: N={len(self.S)} pi={tot_pi/PPO_EPOCHS:.4f} "
              f"V={tot_v/PPO_EPOCHS:.4f} H={tot_H/PPO_EPOCHS:.4f} KL={tot_KL/PPO_EPOCHS:.4f}")
        self._sync_old(); self.reset_buf()


# ------------------------- 奖励（结合主/次 + 探索 + 平滑） ------------------------- #
def reward_from_next(
    gaze_angle: float,
    primary_prev: Optional[FisherDirectionInfo],
    primary_next: Optional[FisherDirectionInfo],
    secondary_next: Optional[FisherDirectionInfo],
    fmap_prev: np.ndarray,
    fmap_next: np.ndarray,
    prev_gaze: float
) -> Tuple[float, Dict]:
    """
    - 对齐：择优使用 s′ 的主/次方向（按强度/置信度归一化），若主方向较 s 的主方向“剧烈跳变”，次方向得加成
    - 探索：局部图 total/density 较上一步增长则给正奖
    - 平滑：惩罚 gaze 的大幅跳变
    """
    vmax_n = float(np.max(fmap_next)) + 1e-6

    # ---- 对齐项（s′）---- #
    def dir_term(info: Optional[FisherDirectionInfo]) -> float:
        if info is None: return 0.0
        diff = angdiff(gaze_angle, info.angle)
        align = 1.0 - diff / 180.0             # 0..1
        s_norm = float(np.clip(info.strength / vmax_n, 0.0, 1.0))
        conf   = float(np.clip(info.confidence, 0.0, 1.0))
        return align * s_norm * conf

    align_p = dir_term(primary_next)
    align_s = dir_term(secondary_next)

    # 主方向是否“剧烈跳变”（相对 s 的主方向）
    jump_bonus = 0.0
    if primary_prev is not None and primary_next is not None:
        jump = angdiff(primary_prev.angle, primary_next.angle)
        if jump >= PRIMARY_JUMP_DEG:
            jump_bonus = SECONDARY_BONUS

    align_term = max(align_p, align_s * (1.0 + jump_bonus))

    # ---- 探索项（s′ vs s）---- #
    def stats(map_):
        flat = map_.flatten()
        non_zero = flat[flat > 0]
        if len(non_zero) == 0:
            return 0.0, 0.0
        mean = float(non_zero.mean())
        total = float(len(non_zero))
        density = total / float(flat.size)
        return mean, density

    _, den_prev = stats(fmap_prev)
    _, den_next = stats(fmap_next)
    # 也可用 core.fisher_map_stats() 传入，这里直接就地计算：
    total_prev = float((fmap_prev > 0).sum()) / float(fmap_prev.size)
    total_next = float((fmap_next > 0).sum()) / float(fmap_next.size)

    explore_gain = max(0.0, (total_next - total_prev)) + 0.5 * max(0.0, (den_next - den_prev))
    explore_term = float(np.clip(explore_gain, 0.0, EXPL_CLIP))  # 限幅，避免过大

    # ---- 平滑项（动作变化）---- #
    d_gaze = angdiff(gaze_angle, prev_gaze) / 180.0
    smooth_penalty = d_gaze * d_gaze  # 二次惩罚

    # ---- 合成 ---- #
    reward = W_ALIGN * align_term + W_EXPLORE * explore_term - W_SMOOTH * smooth_penalty
    info = {
        "align_p": align_p, "align_s": align_s, "jump_bonus": jump_bonus,
        "explore": explore_term, "smooth_penalty": smooth_penalty,
        "total_prev": total_prev, "total_next": total_next, "den_prev": den_prev, "den_next": den_next,
        "reward": reward
    }
    return float(reward), info


# ------------------------- 训练循环 ------------------------- #
def train_loop(core: RobotCore, renderer: Optional[RobotRenderer],
               analyzer: FisherMapAnalyzer, agent: PPO,
               headless: bool, realtime: bool, seed: int, model_path: str):

    set_seed(seed)
    print("== TRAIN ==")
    returns: List[float] = []
    best_mean10 = -1e9

    for ep in range(1, EPISODES + 1):
        core.reset(regenerate_map=True)
        core.update_maps()
        ep_ret = 0.0
        prev_gaze = 0.0

        # s 的主方向（用于 jump 判断）
        _, primary_prev = None, None
        obs, primary_prev, _ = agent.build_obs(core, analyzer)
        fmap_prev = core.feature_map.copy()

        for t in range(1, MAX_STEPS + 1):
            if t % VEL_INTERVAL == 1:
                core.set_velocity(float(np.random.uniform(-1.5, 2.0)),
                                  float(np.random.uniform(-1.0, 1.0)))

            # 观测（s）
            obs, primary_cur, _ = agent.build_obs(core, analyzer)

            # 动作
            gaze, logp, a_idx = agent.act(obs)
            v = agent.value(obs)

            # 执行动作 => s′
            core.set_gaze(gaze)
            core.step()
            core.update_maps()

            # s′ 的方向 & 奖励
            primary_next, secondary_next = analyzer.analyze(core.feature_map)
            r, rb = reward_from_next(
                gaze_angle=gaze,
                primary_prev=primary_prev,
                primary_next=primary_next,
                secondary_next=secondary_next,
                fmap_prev=fmap_prev,
                fmap_next=core.feature_map,
                prev_gaze=prev_gaze
            )
            ep_ret += r

            # 存储样本
            agent.store(obs, a_idx, logp, r, v, False)

            # update
            if len(agent.S) >= UPDATE_FREQ:
                agent.update()

            # 渲染
            if renderer:
                renderer.render()
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q') or k == 27:
                    print("\n⏹️ 用户请求退出")
                    torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()}, model_path)
                    print(f"Saved → {model_path}")
                    return

            if (not headless) and (not realtime):
                time.sleep(0.01)

            # 滚动 s
            prev_gaze = gaze
            primary_prev = primary_next
            fmap_prev = core.feature_map.copy()

        if len(agent.S) > 0:
            agent.update()

        returns.append(ep_ret)
        mean10 = float(np.mean(returns[-10:])) if len(returns) >= 10 else ep_ret
        std10  = float(np.std(returns[-10:]))  if len(returns) >= 10 else 0.0

        # 简单 best 保存
        if len(returns) >= 10 and mean10 > best_mean10:
            best_mean10 = mean10
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()}, model_path)
            print(f"💾 保存最佳 (mean10={mean10:.3f}) → {model_path}")

        print(f"EP {ep:04d} return={ep_ret:.3f} mean10={mean10:.3f}±{std10:.3f}")


# ------------------------- 测试循环 ------------------------- #
def test_loop(core: RobotCore, renderer: Optional[RobotRenderer],
              analyzer: FisherMapAnalyzer, agent: PPO,
              headless: bool, realtime: bool, seed: int):

    set_seed(seed)
    print("== TEST ==")

    for ep in range(1, EPISODES + 1):
        core.reset(regenerate_map=True)
        core.update_maps()
        ep_ret = 0.0
        prev_gaze = 0.0

        # s 的主方向（用于 jump 判断）
        obs, primary_prev, _ = agent.build_obs(core, analyzer)
        fmap_prev = core.feature_map.copy()

        for t in range(1, MAX_STEPS + 1):
            if t % 50 == 1:
                core.set_velocity(float(np.random.uniform(-1.0, 2.0)),
                                  float(np.random.uniform(-1.0, 1.0)))

            # 观测（s）
            obs, _, _ = agent.build_obs(core, analyzer)

            # 贪心动作
            with torch.no_grad():
                x = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
                dist = Categorical(logits=agent.actor(x))
                a_idx = torch.argmax(dist.probs, dim=1).item()
                gaze = (a_idx * agent.angle_step) % 360.0

            # step
            core.set_gaze(gaze)
            core.step()
            core.update_maps()

            # 奖励（用于日志）
            primary_next, secondary_next = analyzer.analyze(core.feature_map)
            r, _ = reward_from_next(
                gaze_angle=gaze,
                primary_prev=primary_prev,
                primary_next=primary_next,
                secondary_next=secondary_next,
                fmap_prev=fmap_prev,
                fmap_next=core.feature_map,
                prev_gaze=prev_gaze
            )
            ep_ret += r

            # 渲染
            if renderer:
                renderer.render()
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q') or k == 27:
                    print("\n⏹️ 用户请求退出")
                    return

            if (not headless) and (not realtime):
                time.sleep(0.01)

            # 滚动 s
            prev_gaze = gaze
            primary_prev = primary_next
            fmap_prev = core.feature_map.copy()

        print(f"[TEST] EP {ep:03d} return={ep_ret:.3f}")


# ------------------------- 参数 & main（极简 Arg） ------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description="PPO Gaze (discrete, full-observation, minimal args)")
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--train", action="store_true", help="train")
    m.add_argument("--test",  action="store_true", help="test")
    p.add_argument("--headless",  action="store_true", help="no GUI")
    p.add_argument("--realtime",  action="store_true", help="real-time stepping")
    p.add_argument("--model",     type=str, default="checkpoints/ppo_full_obs.pth", help="save/load path")
    p.add_argument("--seed",      type=int, default=42, help="random seed")
    return p

def main():
    args = build_parser().parse_args()
    print(f"Mode={'TRAIN' if args.train else 'TEST'}  headless={args.headless} realtime={args.realtime}")

    # 环境/渲染器/分析器
    core = RobotCore(world_width=WORLD_SIZE, world_height=WORLD_SIZE, fov_angle=FOV_ANGLE)
    renderer = None if args.headless else RobotRenderer(core, render_mode="human")
    analyzer = FisherMapAnalyzer(threshold_ratio=0.2, min_points=15, fov_angle=FOV_ANGLE)

    agent = PPO()

    # 训练 or 测试
    if args.train:
        os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
        train_loop(core, renderer, analyzer, agent,
                   headless=args.headless, realtime=args.realtime,
                   seed=args.seed, model_path=args.model)
        torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()}, args.model)
        print(f"Saved → {args.model}")
    else:
        try:
            ckpt = torch.load(args.model, map_location="cpu")
            agent.actor.load_state_dict(ckpt["actor"])
            agent.critic.load_state_dict(ckpt["critic"])
            agent._sync_old()
            print(f"Loaded ← {args.model}")
        except Exception as e:
            print(f"Load failed: {e}")
            return
        test_loop(core, renderer, analyzer, agent,
                  headless=args.headless, realtime=args.realtime, seed=args.seed)

if __name__ == "__main__":
    main()
