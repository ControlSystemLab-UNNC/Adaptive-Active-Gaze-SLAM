#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, os, time, random
from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# —— 你的环境与分析器 —— #
from aag_slam_simulator import RobotCore, RobotRenderer   # ← 不改环境文件
from aag_slam_fisher_analyzer import FisherMapAnalyzer, FisherDirectionInfo  # ← 不改分析器文件

# ----------------- 小工具 ----------------- #
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def angdiff(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)

# ----------------- 网络 ----------------- #
class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# ----------------- PPO Agent ----------------- #
class PPO:
    """
    离散动作：把 360° 等分为 action_dim 份（默认 72 => 5°/格）
    观测： [cos θ_p, sin θ_p, strength_norm]  —— 由 s 时刻的主方向得到
    奖励： 用 s′ 的主方向与“刚执行的 gaze”对齐度 × 强度归一化
    """
    def __init__(self, obs_dim: int, action_dim: int = 72, hidden: int = 64,
                 lr=3e-4, gamma=0.95, lam=0.9, clip=0.2, vcoef=0.5, ecoef=0.01,
                 ppo_epochs=4, max_grad_norm=0.5):
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.angle_step = 360.0 / action_dim
        self.gamma, self.lam = gamma, lam
        self.clip, self.vcoef, self.ecoef = clip, vcoef, ecoef
        self.ppo_epochs, self.max_grad_norm = ppo_epochs, max_grad_norm

        self.actor, self.critic = Actor(obs_dim, action_dim, hidden), Critic(obs_dim, hidden)
        self.actor_old, self.critic_old = Actor(obs_dim, action_dim, hidden), Critic(obs_dim, hidden)
        self._sync_old()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for m in (self.actor, self.critic, self.actor_old, self.critic_old):
            m.to(self.device)
        self.opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

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

    # ---- MDP编解码 ---- #
    @staticmethod
    def obs_from_primary(primary: Optional[FisherDirectionInfo], fmap: np.ndarray) -> np.ndarray:
        if primary is None:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)  # cos0, sin0, 0
        ang = math.radians(primary.angle)
        vmax = float(np.max(fmap)) + 1e-6
        s = float(np.clip(primary.strength / vmax, 0.0, 1.0))
        return np.array([math.cos(ang), math.sin(ang), s], dtype=np.float32)

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

    def _gae(self, R, V, D):
        T = len(R); adv = np.zeros(T, np.float32); gae = 0.0
        for t in reversed(range(T)):
            nv = 0.0 if (t == T - 1 or D[t]) else V[t + 1]
            delta = R[t] + self.gamma * nv * (1.0 - float(D[t])) - V[t]
            gae = delta + self.gamma * self.lam * (1.0 - float(D[t])) * gae
            adv[t] = gae
        return adv

    def update(self):
        if not self.S: return
        # 末样本视作 done，避免 bootstrap 偏差
        self.D[-1] = True

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
        for _ in range(self.ppo_epochs):
            dist = Categorical(logits=self.actor(S))
            new_logp = dist.log_prob(A)
            entropy = dist.entropy().mean()
            Vpred = self.critic(S)

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * ADV
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ADV
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss = nn.MSELoss()(Vpred, RET)
            kl = (old_logp - new_logp).mean()

            loss = pi_loss + self.vcoef * v_loss - self.ecoef * entropy
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),  self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.opt.step()

            tot_pi += pi_loss.item(); tot_v += v_loss.item()
            tot_H += entropy.item();  tot_KL += kl.item()

        print(f"\nPPO Update: N={len(self.S)} pi={tot_pi/self.ppo_epochs:.4f} "
              f"V={tot_v/self.ppo_epochs:.4f} H={tot_H/self.ppo_epochs:.4f} KL={tot_KL/self.ppo_epochs:.4f}")
        self._sync_old(); self.reset_buf()

# ----------------- 奖励（基于 s′） ----------------- #
def reward_from_next(gaze_angle: float,
                     primary_next: Optional[FisherDirectionInfo],
                     fmap_next: np.ndarray) -> Tuple[float, Dict]:
    if primary_next is None:
        return 0.0, {"align": 0.0, "s_norm": 0.0, "diff": 180.0}
    diff = angdiff(gaze_angle, primary_next.angle)          # 0..180
    align = 1.0 - diff / 180.0                              # 0..1
    vmax = float(np.max(fmap_next)) + 1e-6
    s_norm = float(np.clip(primary_next.strength / vmax, 0.0, 1.0))
    return align * s_norm, {"align": align, "s_norm": s_norm, "diff": diff}

# ----------------- 训练 / 测试 ----------------- #
def train_loop(core: RobotCore, renderer: Optional[RobotRenderer],
               analyzer: FisherMapAnalyzer, agent: PPO,
               episodes: int, max_steps: int, update_freq: int,
               vel_interval: int, headless: bool, realtime: bool, seed: int):

    set_seed(seed)
    print("== TRAIN ==")
    best_mean = -1e9; returns: List[float] = []

    for ep in range(1, episodes + 1):
        core.reset(regenerate_map=True)         # ← 你的接口
        core.update_maps()                      # ← reset 后先构图，保证有 fmap 可用  
        ep_ret = 0.0; t0 = time.time()

        for t in range(1, max_steps + 1):
            if t % vel_interval == 1:
                core.set_velocity(np.random.uniform(-1.5, 2.0), np.random.uniform(-1.0, 1.0))

            # s：从当前 fmap 提取主方向
            primary, _ = analyzer.analyze(core.feature_map)          # ← 传 2D numpy map  
            obs = PPO.obs_from_primary(primary, core.feature_map)

            # a
            gaze, logp, a_idx = agent.act(obs)
            v = agent.value(obs)

            # 执行动作 => s′
            core.set_gaze(gaze)                                      # ← 外部控制 gaze  
            core.step()
            core.update_maps()                                       # ← 先更新地图再计算奖励

            primary_next, _ = analyzer.analyze(core.feature_map)
            r, _info = reward_from_next(gaze, primary_next, core.feature_map)
            ep_ret += r

            agent.store(obs, a_idx, logp, r, v, False)
            if len(agent.S) >= update_freq:
                agent.update()

            if renderer:
                renderer.render()
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q') or k == 27:
                    print("\n⏹️ 用户请求退出")
                    return
            if (not headless) and (not realtime): time.sleep(0.01)

        if len(agent.S) > 0: agent.update()
        returns.append(ep_ret)
        m = float(np.mean(returns[-10:])); s = float(np.std(returns[-10:])) if len(returns) >= 10 else 0.0
        if len(returns) >= 10 and m > best_mean:
            best_mean = m
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"actor": agent.actor.state_dict(),
                        "critic": agent.critic.state_dict()}, "checkpoints/best.pth")
            print(f"💾 save best mean10={m:.3f}")
        print(f"EP {ep:04d} return={ep_ret:.3f} mean10={m:.3f}±{s:.3f}  {(time.time()-t0):.1f}s")

def test_loop(core: RobotCore, renderer: Optional[RobotRenderer],
              analyzer: FisherMapAnalyzer, agent: PPO,
              episodes: int, max_steps: int, headless: bool, realtime: bool, seed: int):
    set_seed(seed)
    print("== TEST ==")
    for ep in range(1, episodes + 1):
        core.reset(regenerate_map=True)
        core.update_maps()
        ep_ret = 0.0
        for t in range(1, max_steps + 1):
            if t % 50 == 1:
                core.set_velocity(np.random.uniform(-1.0, 2.0), np.random.uniform(-1.0, 1.0))

            primary, _ = analyzer.analyze(core.feature_map)
            obs = PPO.obs_from_primary(primary, core.feature_map)
            with torch.no_grad():
                x = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
                dist = Categorical(logits=agent.actor(x))
                a_idx = torch.argmax(dist.probs, dim=1).item()
                gaze = (a_idx * agent.angle_step) % 360.0
                print(f"gaze={gaze:.3f}")

            core.set_gaze(gaze); core.step(); core.update_maps()
            primary_next, _ = analyzer.analyze(core.feature_map)
            r, _ = reward_from_next(gaze, primary_next, core.feature_map)
            ep_ret += r

            if renderer: 
                renderer.render()
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q') or k == 27:
                    print("\n⏹️ 用户请求退出")
                    return
            if (not headless) and (not realtime): time.sleep(0.01)
        print(f"[TEST] EP {ep:03d} return={ep_ret:.3f}")

# ----------------- CLI ----------------- #
def build_parser():
    p = argparse.ArgumentParser(description="PPO Gaze (discrete) with RobotCore + FisherMapAnalyzer")
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--train", action="store_true", help="train")
    m.add_argument("--test",  action="store_true", help="test")

    # 与环境一致的参数
    p.add_argument("--headless",  action="store_true", help="no GUI")
    p.add_argument("--realtime",  action="store_true", help="real-time stepping")
    p.add_argument("--world-size", type=float, default=40.0, help="world size (m)")
    p.add_argument("--fov-angle",  type=float, default=90.0, help="FOV angle (deg)")

    # 训练
    p.add_argument("--episodes",    type=int,   default=300,  help="episodes")
    p.add_argument("--max-steps",   type=int,   default=500,  help="steps/ep")
    p.add_argument("--update-freq", type=int,   default=2048, help="PPO batch size")
    p.add_argument("--vel-interval",type=int,   default=50,   help="velocity change interval")
    p.add_argument("--seed",        type=int,   default=42,   help="seed")

    # agent
    p.add_argument("--action-dim",  type=int,   default=72,   help="#bins (72=5°)")
    p.add_argument("--hidden",      type=int,   default=64,   help="MLP hidden")
    p.add_argument("--lr",          type=float, default=3e-4, help="learning rate")
    p.add_argument("--gamma",       type=float, default=0.95, help="discount")
    p.add_argument("--lam",         type=float, default=0.90, help="GAE lambda")
    p.add_argument("--clip",        type=float, default=0.20, help="PPO clip")
    p.add_argument("--vcoef",       type=float, default=0.5,  help="value coef")
    p.add_argument("--ecoef",       type=float, default=0.01, help="entropy coef")
    p.add_argument("--ppo-epochs",  type=int,   default=4,    help="PPO epochs")

    # 模型
    p.add_argument("--model",       type=str,   default="ppo_gaze.pth", help="save/load path")
    return p

def main():
    args = build_parser().parse_args()
    print(f"Mode={'TRAIN' if args.train else 'TEST'}  headless={args.headless} realtime={args.realtime}")

    core = RobotCore(world_width=args.world_size, world_height=args.world_size, fov_angle=args.fov_angle)
    renderer = None if args.headless else RobotRenderer(core, render_mode="human")
    analyzer = FisherMapAnalyzer(threshold_ratio=0.2, min_points=15, fov_angle=args.fov_angle)

    obs_dim = 3
    agent = PPO(obs_dim=obs_dim, action_dim=args.action_dim, hidden=args.hidden,
                lr=args.lr, gamma=args.gamma, lam=args.lam, clip=args.clip,
                vcoef=args.vcoef, ecoef=args.ecoef, ppo_epochs=args.ppo_epochs)

    if args.train:
        train_loop(core, renderer, analyzer, agent,
                   episodes=args.episodes, max_steps=args.max_steps,
                   update_freq=args.update_freq, vel_interval=args.vel_interval,
                   headless=args.headless, realtime=args.realtime, seed=args.seed)
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
            print(f"Load failed: {e}"); return
        test_loop(core, renderer, analyzer, agent,
                  episodes=5, max_steps=args.max_steps,
                  headless=args.headless, realtime=args.realtime, seed=args.seed)

if __name__ == "__main__":
    main()
