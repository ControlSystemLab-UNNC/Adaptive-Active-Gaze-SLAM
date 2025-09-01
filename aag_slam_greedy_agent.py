#!/usr/bin/env python3
"""
重构的贪心Fisher信息算法 v3.0
基于新的模拟器接口：直接使用RobotCore和RobotRenderer
日期: 2025-09-01
"""

import numpy as np
import math
import cv2
import argparse
import random
import time
from typing import Tuple, Optional, Dict, Any
from aag_slam_simulator import RobotCore, RobotRenderer
from aag_slam_fisher_analyzer import FisherMapAnalyzer, FisherDirectionInfo


class EnhancedGreedyFisherAgent:
    """
    增强版贪心Fisher信息代理 - 基于Fisher方向分析器
    
    利用Fisher方向分析器提取主导方向，实现智能凝视控制：
    1. fisher_map: 100x100的Fisher信息地图
    2. fisher_map_stats: [mean_fisher, total_features, density] 
    3. fov_fisher_stats: FOV内的Fisher统计 [mean_fisher, total_features, density]
    4. robot_velocity: [linear_vel, angular_vel]
    5. gaze_angle: [current_gaze_angle]
    
    新的贪心策略：
    - 凝视角度对齐第一或第二主方向（对齐程度×特征强度×置信度）
    - 防止全局Fisher信息衰减过快
    - 保持凝视角度变化的平滑性
    """
    
    def __init__(self, 
                 # Fisher分析器参数
                 fisher_threshold_ratio: float = 0.2,
                 fisher_min_points: int = 15,
                 fov_angle: float = 90.0,
                 
                 # 贪心策略参数
                 alignment_weight: float = 0.6,      # 方向对齐权重
                 anti_decay_weight: float = 0.3,     # 防衰减权重
                 smoothness_weight: float = 0.1,     # 平滑性权重
                 max_angle_change: float = 45.0,     # 最大角度变化（减小以更平滑）
                 
                 # 其他参数
                 adaptive_mode: bool = True):
        """
        初始化增强版贪心Fisher代理
        
        Args:
            fisher_threshold_ratio: Fisher分析器阈值比例
            fisher_min_points: Fisher分析器最小点数
            fov_angle: FOV角度
            alignment_weight: 方向对齐权重
            anti_decay_weight: 防衰减权重  
            smoothness_weight: 平滑性权重
            max_angle_change: 最大角度变化
            adaptive_mode: 是否启用自适应模式
        """
        # 创建Fisher方向分析器
        self.fisher_analyzer = FisherMapAnalyzer(
            threshold_ratio=fisher_threshold_ratio,
            min_points=fisher_min_points,
            fov_angle=fov_angle
        )
        
        # 贪心策略权重
        self.alignment_weight = alignment_weight
        self.anti_decay_weight = anti_decay_weight
        self.smoothness_weight = smoothness_weight
        self.max_angle_change = max_angle_change
        self.adaptive_mode = adaptive_mode
        
        # 状态追踪
        self.current_gaze_angle = 0.0
        self.previous_gaze_angle = 0.0
        self.current_robot_angle = 0.0
        self.decision_history = []
        
        # Fisher衰减监控
        self.previous_fisher_density = 0.0
        self.fisher_decay_rate = 0.0
        
        # 当前主方向信息
        self.current_primary_direction: Optional[FisherDirectionInfo] = None
        self.current_secondary_direction: Optional[FisherDirectionInfo] = None
        
    def get_action(self, core: RobotCore) -> float:
        """
        基于Fisher方向分析器的新贪心策略
        
        贪心目标：
        1. 凝视角度对齐第一或第二主方向（对齐程度×特征强度×置信度）
        2. 防止全局Fisher信息衰减过快
        3. 保持凝视角度变化的平滑性
        
        Args:
            core: RobotCore实例，包含所有状态信息
            
        Returns:
            最优凝视角度 (float)
        """
        # 从core获取观测信息
        fisher_map = core.feature_map
        fisher_map_stats = core.fisher_map_stats()
        fov_fisher_stats = core.fov_fisher_stats()
        state = core.state()
        robot_velocity = [state['linear_velocity'], state['angular_velocity']]
        current_gaze = state['gaze_angle']
        robot_angle = state['angle']
        
        # 更新状态
        self.previous_gaze_angle = self.current_gaze_angle
        self.current_gaze_angle = current_gaze
        self.current_robot_angle = robot_angle
        
        # 使用Fisher分析器获取主导方向
        primary_direction, secondary_direction = self.fisher_analyzer.analyze(fisher_map)
        self.current_primary_direction = primary_direction
        self.current_secondary_direction = secondary_direction
        
        # 计算Fisher衰减率
        current_density = fisher_map_stats['density']
        if self.previous_fisher_density > 0:
            self.fisher_decay_rate = (self.previous_fisher_density - current_density) / self.previous_fisher_density
        self.previous_fisher_density = current_density
        
        # 如果没有找到有效方向，保持当前角度
        if primary_direction is None:
            self._record_decision(core, self.current_gaze_angle, 'no_direction')
            return self.current_gaze_angle
        
        # 计算最优凝视角度
        optimal_angle = self._calculate_optimal_gaze_angle(
            primary_direction, secondary_direction, fisher_map_stats, current_gaze
        )
        
        # 应用平滑约束
        optimal_angle = self._apply_smoothness_constraint(optimal_angle)
        
        # 记录决策历史
        strategy_mode = 'direction_alignment'
        self._record_decision(core, optimal_angle, strategy_mode)
        
        return optimal_angle
    
    def _calculate_optimal_gaze_angle(self, 
                                    primary_direction: FisherDirectionInfo,
                                    secondary_direction: Optional[FisherDirectionInfo],
                                    fisher_map_stats: Dict[str, float],
                                    current_gaze: float) -> float:
        """
        基于主导方向计算最优凝视角度
        
        贪心目标函数 = alignment_score + anti_decay_score - smoothness_penalty
        """
        # 候选角度：主方向、次方向（如果存在）
        candidate_angles = [primary_direction.angle]
        if secondary_direction is not None:
            candidate_angles.append(secondary_direction.angle)
        
        best_angle = current_gaze
        best_score = -np.inf
        
        for angle in candidate_angles:
            # 1. 计算方向对齐得分
            if angle == primary_direction.angle:
                direction_info = primary_direction
            else:
                direction_info = secondary_direction
                
            # 确保direction_info不为None
            if direction_info is None:
                continue
            
            alignment_score = self._calculate_alignment_score(angle, direction_info, current_gaze)
            
            # 2. 计算防衰减得分
            anti_decay_score = self._calculate_anti_decay_score(angle, fisher_map_stats)
            
            # 3. 计算平滑性惩罚
            smoothness_penalty = self._calculate_smoothness_penalty(angle)
            
            # 4. 综合评分
            total_score = (
                self.alignment_weight * alignment_score +
                self.anti_decay_weight * anti_decay_score -
                self.smoothness_weight * smoothness_penalty
            )
            
            if total_score > best_score:
                best_score = total_score
                best_angle = angle
        
        return float(best_angle)
    
    def _calculate_alignment_score(self, angle: float, 
                                 direction_info: FisherDirectionInfo, 
                                 current_gaze: float) -> float:
        """
        计算方向对齐得分：对齐程度 × 特征强度 × 置信度
        """
        # 计算对齐程度（角度差异越小，对齐程度越高）
        angle_diff = abs(angle - direction_info.angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # 对齐程度：从1.0（完全对齐）到0.0（完全不对齐）
        alignment_degree = 1.0 - (angle_diff / 180.0)
        
        # 特征强度已经由Fisher分析器计算
        feature_strength = direction_info.strength
        
        # 置信度
        confidence = direction_info.confidence
        
        # 综合得分
        alignment_score = alignment_degree * feature_strength * confidence
        
        return alignment_score
    
    def _calculate_anti_decay_score(self, angle: float, fisher_map_stats: Dict[str, float]) -> float:
        """
        计算防衰减得分：鼓励选择能维持或增加Fisher信息的角度
        """
        current_density = fisher_map_stats['density']  # 当前Fisher密度
        current_features = fisher_map_stats['total_features']  # 当前特征数
        
        # 如果Fisher信息正在衰减，给予额外奖励
        anti_decay_bonus = 0.0
        if self.fisher_decay_rate > 0.01:  # 衰减超过1%
            # 衰减越严重，防衰减奖励越高
            anti_decay_bonus = self.fisher_decay_rate * 10.0
        
        # 基础得分：当前Fisher信息的质量
        base_score = current_density * current_features / 1000.0  # 归一化
        
        return base_score + anti_decay_bonus
    
    def _calculate_smoothness_penalty(self, angle: float) -> float:
        """
        计算平滑性惩罚（角度变化过大的惩罚）
        """
        angle_change = abs(angle - self.current_gaze_angle)
        if angle_change > 180:
            angle_change = 360 - angle_change
        
        # 归一化到0-1
        normalized_change = angle_change / 180.0
        
        # 二次惩罚
        return normalized_change ** 2
    
    def _apply_smoothness_constraint(self, target_angle: float) -> float:
        """
        应用平滑性约束，限制角度变化
        """
        angle_diff = target_angle - self.current_gaze_angle
        
        # 处理360度边界
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        
        # 限制最大变化
        limited_diff = np.clip(angle_diff, -self.max_angle_change, self.max_angle_change)
        
        # 计算最终角度
        final_angle = (self.current_gaze_angle + limited_diff) % 360
        
        return final_angle
    
    def _record_decision(self, core: RobotCore, 
                        chosen_angle: float, strategy_mode: str):
        """
        记录决策历史，用于分析和调试
        """
        fisher_map_stats = core.fisher_map_stats()
        fov_fisher_stats = core.fov_fisher_stats()
        state = core.state()
        
        decision_record = {
            'timestamp': len(self.decision_history),
            'fisher_map_stats': [fisher_map_stats['mean_fisher'], fisher_map_stats['total_features'], fisher_map_stats['density']],
            'fov_fisher_stats': [fov_fisher_stats['mean_fisher'], fov_fisher_stats['total_features'], fov_fisher_stats['density']],
            'robot_velocity': [state['linear_velocity'], state['angular_velocity']],
            'current_gaze': state['gaze_angle'],
            'chosen_angle': chosen_angle,
            'strategy_mode': strategy_mode,
            'angle_change': abs(chosen_angle - self.current_gaze_angle),
            'primary_direction': self.current_primary_direction.angle if self.current_primary_direction else None,
            'primary_strength': self.current_primary_direction.strength if self.current_primary_direction else 0.0,
            'secondary_direction': self.current_secondary_direction.angle if self.current_secondary_direction else None,
            'secondary_strength': self.current_secondary_direction.strength if self.current_secondary_direction else 0.0,
            'fisher_decay_rate': self.fisher_decay_rate
        }
        
        self.decision_history.append(decision_record)
        
        # 保持历史长度
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    def get_statistics(self) -> Dict:
        """
        获取代理统计信息
        """
        if not self.decision_history:
            return {}
        
        recent_decisions = self.decision_history[-100:]
        
        strategy_counts = {}
        for decision in recent_decisions:
            mode = decision['strategy_mode']
            strategy_counts[mode] = strategy_counts.get(mode, 0) + 1
        
        angle_changes = [d['angle_change'] for d in recent_decisions]
        fisher_decay_rates = [d['fisher_decay_rate'] for d in recent_decisions if d['fisher_decay_rate'] is not None]
        
        # 统计主方向跟踪
        primary_directions = [d['primary_direction'] for d in recent_decisions if d['primary_direction'] is not None]
        primary_strengths = [d['primary_strength'] for d in recent_decisions if d['primary_strength'] > 0]
        
        return {
            'total_decisions': len(self.decision_history),
            'recent_strategy_distribution': strategy_counts,
            'average_angle_change': np.mean(angle_changes) if angle_changes else 0,
            'current_gaze_angle': self.current_gaze_angle,
            'adaptive_mode': self.adaptive_mode,
            'average_fisher_decay_rate': np.mean(fisher_decay_rates) if fisher_decay_rates else 0,
            'primary_direction_count': len(primary_directions),
            'average_primary_strength': np.mean(primary_strengths) if primary_strengths else 0,
            'current_primary_direction': self.current_primary_direction.angle if self.current_primary_direction else None,
            'current_secondary_direction': self.current_secondary_direction.angle if self.current_secondary_direction else None
        }


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description="增强版贪心Fisher信息代理 v2.0 - 基于Fisher方向分析器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 运行控制参数
    parser.add_argument('--headless', action='store_true', 
                       help='禁用可视化，以无头模式运行')
    parser.add_argument('--realtime', action='store_true',
                       help='以实时速度运行')
    parser.add_argument('--epochs', type=int, default=2,
                       help='运行的轮数（每轮结束后重置环境）')
    parser.add_argument('--steps', type=int, default=1000,
                       help='每轮的最大步数')
    
    # 环境参数
    parser.add_argument('--world-size', type=float, default=50.0,
                       help='世界大小（米）')
    parser.add_argument('--fov-angle', type=float, default=90.0,
                       help='FOV角度（度）')
    parser.add_argument('--fov-distance', type=float, default=12.5,
                       help='FOV距离（米）')
    
    # Fisher分析器参数
    parser.add_argument('--fisher-threshold-ratio', type=float, default=0.2,
                       help='Fisher分析器阈值比例')
    parser.add_argument('--fisher-min-points', type=int, default=10,
                       help='Fisher分析器最小点数')
    
    # 贪心策略参数
    parser.add_argument('--alignment-weight', type=float, default=0.6,
                       help='方向对齐权重')
    parser.add_argument('--anti-decay-weight', type=float, default=0.3,
                       help='防衰减权重')
    parser.add_argument('--smoothness-weight', type=float, default=0.1,
                       help='平滑性权重')
    parser.add_argument('--max-angle-change', type=float, default=30.0,
                       help='最大角度变化（度）')
    
    # 运动控制参数
    parser.add_argument('--velocity-change-interval', type=int, default=50,
                       help='速度变化间隔（步数）')
    parser.add_argument('--status-report-interval', type=int, default=20,
                       help='状态报告间隔（步数）')
    parser.add_argument('--linear-vel-range', type=float, nargs=2, default=[-2.0, 3.0],
                       help='线速度范围 [min, max] (m/s)')
    parser.add_argument('--angular-vel-range', type=float, nargs=2, default=[-0.8, 0.8],
                       help='角速度范围 [min, max] (rad/s)')
    
    # 其他参数
    parser.add_argument('--adaptive-mode', action='store_true', default=True,
                       help='启用自适应模式')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    
    return parser.parse_args()


def run_greedy_fisher_test(args):
    """
    运行贪心Fisher算法测试
    
    Args:
        args: 命令行参数
    """
    print("🎯 增强版贪心Fisher信息代理 v3.0")
    print("=" * 60)
    print(f"🚀 开始测试 - 轮数: {args.epochs}, 每轮步数: {args.steps}")
    print(f"📺 可视化模式: {'禁用 (无头模式)' if args.headless else '启用'}")
    print(f"🌍 世界尺寸: {args.world_size}m x {args.world_size}m")
    print("=" * 60)
    
    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"🌱 随机种子设置为: {args.seed}")
    
    # 创建模拟器核心
    core = RobotCore(
        world_width=args.world_size,
        world_height=args.world_size,
        fov_angle=args.fov_angle,
        fov_distance=args.fov_distance,
        control_frequency=5.0
    )
    
    # 创建渲染器（如果需要）
    if not args.headless:
        renderer = RobotRenderer(core, render_mode="human")
    else:
        renderer = None
    
    # 创建代理
    agent = EnhancedGreedyFisherAgent(
        fisher_threshold_ratio=args.fisher_threshold_ratio,
        fisher_min_points=args.fisher_min_points,
        fov_angle=args.fov_angle,
        alignment_weight=args.alignment_weight,
        anti_decay_weight=args.anti_decay_weight,
        smoothness_weight=args.smoothness_weight,
        max_angle_change=args.max_angle_change,
        adaptive_mode=args.adaptive_mode
    )
    
    # 统计信息
    total_steps = 0
    total_decisions = 0
    epoch_stats = []
    
    # 时间控制
    start_real = time.time()
    expected_sim_t = 0.0
    
    try:
        for epoch in range(args.epochs):
            print(f"\n🔄 开始第 {epoch + 1}/{args.epochs} 轮")
            # 重置环境
            core.reset(regenerate_map=True)
            state = core.state()
            print(f"   环境初始化完成，机器人位置: [{state['position'][0]:.2f}, {state['position'][1]:.2f}]")
            epoch_step = 0
            epoch_fisher_collected = 0
            
            for step in range(args.steps):
                # 外部速度控制
                if step % args.velocity_change_interval == 0:
                    linear_vel = np.random.uniform(args.linear_vel_range[0], args.linear_vel_range[1])
                    angular_vel = np.random.uniform(args.angular_vel_range[0], args.angular_vel_range[1])
                    core.set_velocity(linear_vel, angular_vel)
                    if step % (args.velocity_change_interval * 2) == 0:  # 减少输出频率
                        print(f"\n📍 步骤 {step}: 设置速度 [{linear_vel:.2f}m/s, {angular_vel:.2f}rad/s]")
                
                # 获取贪心算法动作
                optimal_gaze = agent.get_action(core)
                
                # 设置凝视角度
                core.set_gaze(optimal_gaze)
                
                # 执行仿真步骤
                core.step()
                core.update_maps()
                
                epoch_step += 1
                total_steps += 1
                
                # 状态报告
                if step % args.status_report_interval == 0:
                    fisher_stats = core.fisher_map_stats()
                    fov_stats = core.fov_fisher_stats()
                    state = core.state()
                    agent_stats = agent.get_statistics()
                    
                    print(f"\n📊 轮{epoch + 1} 步骤 {step}:")
                    print(f"  位置: [{state['position'][0]:.1f}, {state['position'][1]:.1f}]")
                    print(f"  凝视: {state['gaze_angle']:.1f}° -> {optimal_gaze:.1f}°")
                    print(f"  Fisher: 均值={fisher_stats['mean_fisher']:.3f}, 特征={fisher_stats['total_features']:.0f}, 密度={fisher_stats['density']:.3f}")
                    print(f"  FOV: 均值={fov_stats['mean_fisher']:.3f}, 特征={fov_stats['total_features']:.0f}")
                    
                    # 显示Fisher方向信息
                    if agent_stats.get('current_primary_direction') is not None:
                        primary_dir = agent_stats['current_primary_direction']
                        secondary_dir = agent_stats.get('current_secondary_direction')
                        print(f"  🎯 主方向: {primary_dir:.1f}°", end="")
                        if secondary_dir is not None:
                            print(f", 次方向: {secondary_dir:.1f}°")
                        else:
                            print()
                        print(f"  📈 衰减率: {agent_stats.get('average_fisher_decay_rate', 0):.4f}")
                    else:
                        print(f"  🎯 无有效Fisher方向")
                
                # 累计Fisher特征数
                current_fisher_stats = core.fisher_map_stats()
                epoch_fisher_collected += current_fisher_stats['total_features']
                
                # 渲染（如果启用）
                if renderer:
                    renderer.render()
                
                # 时间控制
                if args.realtime:
                    expected_sim_t += core.dt
                    now = time.time() - start_real
                    sleep_t = expected_sim_t - now
                    if sleep_t > 0:
                        time.sleep(sleep_t)
                else:
                    # 非实时：轻微睡眠便于 UI
                    if renderer:
                        time.sleep(0.01)
                
                # UI退出检查
                if renderer:
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q') or k == 27:
                        print("\n⏹️ 用户请求退出")
                        return
            
            # 轮次统计
            epoch_stats.append({
                'epoch': epoch + 1,
                'steps': epoch_step,
                'fisher_features': epoch_fisher_collected,
                'avg_fisher_per_step': epoch_fisher_collected / max(epoch_step, 1)
            })
            
            print(f"\n✅ 第 {epoch + 1} 轮完成:")
            print(f"   步骤数: {epoch_step}")
            print(f"   Fisher特征总数: {epoch_fisher_collected:.0f}")
            print(f"   平均每步Fisher: {epoch_fisher_collected / max(epoch_step, 1):.2f}")
        
        # 最终统计
        final_agent_stats = agent.get_statistics()
        total_decisions = final_agent_stats.get('total_decisions', 0)
        
        print(f"\n🏆 所有轮次完成统计:")
        print(f"=" * 60)
        print(f"  总轮数: {args.epochs}")
        print(f"  总步数: {total_steps}")
        print(f"  总决策次数: {total_decisions}")
        print(f"  策略分布: {final_agent_stats.get('recent_strategy_distribution', {})}")
        print(f"  平均角度变化: {final_agent_stats.get('average_angle_change', 0):.1f}°")
        print(f"  主方向检测次数: {final_agent_stats.get('primary_direction_count', 0)}")
        print(f"  平均主方向强度: {final_agent_stats.get('average_primary_strength', 0):.3f}")
        print(f"  平均Fisher衰减率: {final_agent_stats.get('average_fisher_decay_rate', 0):.4f}")
        
        # 各轮次详细统计
        if args.epochs > 1:
            print(f"\n📊 各轮次详细统计:")
            for stat in epoch_stats:
                print(f"  轮{stat['epoch']}: {stat['steps']}步, "
                      f"Fisher={stat['fisher_features']:.0f}, "
                      f"平均={stat['avg_fisher_per_step']:.2f}/步")
        
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if renderer:
            cv2.destroyAllWindows()
        print("✅ 测试完成")


if __name__ == "__main__":
    """
    增强版贪心Fisher信息算法的命令行测试接口
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 运行测试
    run_greedy_fisher_test(args)
