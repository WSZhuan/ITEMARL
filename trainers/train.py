# trainers/train.py
import argparse
import importlib
import yaml
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from env.uav_env import UAVEnv
from collections import deque
import random as _py_random
from torch.utils.tensorboard import SummaryWriter
import itertools
import threading

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def clear_replay_buffer_if_needed(agent):
    rb = getattr(agent, "replay_buffer", None)
    if rb is None:
        return
    if hasattr(rb, "clear"):
        try:
            rb.clear()
            return
        except Exception:
            pass
    if hasattr(rb, "storage"):
        try:
            rb.storage = []
            return
        except Exception:
            pass
    if hasattr(rb, "buffer"):
        try:
            rb.buffer = []
            return
        except Exception:
            pass
    # try __len__ and pop if it's a list-like
    try:
        if isinstance(rb, list):
            rb.clear()
            return
    except Exception:
        pass
    print("[WARN] clear_replay_buffer_if_needed: could not clear replay buffer (unknown type)")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env_config',    default='configs/env.yaml')
    p.add_argument('--agent_config',  default='configs/agent.yaml')
    p.add_argument('--train_config',  default='configs/train.yaml')
    p.add_argument('--output_dir',    required=True,
                   help='training results save root directory')
    p.add_argument('--seq_len',       type=int,   help='override observation_seq_len')
    p.add_argument('--p_drop',        type=float, help='override p_drop')
    p.add_argument('--agent_type',    required=True,
                   choices=['itemarl_td3', 'tosrl_sac', 'tosrl_td3', 'tosrl_td3bc', 'lstm_td3', 'lstm_sac'],
                   help='select agent type to train')
    args = p.parse_args()

    # Load configs
    env_cfg   = load_config(args.env_config)
    agent_cfg = load_config(args.agent_config)
    train_cfg = load_config(args.train_config)
    if args.seq_len is not None: env_cfg['observation_seq_len'] = args.seq_len
    if args.p_drop   is not None: env_cfg['p_drop']               = args.p_drop
    p_drop_val = float(env_cfg.get('p_drop', 0.0))
    if not (0.0 <= p_drop_val <= 0.8):
        raise ValueError(f"env.p_drop must be inside [0.0, 0.8]. Got {p_drop_val}. "
                         "Please adjust configs/train or CLI --p_drop.")

    env_cfg['p_drop'] = p_drop_val
    env_cfg['test_mode'] = False
    seq_len = int(env_cfg.get('observation_seq_len', 8))


    # --------------------------
    # MASTER SEED management
    # --------------------------
    # master_seed: from env_cfg['seed'] (if exists), otherwise randomly generated.
    # master_rng: used to derive sub-seeds for each env from master_seed, ensuring different envs but reproducible within the same master_seed.
    master_seed = env_cfg.get('seed', None)
    if master_seed is None:
        master_seed = int(np.random.randint(0, 2 ** 31 - 1))
    else:
        master_seed = int(master_seed)
    master_rng = np.random.RandomState(master_seed)

    # set global random state (for reproducibility)
    _py_random.seed(master_seed)
    np.random.seed(master_seed)
    torch.manual_seed(master_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(master_seed)

    print(f"[SEED] master_seed={master_seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize environment (use copy as requested)
    env_kwargs = env_cfg.copy()
    env_kwargs.pop('obs_dim', None)
    env_kwargs.pop('act_dim', None)
    # IMPORTANT: remove any fixed 'seed' from env_kwargs so env_factory won't create identical envs
    env_kwargs.pop('seed', None)
    # Create main env with a unique seed derived from master_rng
    main_env_seed = int(master_rng.randint(0, 2**31 - 1))
    env = UAVEnv(**{**env_kwargs, 'seed': main_env_seed})
    # --- ensure env_kwargs matches actual env initial state (for env_factory to use same num_obstacles) ---
    env_kwargs['num_obstacles'] = int(
        getattr(env, 'num_obstacles', env_kwargs.get('num_obstacles', env_kwargs.get('num_obstacles', 0))))


    # Curriculum settings (optional)
    curriculum = train_cfg.get('curriculum', {})
    curriculum_enabled = bool(curriculum.get('enabled', False))
    curriculum_mode = curriculum.get('mode', 'static')  # 'static' or 'adaptive'
    curriculum_strategy = curriculum.get('strategy', 'mixed')  # preserved for semantics
    if curriculum_enabled:
        stages = curriculum.get('stages', {})
        stage_obstacles = stages.get('obstacles', [])
        stage_episodes = stages.get('episodes', [])
        preserve_rb = bool(curriculum.get('preserve_replay_buffer', True))

        if len(stage_obstacles) != len(stage_episodes):
            print("[WARN] curriculum: stages.obstacles and stages.episodes length mismatch -> disabling curriculum")
            curriculum_enabled = False
        else:
            # verify sum episodes
            total_stage_eps = sum(stage_episodes)
            if total_stage_eps != train_cfg['num_episodes']:
                print(f"[WARN] curriculum total episodes ({total_stage_eps}) != train.num_episodes ({train_cfg['num_episodes']})")
            num_stages = len(stage_obstacles)
            # adaptive params
            adaptive_cfg = curriculum.get('adaptive', {})
            adaptive_N = int(adaptive_cfg.get('N', 50))
            threshold_list = adaptive_cfg.get('threshold_list', [0.0] * num_stages)
            if len(threshold_list) != num_stages:
                print("[WARN] curriculum.adaptive.threshold_list length mismatch -> filling with zeros")
                # pad/truncate
                if len(threshold_list) < num_stages:
                    threshold_list = threshold_list + [0.0] * (num_stages - len(threshold_list))
                else:
                    threshold_list = threshold_list[:num_stages]

            # initialize stage 0
            current_stage_idx = 0
            stage_episode_counter = 0
            # set initial obstacles in env
            if num_stages > 0:
                n0 = int(stage_obstacles[0])
                print(f"[Curriculum] Initialize stage 0 with {n0} obstacles (strategy={curriculum_strategy})")
                if getattr(env, '_regen_obstacles_per_episode', False):
                    # on each episode, obstacles will be regen with n0
                    env.num_obstacles = n0
                else:
                    # if not regen per episode, generate and fix layout immediately
                    env.set_num_obstacles(n0, forbidden_box=getattr(env, 'pursuer_spawn_box', None))

                # --- ensure env_kwargs matches actual env initial state (for env_factory to use same num_obstacles) ---
                env_kwargs['num_obstacles'] = n0
            # sliding window for adaptive decisions
            recent_track_window = deque(maxlen=adaptive_N)
    else:
        stage_obstacles = []
        stage_episodes = []
        preserve_rb = True
        curriculum_mode = 'static'
        current_stage_idx = -1
        stage_episode_counter = 0
        threshold_list = []
        recent_track_window = deque(maxlen=1)

    # Agent dynamic import map: ensure module/class names match your project
    agent_map = {
        'itemarl_td3': ('agents.itemarl_td3_agent', 'ITEMARLTD3Agent'),
        'tosrl_sac':           ('agents.tosrl_sac_agent',        'TOSRLSACAgent'),
        'tosrl_td3':           ('agents.tosrl_td3_agent',        'TOSRLTD3Agent'),
        'tosrl_td3bc':         ('agents.tosrl_td3_bc_agent', 'TOSRLTD3BCAgent'),
        'lstm_td3': ('agents.lstm_td3_agent', 'LSTMTD3Agent'),
        'lstm_sac': ('agents.lstm_sac_agent', 'LSTMSACAgent'),
    }
    if args.agent_type not in agent_map:
        raise ValueError("Unknown agent_type")
    module_path, class_name = agent_map[args.agent_type]
    mod = importlib.import_module(module_path)
    Agent = getattr(mod, class_name)


    # -----------------------
    # fixed K seeds env_factory（round-robin distribution）
    # -----------------------
    K = 8
    # from master_rng derive K fixed sub-seeds (for reproducibility)
    base_seeds = [int(master_rng.randint(0, 2 ** 31 - 1)) for _ in range(K)]
    seed_cycle = itertools.cycle(base_seeds)
    _seed_cycle_lock = threading.Lock()

    # print(f"[ENV FACTORY] Using fixed seed pool K={K}, seeds={base_seeds}")
    # env_factory: each call derives an independent sub-seed from master_rng (round-robin)
    def make_env_from_factory():
        # thread-safe seed cycle access
        with _seed_cycle_lock:
            sub_seed = int(next(seed_cycle))
        print(f"[ENV_FACTORY] create env with seed: {sub_seed}")
        env_f = UAVEnv(**{**env_kwargs, 'seed': sub_seed})
        initial_obs = env.reset()
        print(f"[ENV_FACTORY] env created with seed {sub_seed}, initial obs shape: {initial_obs.shape}")
        return env_f

    env_factory = make_env_from_factory

    # Try to instantiate Agent with unified signature:
    try:
        agent = Agent(
            obs_dim=env_cfg['obs_dim'],
            act_dim=env_cfg['act_dim'],
            seq_len=env_cfg['observation_seq_len'],
            config=agent_cfg,
            device=device,
            env_fn=env_factory  # pass factory; agents that don't use it may ignore it
        )
    except TypeError as e:
        # helpful fallback / clearer error message
        raise TypeError(
            "Failed to construct Agent with unified signature. "
            "Please ensure all Agent classes implement:\n"
            "def __init__(self, obs_dim, act_dim, seq_len, config, device='cpu', env_fn=None)\n"
            f"TypeError: {e}"
        )

    # Prepare output dirs and logs
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    log_dir  = os.path.join(args.output_dir, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    reward_log_file  = open(os.path.join(log_dir, 'reward.txt'),       "w")
    track_log_file   = open(os.path.join(log_dir, 'track_rate.txt'),   "w")
    failure_log_file = open(os.path.join(log_dir, 'failure_rate.txt'), "w")

    # TensorBoard writer (write into logs/tb)
    tb_logdir = os.path.join(log_dir, 'tb')
    writer = SummaryWriter(log_dir=tb_logdir)
    print(f"[TensorBoard] Logging to {tb_logdir} (run `tensorboard --logdir {tb_logdir}`)")

    # Training constants
    T      = train_cfg['max_steps']
    M      = 100
    Ri, Ro = 12.0, 25.0
    total_episodes  = train_cfg['num_episodes']
    batch_size      = train_cfg['batch_size']
    updates_per_step= train_cfg['update_per_step']

    print(f"Training {args.agent_type} for {total_episodes} episodes...")
    if curriculum_enabled:
        print(f"Curriculum enabled: mode={curriculum_mode}, strategy={curriculum_strategy}, stages={len(stage_obstacles)}")

    # Debugging traces (sub-rewards)
    # DEBUG_EPISODES = 20
    # DEBUG_STEPS    = min(T, 1000)
    # debug_rewards = {k: [[] for _ in range(DEBUG_EPISODES)]
    #                  for k in ['rT','r_shape','r_zone','rTheta','rV','rdir','robs','total']}

    fail_count = 0
    rewards_log     = []
    track_rates_log = []

    # per-stage max episodes (if curriculum static we will use stage_episodes; if adaptive we still obey stage_episodes as max)
    stage_max_eps = stage_episodes[:] if curriculum_enabled else []

    # global step counter for TensorBoard (counts env steps)
    global_step = 0
    # Main loop
    for ep in range(total_episodes):
        # In static curriculum mode we may need to switch at pre-determined boundaries
        if curriculum_enabled and curriculum_mode == 'static':
            # compute cumulative starts
            # find which stage this ep belongs to by accumulating stage_episodes
            cum = 0
            new_stage = 0
            for idx, cnt in enumerate(stage_episodes):
                if ep >= cum and ep < cum + cnt:
                    new_stage = idx
                    break
                cum += cnt
            if new_stage != current_stage_idx:
                current_stage_idx = new_stage
                stage_episode_counter = ep - cum
                nobs = int(stage_obstacles[current_stage_idx])
                print(f"[Curriculum][static] Episode {ep}: switch to stage {current_stage_idx}, set_num_obstacles({nobs})")
                if getattr(env, '_regen_obstacles_per_episode', False):
                    # randomize obstacles per-episode, but keep num_obstacles constant per-stage
                    env.num_obstacles = nobs
                else:
                    env.set_num_obstacles(nobs, forbidden_box=getattr(env, 'pursuer_spawn_box', None))

                # --- sync env_kwargs to ensure future env_factory() uses same num_obstacles ---
                env_kwargs['num_obstacles'] = nobs

                if not preserve_rb:
                    print("[Curriculum] Clearing replay buffer (preserve_replay_buffer=False).")
                    clear_replay_buffer_if_needed(agent)
                # save stage checkpoint
                ckpt = {
                    'encoder': agent.encoder.state_dict(),
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    **({'actor2': agent.actor2.state_dict()} if hasattr(agent, 'actor2') else {}),
                    'log_alpha': getattr(agent, 'log_alpha', None),
                    'episode': ep,
                    'stage': current_stage_idx
                }
                torch.save(ckpt, os.path.join(ckpt_dir, f'stage{current_stage_idx}_ep{ep}.pt'))
                torch.save(ckpt, os.path.join(ckpt_dir, 'latest.pt'))

        # Reset environment (the env uses its own RNG; you already set obstacles above on stage changes)
        obs_seq = env.reset()
        ep_reward = 0.0
        zone_flags = []
        episode_failed = False
        t = 0

        for t in range(T):
            action = agent.select_action(obs_seq)
            next_seq, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)

            # debug logging of sub-rewards
            # if ep < DEBUG_EPISODES and t < DEBUG_STEPS:
            #     for k in debug_rewards:
            #         debug_rewards[k][ep].append(info.get(k, 0.0))
            #     # Also write per-step subreward scalars to TensorBoard
            #     for k in debug_rewards:
            #         writer.add_scalar(f"subreward/{k}", float(info.get(k, 0.0)), global_step)

            # failure?
            if info.get('collision') or info.get('out_of_bounds'):
                fail_count += 1
                episode_failed = True
                break

            # tracking zone flag
            dist = info.get('dist', None)
            zone_flags.append(1 if (dist is not None and Ri < dist <= Ro) else 0)

            # store transition and update
            agent.store_transition(obs_seq, action, reward, next_seq, float(done))

            # robust access to replay buffer length
            rb_len = 0
            rb = getattr(agent, 'replay_buffer', None)
            if rb is not None:
                if hasattr(rb, 'storage'):
                    rb_len = len(rb.storage)
                elif hasattr(rb, 'buffer'):
                    rb_len = len(rb.buffer)
                else:
                    try:
                        rb_len = len(rb)  # some buffers implement __len__
                    except Exception:
                        rb_len = 0

            if rb_len >= batch_size:
                for _ in range(updates_per_step):
                    agent.update()

            obs_seq = next_seq
            global_step += 1
            if done:
                break

        # per-episode tracking rate (use last M steps, if fewer steps use actual count)
        if not episode_failed and len(zone_flags) > 0:
            last_m = zone_flags[-M:] if len(zone_flags) >= M else zone_flags
            denom = min(M, len(zone_flags))
            ep_track_rate = sum(last_m) / float(denom)
        else:
            ep_track_rate = 0.0

        # bookkeeping
        rewards_log.append(ep_reward)
        track_rates_log.append(ep_track_rate)
        reward_log_file.write(f"{ep},{ep_reward:.2f}\n")
        track_log_file.write(f"{ep},{ep_track_rate:.4f}\n")
        reward_log_file.flush()
        track_log_file.flush()

        # TensorBoard: per-episode scalars
        writer.add_scalar('episode/reward', float(ep_reward), ep)
        writer.add_scalar('episode/track_rate', float(ep_track_rate), ep)
        writer.add_scalar('episode/failed', float(1 if episode_failed else 0), ep)

        # If debug episode, also log aggregated subreward statistics for the episode
        # if ep < DEBUG_EPISODES:
        #     for k in debug_rewards:
        #         vlist = debug_rewards[k][ep]
        #         if len(vlist) > 0:
        #             writer.add_scalar(f"debug_ep/{k}_mean", float(np.mean(vlist)), ep)
        #             writer.add_scalar(f"debug_ep/{k}_max", float(np.max(vlist)), ep)
        #             writer.add_scalar(f"debug_ep/{k}_min", float(np.min(vlist)), ep)

        # Curriculum adaptive logic: check whether we should move to next stage
        if curriculum_enabled and curriculum_mode == 'adaptive':
            # update counters & recent window
            stage_episode_counter += 1
            recent_track_window.append(ep_track_rate)

            # compute forced move: if reached stage max episodes
            forced_advance = False
            max_eps_for_stage = int(stage_episodes[current_stage_idx])
            if stage_episode_counter >= max_eps_for_stage:
                forced_advance = True

            # compute performance-based move: enough recent episodes and avg >= threshold
            perf_advance = False
            if len(recent_track_window) == recent_track_window.maxlen:
                avg_recent = float(np.mean(recent_track_window))
                cur_threshold = float(threshold_list[current_stage_idx]) if current_stage_idx < len(threshold_list) else 0.0
                if avg_recent >= cur_threshold:
                    perf_advance = True

            if (forced_advance or perf_advance) and (current_stage_idx < len(stage_obstacles) - 1):
                old_idx = current_stage_idx
                current_stage_idx += 1
                # reset per-stage episode counter
                stage_episode_counter = 0
                # set new obstacle count
                nobs = int(stage_obstacles[current_stage_idx])
                print(f"[Curriculum][adaptive] Advancing stage {old_idx} -> {current_stage_idx} at episode {ep} (forced={forced_advance}, perf={perf_advance})")
                if getattr(env, '_regen_obstacles_per_episode', False):
                    # randomize obstacles per-episode, but keep num_obstacles constant per-stage
                    env.num_obstacles = nobs
                else:
                    env.set_num_obstacles(nobs, forbidden_box=getattr(env, 'pursuer_spawn_box', None))
                # --- sync env_kwargs to ensure future env_factory() uses same num_obstacles ---
                env_kwargs['num_obstacles'] = nobs
                if not preserve_rb:
                    print("[Curriculum] Clearing replay buffer (preserve_replay_buffer=False).")
                    clear_replay_buffer_if_needed(agent)
                # save stage checkpoint
                ckpt = {
                    'encoder': agent.encoder.state_dict(),
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    **({'actor2': agent.actor2.state_dict()} if hasattr(agent, 'actor2') else {}),
                    **({'critic2': agent.critic2.state_dict()} if hasattr(agent, 'critic2') else {}),
                    'log_alpha': getattr(agent, 'log_alpha', None),
                    'episode': ep,
                    'stage': current_stage_idx
                }
                torch.save(ckpt, os.path.join(ckpt_dir, f'stage{current_stage_idx}_ep{ep}.pt'))
                torch.save(ckpt, os.path.join(ckpt_dir, 'latest.pt'))

        # periodic checkpoint & logging
        if ep % train_cfg.get('log_interval', 10) == 0:
            print(f"[Ep {ep:4d}/{total_episodes}] Reward: {ep_reward:.2f}, TrackRate: {ep_track_rate:.4f}"
                  f"{' | Stage:' + str(current_stage_idx) if curriculum_enabled else ''}")
            ckpt = {
                'encoder': agent.encoder.state_dict(),
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                **({'actor2': agent.actor2.state_dict()} if hasattr(agent, 'actor2') else {}),
                'log_alpha': getattr(agent, 'log_alpha', None),
                'episode': ep,
                'stage': current_stage_idx if curriculum_enabled else None
            }
            torch.save(ckpt, os.path.join(ckpt_dir, f'ep{ep}.pt'))
            torch.save(ckpt, os.path.join(ckpt_dir, 'latest.pt'))

    # Close files
    reward_log_file.close()
    track_log_file.close()

    # failure summary
    failure_rate = fail_count / total_episodes
    failure_log_file.write(f"TotalEpisodes: {total_episodes}\n")
    failure_log_file.write(f"FailedEpisodes: {fail_count}\n")
    failure_log_file.write(f"FailureRate: {failure_rate:.3f}\n")
    failure_log_file.close()
    print(f"\n=== Overall Failure Rate: {failure_rate:.3f} ===\n")

    # save debug subreward plots & training curves
    # for ep in range(DEBUG_EPISODES):
    #     plt.figure(figsize=(8,5))
    #     for k, curve in debug_rewards.items():
    #         plt.plot(curve[ep], label=k)
    #     plt.title(f'Ep{ep} Sub-Rewards')
    #     plt.xlabel('Step'); plt.ylabel('Value')
    #     plt.legend(); plt.grid(True)
    #     plt.savefig(os.path.join(log_dir, f'debug_ep{ep}_sub.png'))
    #     plt.close()

    eps = np.arange(len(rewards_log))
    plt.figure()
    plt.plot(eps, rewards_log, label='Cumulative Reward')
    plt.xlabel('Episode'); plt.ylabel('Reward')
    plt.title('Training Cumulative Reward')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'train_reward_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(eps, track_rates_log, label='Tracking Rate', color='orange')
    plt.xlabel('Episode'); plt.ylabel('Tracking Rate')
    plt.title('Training Tracking Rate')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'train_track_rate_curve.png'))
    plt.close()

    # TensorBoard: flush & close
    writer.flush()
    writer.close()
    print(f"[TensorBoard] flushed and closed writer at {tb_logdir}")
