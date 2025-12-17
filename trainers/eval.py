# trainers/eval.py
import argparse
import importlib
import yaml
import torch
import os
import numpy as np
from env.uav_env import UAVEnv

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_seeds_arg(seeds_arg, default_seed):
    """
    seeds_arg: None or comma-separated string "101,102" or single int string "36"
    returns: list of ints
    """
    if seeds_arg is None:
        return [int(default_seed)]
    if isinstance(seeds_arg, int):
        return [int(seeds_arg)]
    s = str(seeds_arg).strip()
    if ',' in s:
        parts = [p.strip() for p in s.split(',') if p.strip()]
        return [int(p) for p in parts]
    # single value
    return [int(s)]

def checksum(net):
    s = 0.0
    for p in net.parameters():
        s += float(p.detach().cpu().sum().item())
    return s

def evaluate_single_seed(env_cfg, agent_cfg, train_cfg, AgentClass, checkpoint_path,
                         num_episodes, seed):
    """
    Evaluate one seed for num_episodes episodes.
    Returns: dict with per-episode rates, per-episode rewards, failure_count
    """
    env_kwargs = env_cfg.copy()
    env_kwargs.pop('obs_dim', None)
    env_kwargs.pop('act_dim', None)
    # make sure regen flag follows env config
    env_kwargs['regen_obstacles_per_episode'] = env_cfg.get('regen_obstacles_per_episode', False)

    env_kwargs.pop('seed', None)
    # Important: construct env with seed so initial obstacle generation (in __init__) is seeded
    # env = UAVEnv(seed=int(seed), **env_kwargs)
    env_kwargs['test_mode'] = True

    # Agent init and load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = AgentClass(
        obs_dim=env_cfg['obs_dim'],
        act_dim=env_cfg['act_dim'],
        seq_len=env_cfg['observation_seq_len'],
        config=agent_cfg,
        device=device,
        env_fn=None
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    # load available keys robustly
    if 'encoder' in ckpt and hasattr(agent, 'encoder'):
        agent.encoder.load_state_dict(ckpt['encoder'])
    if 'actor' in ckpt:
        agent.actor.load_state_dict(ckpt['actor'])
    if hasattr(agent, 'critic') and 'critic' in ckpt:
        agent.critic.load_state_dict(ckpt['critic'])
    if 'actor2' in ckpt and hasattr(agent, 'actor2'):
        agent.actor2.load_state_dict(ckpt['actor2'])
    if 'critic2' in ckpt and hasattr(agent, 'critic2'):
        agent.critic2.load_state_dict(ckpt['critic2'])
    if 'log_alpha' in ckpt and hasattr(agent, 'log_alpha'):
        agent.log_alpha = ckpt['log_alpha'].to(device)

    agent.device = device
    agent.to(device) if hasattr(agent, 'to') else None
    try:
        agent.actor.eval()
        if hasattr(agent, 'encoder'):
            agent.encoder.eval()
    except Exception:
        pass

    # debug prints
    print(f"[DEBUG] Using checkpoint: {checkpoint_path}")
    print(f"[DEBUG] env_kwargs: num_obstacles={env_kwargs.get('num_obstacles')}, p_drop={env_kwargs.get('p_drop')}")
    try:
        print(f"[DEBUG] actor param checksum: {checksum(agent.actor):.6e}")
    except Exception:
        pass

    # Basic episode / eval params
    T = train_cfg['max_steps']
    M = 100
    Ri, Ro = 12.0, 25.0

    per_episode_rates = []
    per_episode_rewards = []
    total_fail = 0

    # For each episode, to ensure full determinism we recreate env with per-episode seed
    # seed_base can be seed, seed+1, ... We'll create per-episode envs seeded deterministically.
    # If env_cfg has 'regen_obstacles_per_episode'=True we could reuse env and seed reset,
    # but to be safe & reproducible create a new env per episode seeded by (seed + ep).
    for ep in range(num_episodes):
        ep_seed = int(seed + ep)
        env_ep = UAVEnv(seed=ep_seed, **env_kwargs)

        obs_seq = env_ep.reset()  # reset uses seed from constructor

        zone_flags = []
        failed = False
        ep_reward = 0.0

        for t in range(T):
            # deterministic action
            action = agent.select_action(obs_seq, deterministic=True)
            # add action debug
            if ep == 0 and t < 5:
                print(f"[DEBUG] Episode {ep}, Step {t}: Action = {action}")
            next_seq, reward, done, _, info = env_ep.step(action)
            obs_seq = next_seq
            ep_reward += float(reward)

            d = info.get('dist')
            zone_flags.append(1 if (d is not None and Ri < d <= Ro) else 0)

            if info.get('collision') or info.get('out_of_bounds'):
                total_fail += 1
                failed = True
                break

            if done:
                break

        # compute tracking rate (last M steps)
        if not failed and len(zone_flags) > 0:
            last_m = zone_flags[-M:] if len(zone_flags) >= M else zone_flags
            ep_rate = sum(last_m) / float(len(last_m))
        else:
            ep_rate = 0.0

        per_episode_rates.append(float(ep_rate))
        per_episode_rewards.append(float(ep_reward))

        if (ep + 1) % 100 == 0:
            print(f"[Eval seed {seed}] ep {ep + 1}/{num_episodes}")

    return {
        'seed': seed,
        'per_episode_rates': per_episode_rates,
        'per_episode_rewards': per_episode_rewards,
        'failure_count': total_fail
    }

def summarize_array(xs):
    xs = np.array(xs, dtype=float)
    n = len(xs)
    mean = float(np.mean(xs)) if n > 0 else 0.0
    std = float(np.std(xs, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    return mean, std, (ci_low, ci_high), n

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_config',       type=str, required=True)
    parser.add_argument('--agent_config',     type=str, required=True)
    parser.add_argument('--train_config',     type=str, required=True)
    parser.add_argument('--agent_type',       required=True,
                        choices=[
                            'itemarl_td3',
                            'tosrl_sac',
                            'tosrl_td3',
                            'tosrl_td3bc',
                            'lstm_td3',
                            'lstm_sac',
                        ])
    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser.add_argument('--episodes',         type=int, default=1000)
    parser.add_argument('--seed',             type=int, default=36,
                        help='base seed (used if --seeds not provided)')
    parser.add_argument('--seeds',            type=str, default=None,
                        help='comma-separated list of seeds, e.g. "101,102,103" (overrides --seed)')
    parser.add_argument('--seq_len',          type=int, help='Override observation_seq_len')
    parser.add_argument('--p_drop',           type=float, help='Override p_drop')
    parser.add_argument('--num_obstacles',    type=int, default=None,
                        help='Set number of obstacles for eval (overrides env.yaml num_obstacles)')

    args = parser.parse_args()

    env_cfg   = load_config(args.env_config)
    agent_cfg = load_config(args.agent_config)
    train_cfg = load_config(args.train_config)

    if args.seq_len is not None:
        env_cfg['observation_seq_len'] = args.seq_len
    if args.p_drop is not None:
        env_cfg['p_drop'] = args.p_drop
    if args.num_obstacles is not None:
        env_cfg['num_obstacles'] = int(args.num_obstacles)
        print(f"[Eval] override env.num_obstacles -> {env_cfg['num_obstacles']}")

    agent_map = {
        'itemarl_td3': ('agents.itemarl_td3_agent', 'ITEMARLTD3Agent'),
        'tosrl_sac': ('agents.tosrl_sac_agent', 'TOSRLSACAgent'),
        'tosrl_td3': ('agents.tosrl_td3_agent', 'TOSRLTD3Agent'),
        'tosrl_td3bc': ('agents.tosrl_td3_bc_agent', 'TOSRLTD3BCAgent'),
        'lstm_td3': ('agents.lstm_td3_agent', 'LSTMTD3Agent'),
        'lstm_sac': ('agents.lstm_sac_agent', 'LSTMSACAgent'),
    }
    if args.agent_type not in agent_map:
        raise ValueError("Unknown agent_type")
    module_path, class_name = agent_map[args.agent_type]
    mod    = importlib.import_module(module_path)
    Agent = getattr(mod, class_name)

    # parse seeds list
    seed_list = parse_seeds_arg(args.seeds, args.seed)

    # prepare outputs (checkpoint dir logs)
    ckpt_dir = os.path.dirname(args.model_checkpoint)
    root_dir = os.path.abspath(os.path.join(ckpt_dir, ".."))
    eval_log_dir = os.path.join(root_dir, "logs")
    os.makedirs(eval_log_dir, exist_ok=True)

    all_seed_summaries = []
    master_rows = []  # collect tuples (seed, episode, tracking_rate, reward) for pooled CSV & stats

    import random
    # Evaluate each seed separately
    for seed in seed_list:
        print(f"\n=== Evaluating {args.agent_type} at seed {seed} for {args.episodes} episodes ===")
        # ---- set global seeds for full reproducibility for this seed ----
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        result = evaluate_single_seed(env_cfg, agent_cfg, train_cfg, Agent, args.model_checkpoint,
                                      num_episodes=args.episodes, seed=seed)

        # write per-episode CSV for tracking rate
        csv_file = os.path.join(eval_log_dir, f"eval_per_episode_seed{seed}_eps{args.episodes}.csv")
        with open(csv_file, "w") as cf:
            cf.write("episode,tracking_rate,reward\n")
            for i, (r_rate, r_reward) in enumerate(zip(result['per_episode_rates'], result['per_episode_rewards'])):
                cf.write(f"{i},{r_rate:.6f},{r_reward:.6f}\n")
                master_rows.append((seed, i, r_rate, r_reward))
        print(f"[Eval] per-episode CSV -> {csv_file}")

        # compute summaries for both tracking rate and reward
        tr_mean, tr_std, tr_ci, tr_n = summarize_array(result['per_episode_rates'])
        rw_mean, rw_std, rw_ci, rw_n = summarize_array(result['per_episode_rewards'])
        fail_c = int(result['failure_count'])
        fail_r = float(fail_c) / float(args.episodes) if args.episodes > 0 else 0.0

        seed_summary = {
            'seed': seed,
            'episodes': args.episodes,
            'tracking_mean': tr_mean,
            'tracking_std': tr_std,
            'tracking_95ci': tr_ci,
            'reward_mean': rw_mean,
            'reward_std': rw_std,
            'reward_95ci': rw_ci,
            'failure_count': fail_c,
            'failure_rate': fail_r,
            'per_episode_rates': result['per_episode_rates'],
            'per_episode_rewards': result['per_episode_rewards']
        }
        all_seed_summaries.append(seed_summary)

        # append human-readable small summary per seed
        out_file = os.path.join(eval_log_dir, f"eval_summary_seed{seed}.txt")
        with open(out_file, "w") as f:
            f.write(f"seed: {seed}\n")
            f.write(f"episodes: {args.episodes}\n")
            f.write(f"tracking_mean: {tr_mean:.6f}\n")
            f.write(f"tracking_std:  {tr_std:.6f}\n")
            f.write(f"tracking_95ci: [{tr_ci[0]:.6f}, {tr_ci[1]:.6f}]\n")
            f.write(f"reward_mean: {rw_mean:.6f}\n")
            f.write(f"reward_std:  {rw_std:.6f}\n")
            f.write(f"reward_95ci: [{rw_ci[0]:.6f}, {rw_ci[1]:.6f}]\n")
            f.write(f"failure_count: {fail_c}\n")
            f.write(f"failure_rate:  {fail_r:.6f}\n")
        print(f"[Eval] summary -> {out_file}")


    # write combined per-episode CSV (seed,episode,tracking_rate,reward)
    combined_csv = os.path.join(eval_log_dir, "eval_per_episode_all_seeds.csv")
    with open(combined_csv, "w") as cf:
        cf.write("seed,episode,tracking_rate,reward\n")
        for seed, ep, trv, rwv in master_rows:
            cf.write(f"{seed},{ep},{trv:.6f},{rwv:.6f}\n")
    print(f"[Eval] combined per-episode CSV -> {combined_csv}")

    # ---------------------------
    # POOLED (all seeds) statistics
    # ---------------------------
    all_rates = np.concatenate([np.array(s['per_episode_rates'], dtype=float) for s in all_seed_summaries]) if all_seed_summaries else np.array([], dtype=float)
    all_rewards = np.concatenate([np.array(s['per_episode_rewards'], dtype=float) for s in all_seed_summaries]) if all_seed_summaries else np.array([], dtype=float)

    pooled_tr_mean, pooled_tr_std, pooled_tr_ci, pooled_tr_n = summarize_array(all_rates)
    pooled_rw_mean, pooled_rw_std, pooled_rw_ci, pooled_rw_n = summarize_array(all_rewards)

    # Seed-level means (each seed -> mean), gives between-seed variability
    seed_tr_means = np.array([s['tracking_mean'] for s in all_seed_summaries], dtype=float) if all_seed_summaries else np.array([])
    seed_rw_means = np.array([s['reward_mean'] for s in all_seed_summaries], dtype=float) if all_seed_summaries else np.array([])

    seedlevel_tr_mean = float(np.mean(seed_tr_means)) if seed_tr_means.size > 0 else 0.0
    seedlevel_tr_std  = float(np.std(seed_tr_means, ddof=1)) if seed_tr_means.size > 1 else 0.0
    seedlevel_rw_mean = float(np.mean(seed_rw_means)) if seed_rw_means.size > 0 else 0.0
    seedlevel_rw_std  = float(np.std(seed_rw_means, ddof=1)) if seed_rw_means.size > 1 else 0.0

    # master summary file
    master_out = os.path.join(eval_log_dir, "eval_summary_all_seeds.txt")
    with open(master_out, "w") as mf:
        mf.write(f"model_checkpoint: {args.model_checkpoint}\n")
        mf.write(f"agent_type: {args.agent_type}\n")
        mf.write(f"seeds: {seed_list}\n\n")

        mf.write("=== Per-seed summaries ===\n")
        for s in all_seed_summaries:
            mf.write(f"--- seed {s['seed']} ---\n")
            mf.write(f"episodes: {s['episodes']}\n")
            mf.write(f"tracking_mean: {s['tracking_mean']:.6f}\n")
            mf.write(f"tracking_std:  {s['tracking_std']:.6f}\n")
            mf.write(f"tracking_95ci: [{s['tracking_95ci'][0]:.6f}, {s['tracking_95ci'][1]:.6f}]\n")
            mf.write(f"reward_mean: {s['reward_mean']:.6f}\n")
            mf.write(f"reward_std:  {s['reward_std']:.6f}\n")
            mf.write(f"reward_95ci: [{s['reward_95ci'][0]:.6f}, {s['reward_95ci'][1]:.6f}]\n")
            mf.write(f"failure_count: {s['failure_count']}\n")
            mf.write(f"failure_rate: {s['failure_count']/float(s['episodes']):.6f}\n\n")

        mf.write("=== POOLED (all seeds, concatenate all episodes) ===\n")
        mf.write(f"total_samples (episodes pooled): {pooled_tr_n}\n")
        mf.write(f"pooled_tracking_mean: {pooled_tr_mean:.6f}\n")
        mf.write(f"pooled_tracking_std:  {pooled_tr_std:.6f}\n")
        mf.write(f"pooled_tracking_95ci: [{pooled_tr_ci[0]:.6f}, {pooled_tr_ci[1]:.6f}]\n\n")
        mf.write(f"total_reward_samples: {pooled_rw_n}\n")
        mf.write(f"pooled_reward_mean: {pooled_rw_mean:.6f}\n")
        mf.write(f"pooled_reward_std:  {pooled_rw_std:.6f}\n")
        mf.write(f"pooled_reward_95ci: [{pooled_rw_ci[0]:.6f}, {pooled_rw_ci[1]:.6f}]\n\n")

        mf.write("=== Seed-level statistics (treat seed means as samples) ===\n")
        mf.write(f"num_seeds: {len(all_seed_summaries)}\n")
        mf.write(f"seedlevel_tracking_mean_of_means: {seedlevel_tr_mean:.6f}\n")
        mf.write(f"seedlevel_tracking_std_of_means:  {seedlevel_tr_std:.6f}\n")
        mf.write(f"seedlevel_reward_mean_of_means:   {seedlevel_rw_mean:.6f}\n")
        mf.write(f"seedlevel_reward_std_of_means:    {seedlevel_rw_std:.6f}\n")
        mf.write(f"seedlevel_tracking_mean ± std:    {seedlevel_tr_mean:.4f} ± {seedlevel_tr_std:.4f}\n")
        mf.write(f"seedlevel_reward_mean ± std:    {seedlevel_rw_mean:.4f} ± {seedlevel_rw_std:.4f}\n")

    print(f"\nCombined summary -> {master_out}")