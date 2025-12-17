# trainers/ablation_eval.py
import argparse
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
    return [int(s)]


def checksum(net):
    s = 0.0
    for p in net.parameters():
        s += float(p.detach().cpu().sum().item())
    return s


def evaluate_ablation_experiment(env_cfg, agent_cfg, train_cfg, experiment_config,
                                 checkpoint_path, num_episodes, seeds):
    """
    evaluate ablation experiment
    """
    from agents.ablation_base_agent import AblationITEMATD3Agent

    # prepare environment config
    env_kwargs = env_cfg.copy()
    env_kwargs.pop('obs_dim', None)
    env_kwargs.pop('act_dim', None)
    env_kwargs['regen_obstacles_per_episode'] = env_cfg.get('regen_obstacles_per_episode', False)
    env_kwargs.pop('seed', None)
    env_kwargs['test_mode'] = True

    # merge agent config with experiment config
    merged_agent_cfg = agent_cfg.copy()
    merged_agent_cfg.update({
        'transformer_block': experiment_config.get('transformer_block', 'improve'),
        'projection_block': experiment_config.get('projection_block', 'improve'),
        'num_workers': experiment_config.get('multithreaded', 3),
        'curriculum': experiment_config.get('curriculum', 'adaptive')
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize ablation agent
    agent = AblationITEMATD3Agent(
        obs_dim=env_cfg['obs_dim'],
        act_dim=env_cfg['act_dim'],
        seq_len=env_cfg['observation_seq_len'],
        config=merged_agent_cfg,
        device=device
    )

    # load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'encoder' in ckpt and hasattr(agent, 'encoder'):
        agent.encoder.load_state_dict(ckpt['encoder'])
    if 'actor' in ckpt:
        agent.actor.load_state_dict(ckpt['actor'])
    if hasattr(agent, 'critic') and 'critic' in ckpt:
        agent.critic.load_state_dict(ckpt['critic'])

    agent.device = device
    agent.to(device) if hasattr(agent, 'to') else None
    agent.actor.eval()
    if hasattr(agent, 'encoder'):
        agent.encoder.eval()

    print(f"[AblationEval] experiment config: {experiment_config}")
    print(f"[AblationEval] using checkpoint: {checkpoint_path}")

    # evaluation parameters
    T = train_cfg['max_steps']
    M = 100
    Ri, Ro = 12.0, 25.0

    all_results = []

    for seed in seeds:
        print(f"[AblationEval] evaluating seed {seed}...")

        seed_results = {
            'seed': seed,
            'per_episode_rates': [],
            'per_episode_rewards': [],
            'failure_count': 0
        }

        for ep in range(num_episodes):
            ep_seed = int(seed + ep)
            env_ep = UAVEnv(seed=ep_seed, **env_kwargs)

            obs_seq = env_ep.reset()
            zone_flags = []
            failed = False
            ep_reward = 0.0

            for t in range(T):
                action = agent.select_action(obs_seq, deterministic=True)
                next_seq, reward, done, _, info = env_ep.step(action)
                obs_seq = next_seq
                ep_reward += float(reward)

                d = info.get('dist')
                zone_flags.append(1 if (d is not None and Ri < d <= Ro) else 0)

                if info.get('collision') or info.get('out_of_bounds'):
                    seed_results['failure_count'] += 1
                    failed = True
                    break

                if done:
                    break

            # calculate tracking rate
            if not failed and len(zone_flags) > 0:
                last_m = zone_flags[-M:] if len(zone_flags) >= M else zone_flags
                ep_rate = sum(last_m) / float(len(last_m))
            else:
                ep_rate = 0.0

            seed_results['per_episode_rates'].append(float(ep_rate))
            seed_results['per_episode_rewards'].append(float(ep_reward))

        all_results.append(seed_results)

    return all_results


def summarize_ablation_results(results):
    """summarize ablation experiment results"""
    summary = {}

    # collect all data
    all_rates = []
    all_rewards = []
    all_failures = []

    for result in results:
        all_rates.extend(result['per_episode_rates'])
        all_rewards.extend(result['per_episode_rewards'])
        all_failures.append(result['failure_count'])

    # calculate summary statistics
    if all_rates:
        summary['tracking_rate_mean'] = float(np.mean(all_rates))
        summary['tracking_rate_std'] = float(np.std(all_rates))
        summary['tracking_rate_95ci'] = (
            summary['tracking_rate_mean'] - 1.96 * summary['tracking_rate_std'] / np.sqrt(len(all_rates)),
            summary['tracking_rate_mean'] + 1.96 * summary['tracking_rate_std'] / np.sqrt(len(all_rates))
        )

    if all_rewards:
        summary['reward_mean'] = float(np.mean(all_rewards))
        summary['reward_std'] = float(np.std(all_rewards))
        summary['reward_95ci'] = (
            summary['reward_mean'] - 1.96 * summary['reward_std'] / np.sqrt(len(all_rewards)),
            summary['reward_mean'] + 1.96 * summary['reward_std'] / np.sqrt(len(all_rewards))
        )

    summary['total_failures'] = sum(all_failures)
    summary['failure_rate'] = sum(all_failures) / (
                len(results) * len(results[0]['per_episode_rates'])) if results else 0

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ablation experiment evaluation')
    parser.add_argument('--ablation_config', type=str, required=True, help='ablation experiment config file path')
    parser.add_argument('--env_config', type=str, required=True, help='environment config file path')
    parser.add_argument('--agent_config', type=str, required=True, help='Agent config file path')
    parser.add_argument('--train_config', type=str, required=True, help='training config file path')
    parser.add_argument('--results_dir', type=str, required=True, help='results directory')
    parser.add_argument('--episodes', type=int, default=200, help='number of episodes to evaluate per experiment')
    parser.add_argument('--seeds', type=str, default='101,102,103', help='evaluation seeds, comma-separated')
    parser.add_argument('--experiment', type=str, help='evaluate a single experiment')
    parser.add_argument('--seq_len', type=int, help='observation sequence length')

    args = parser.parse_args()

    # load configuration files
    ablation_cfg = load_config(args.ablation_config)
    env_cfg = load_config(args.env_config)
    agent_cfg = load_config(args.agent_config)
    train_cfg = load_config(args.train_config)

    seeds = parse_seeds_arg(args.seeds, 101)
    if args.seq_len is not None:
        env_cfg['observation_seq_len'] = args.seq_len

    experiments = ablation_cfg['ablation_study']['experiments']

    # determine experiments to evaluate
    if args.experiment:
        if args.experiment in experiments:
            exp_list = [(args.experiment, experiments[args.experiment])]
        else:
            # find experiment in single_factor
            exp_list = []
            for exp in experiments.get('single_factor', []):
                if exp['name'] == args.experiment:
                    exp_list.append((args.experiment, exp))
                    break
            if not exp_list:
                raise ValueError(f"ablation experiment {args.experiment} not found")
    else:
        # evaluate all experiments
        exp_list = [('baseline', experiments['baseline'])]
        for exp in experiments.get('single_factor', []):
            exp_list.append((exp['name'], exp))

    # evaluate each experiment
    all_summaries = {}

    for exp_name, exp_config in exp_list:
        print(f"\n{'=' * 60}")
        print(f"evaluating ablation experiment: {exp_name}")
        print(f"{'=' * 60}")

        # handle keep_others configuration
        if 'keep_others' in exp_config and exp_config['keep_others'] == 'baseline':
            baseline_config = experiments['baseline'].copy()
            for key, value in exp_config.items():
                if key not in ['name', 'keep_others', 'description']:
                    baseline_config[key] = value
            exp_config = baseline_config

        # build checkpoint path
        checkpoint_path = os.path.join(args.results_dir, 'ablation', exp_name, 'checkpoints', 'latest.pt')

        if not os.path.exists(checkpoint_path):
            print(f"warning: checkpoint not found {checkpoint_path}，skip experiment {exp_name}")
            continue

        # run evaluation
        results = evaluate_ablation_experiment(
            env_cfg, agent_cfg, train_cfg, exp_config,
            checkpoint_path, args.episodes, seeds
        )

        # summarize results
        summary = summarize_ablation_results(results)
        all_summaries[exp_name] = summary

        # save detailed results
        exp_output_dir = os.path.join(args.results_dir, 'ablation', exp_name, 'eval')
        os.makedirs(exp_output_dir, exist_ok=True)

        # save results for each seed
        for result in results:
            csv_file = os.path.join(exp_output_dir, f"eval_seed{result['seed']}.csv")
            with open(csv_file, "w") as f:
                f.write("episode,tracking_rate,reward\n")
                for i, (rate, reward) in enumerate(zip(result['per_episode_rates'], result['per_episode_rewards'])):
                    f.write(f"{i},{rate:.6f},{reward:.6f}\n")

        # save summary results
        summary_file = os.path.join(exp_output_dir, "summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"experiment: {exp_name}\n")
            f.write(f"config: {exp_config}\n")
            f.write(f"tracking rate: {summary['tracking_rate_mean']:.4f} ± {summary['tracking_rate_std']:.4f}\n")
            f.write(f"reward: {summary['reward_mean']:.4f} ± {summary['reward_std']:.4f}\n")
            f.write(f"failure rate: {summary['failure_rate']:.4f}\n")
            f.write(f"total failures: {summary['total_failures']}\n")

        print(f"experiment {exp_name} evaluation completed")
        print(f"  tracking rate: {summary['tracking_rate_mean']:.4f} ± {summary['tracking_rate_std']:.4f}")
        print(f"  reward: {summary['reward_mean']:.4f} ± {summary['reward_std']:.4f}")
        print(f"  failure rate: {summary['failure_rate']:.4f}")

    # generate comparison report
    if len(all_summaries) > 1:
        print(f"\n{'=' * 60}")
        print("ablation experiment comparison report")
        print(f"{'=' * 60}")

        report_file = os.path.join(args.results_dir, 'ablation', 'ablation_comparison.txt')
        with open(report_file, "w") as f:
            f.write("ablation experiment comparison report\n")
            f.write("=" * 50 + "\n\n")

            # order experiments by tracking rate
            sorted_experiments = sorted(all_summaries.items(),
                                        key=lambda x: x[1]['tracking_rate_mean'],
                                        reverse=True)

            f.write("rank | experiment name | tracking rate | reward | failure rate\n")
            f.write("-" * 60 + "\n")

            for i, (exp_name, summary) in enumerate(sorted_experiments):
                f.write(
                    f"{i + 1:4d} | {exp_name:20} | {summary['tracking_rate_mean']:.4f} ± {summary['tracking_rate_std']:.4f} | "
                    f"{summary['reward_mean']:.2f} ± {summary['reward_std']:.2f} | {summary['failure_rate']:.4f}\n")

        print(f"ablation experiment comparison report saved to: {report_file}")

        print("\nrank | experiment name | tracking rate | reward | failure rate")
        print("-" * 60)
        for i, (exp_name, summary) in enumerate(sorted_experiments):
            print(
                f"{i + 1:4d} | {exp_name:20} | {summary['tracking_rate_mean']:.4f} ± {summary['tracking_rate_std']:.4f} | "
                f"{summary['reward_mean']:.2f} ± {summary['reward_std']:.2f} | {summary['failure_rate']:.4f}")