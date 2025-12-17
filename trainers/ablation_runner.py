# trainers/ablation_runner.py
import os
import yaml
import subprocess
import argparse
import shutil
import tempfile
import sys
from typing import Dict, List, Any


class AblationExperimentRunner:
    """
    ablation experiment runner - using configurable base class
    """

    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = os.path.join(self.base_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """load ablation experiment config"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def run_single_experiment(self, experiment_name: str, ablation_config: Dict[str, Any],
                              seq_len: int = 12, p_drop: float = 0.2):
        """run single experiment"""
        print(f"\n{'=' * 60}")
        print(f"run ablation experiment: {experiment_name}")
        # handle keep_others config
        if 'keep_others' in ablation_config and ablation_config['keep_others'] == 'baseline':
            # get baseline config
            baseline_config = self.config['ablation_study']['experiments']['baseline']
            # merge config: baseline config + current experiment's override config
            merged_config = baseline_config.copy()
            for key, value in ablation_config.items():
                if key not in ['name', 'keep_others', 'description']:
                    merged_config[key] = value
            ablation_config = merged_config
        print(f"merged config: {ablation_config}")
        print(f"{'=' * 60}")

        # generate output directory
        output_dir = os.path.join(self.results_dir, "ablation", experiment_name)

        # generate training command
        cmd = [
            "python3", "-u", "trainers/ablation_train.py",
            "--env_config", os.path.join(self.base_dir, self.config['ablation_study']['base_config']['env_config']),
            "--agent_config", os.path.join(self.base_dir, self.config['ablation_study']['base_config']['agent_config']),
            "--train_config", os.path.join(self.base_dir, self.config['ablation_study']['base_config']['train_config']),
            "--output_dir", output_dir,
            "--seq_len", str(seq_len),
            "--p_drop", str(p_drop),
            "--agent_type", "ablation_base",
        ]

        # add ablation params
        for key, value in ablation_config.items():
            if key not in ['name', 'keep_others', 'description']:
                cmd.extend([f"--{key}", str(value)])

        print(f"run command: {' '.join(cmd)}")

        # set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = self.base_dir + (':' + env.get('PYTHONPATH', ''))

        # run training
        try:
            result = subprocess.run(cmd, check=True, cwd=self.base_dir, env=env)
            if result.returncode == 0:
                print(f"✓ experiment {experiment_name} completed successfully")
                return True
            else:
                print(f"✗ experiment {experiment_name} failed")
                return False
        except subprocess.CalledProcessError as e:
            print(f"✗ experiment {experiment_name} failed with error: {e}")
            return False

    def run_all_experiments(self, seq_len: int = 12, p_drop: float = 0.2):
        """run all ablation experiments"""
        experiments = self.config['ablation_study']['experiments']

        # run baseline experiment
        baseline_config = experiments['baseline']
        print("start running baseline experiment...")
        self.run_single_experiment("baseline", baseline_config, seq_len, p_drop)

        # run single factor ablation experiments
        print("\nstart running single factor ablation experiments...")
        for exp_config in experiments['single_factor']:
            exp_name = exp_config['name']
            self.run_single_experiment(exp_name, exp_config, seq_len, p_drop)


def main():
    parser = argparse.ArgumentParser(description="ITEMARL ablation experiment runner")
    parser.add_argument("--config", default="configs/ablation.yaml", help="ablation experiment config path")
    parser.add_argument("--seq_len", type=int, default=12, help="observation sequence length")
    parser.add_argument("--p_drop", type=float, default=0.2, help="dropout probability")
    parser.add_argument("--experiment", help="run single experiment")
    parser.add_argument("--cleanup", action="store_true", help="clean up temporary files after run")

    args = parser.parse_args()

    runner = AblationExperimentRunner(args.config)

    try:
        if args.experiment:
            # run single experiment
            experiments = runner.config['ablation_study']['experiments']
            all_experiments = {}

            # collect all experiment configs
            all_experiments['baseline'] = experiments['baseline']
            for exp in experiments['single_factor']:
                all_experiments[exp['name']] = exp

            if args.experiment in all_experiments:
                config = all_experiments[args.experiment]
                runner.run_single_experiment(args.experiment, config, args.seq_len, args.p_drop)
            else:
                print(f"unknown experiment: {args.experiment}")
        else:
            # run all experiments
            runner.run_all_experiments(args.seq_len, args.p_drop)

    finally:
        if args.cleanup:
            print("clean up completed")


if __name__ == "__main__":
    main()