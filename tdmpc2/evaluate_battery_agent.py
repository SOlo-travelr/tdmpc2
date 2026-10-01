"""Evaluate a trained TD-MPC2 battery agent on safety-relevant metrics.

Loads a checkpoint, rolls out `eval_episodes` episodes on the battery
fast-charging environment, and reports charge time, safety-constraint
violations, degradation (SOH loss) and reward. Results are appended to a shared
comparison CSV so the unsafe TD-MPC2 baseline and the Safe-TDMPC variant can be
plotted side by side (the central table/figure of the paper).

Run from this directory, e.g.:

    python evaluate_battery_agent.py task=battery-charge model_size=1 \
        episodic=true compile=false enable_wandb=false save_video=false \
        eval_episodes=20 use_cost_head=true use_safe_planner=true \
        checkpoint=<repo>/logs/battery-charge/1/safe_tdmpc/models/final.pt \
        label=safe_tdmpc
"""

import os
os.environ['LAZY_LEGACY_OP'] = '0'
import csv
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import hydra
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2


VIOLATION_KEYS = {
	"voltage": "voltage_violation",
	"temperature": "temperature_violation",
	"soc": "soc_violation",
	"current": "current_violation",
}


def _rollout_episode(agent, env, dt_seconds):
	"""Roll out one greedy episode and return per-episode safety metrics."""
	obs, done, t = env.reset(), False, 0
	total_cost = total_reward = 0.0
	violation_steps = 0
	per_violation = {name: 0 for name in VIOLATION_KEYS}
	max_temp_k = -np.inf
	max_voltage_v = -np.inf
	info = {}
	while not done:
		action = agent.act(obs, t0=t == 0, eval_mode=True)
		obs, reward, done, info = env.step(action)
		t += 1
		cost = float(info.get("cost", 0.0))
		total_cost += cost
		total_reward += float(reward)
		if cost > 0.0:
			violation_steps += 1
		for name, key in VIOLATION_KEYS.items():
			if float(info.get(key, 0.0)) > 0.0:
				per_violation[name] += 1
		max_temp_k = max(max_temp_k, float(info.get("temperature", max_temp_k)))
		max_voltage_v = max(max_voltage_v, float(info.get("voltage", max_voltage_v)))
	metrics = {
		"steps": float(t),
		"charge_time_min": float(t * dt_seconds / 60.0),
		"success": float(info.get("success", 0.0)),
		"final_soc": float(info.get("soc", 0.0)),
		"soh_loss": float(1.0 - info.get("soh", 1.0)),
		"total_reward": total_reward,
		"total_cost": total_cost,
		"violation_steps": float(violation_steps),
		"voltage_violations": float(per_violation["voltage"]),
		"temperature_violations": float(per_violation["temperature"]),
		"soc_violations": float(per_violation["soc"]),
		"current_violations": float(per_violation["current"]),
		"max_temperature_c": float(max_temp_k - 273.15),
		"max_voltage_v": float(max_voltage_v),
	}
	return metrics


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	label = cfg.get('label', None) or cfg.exp_name
	output_csv = cfg.get('comparison_csv', None) or \
		str(Path(hydra.utils.get_original_cwd()) / 'results' / 'battery' / 'tdmpc2_comparison.csv')
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored(f'Evaluating "{label}" on {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))

	env = make_env(cfg)
	dt_seconds = float(getattr(env.unwrapped.params, 'dt_seconds', 10.0))

	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found!'
	agent.load(cfg.checkpoint)

	episodes = [_rollout_episode(agent, env, dt_seconds) for _ in range(cfg.eval_episodes)]
	keys = list(episodes[0].keys())
	mean = {k: float(np.mean([ep[k] for ep in episodes])) for k in keys}

	print(colored(f'--- {label} ({cfg.eval_episodes} episodes) ---', 'yellow', attrs=['bold']))
	for k in keys:
		print(f'  {k:<24}: {mean[k]:.4g}')

	row = {'label': label, 'task': cfg.task, 'seed': cfg.seed,
		'eval_episodes': cfg.eval_episodes, **{k: mean[k] for k in keys}}
	out_path = Path(output_csv)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	write_header = not out_path.exists()
	with out_path.open('a', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=list(row.keys()))
		if write_header:
			writer.writeheader()
		writer.writerow(row)
	print(colored(f'Appended summary to {out_path}', 'green'))


if __name__ == '__main__':
	evaluate()
