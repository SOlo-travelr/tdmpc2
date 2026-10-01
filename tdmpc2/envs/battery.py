"""TD-MPC2 environment factory for the battery fast-charging tasks.

Bridges the standalone `battery_envs` package (an equivalent-circuit Gymnasium
environment) into TD-MPC2's four-value env API via `TDMPC2BatteryAdapter`.
"""

import sys
from pathlib import Path

# Make the repository root importable so `battery_envs` / `safety` resolve when
# TD-MPC2 is launched from the `tdmpc2/tdmpc2` working directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from battery_envs.battery_ecm_env import BatteryECMEnv, TDMPC2BatteryAdapter
from battery_envs.battery_params import BatteryParams


# Task name -> BatteryParams overrides. These mirror the standalone YAML configs
# and expose the Phase-8 uncertainty knobs through distinct task ids.
BATTERY_TASKS = {
	"battery-charge": dict(),
	"battery-thermal": dict(randomize_thermal=True),
	"battery-aging": dict(randomize_aging=True),
	"battery-robust": dict(
		randomize_thermal=True,
		randomize_aging=True,
		use_sensor_noise=True,
	),
}


def make_env(cfg):
	"""Create a TD-MPC2-compatible battery environment for the given config."""
	if cfg.task not in BATTERY_TASKS:
		raise ValueError("Unknown task:", cfg.task)
	overrides = dict(BATTERY_TASKS[cfg.task])
	params = BatteryParams(**overrides)
	env = BatteryECMEnv(params=params, normalize_action=True)
	env = TDMPC2BatteryAdapter(env)
	env.max_episode_steps = params.max_steps
	return env
