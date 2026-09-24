from dataclasses import asdict

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner

from tasks.registry import load_env_cls, load_runner_cls


def make_runner(task, env_cfg, agent_cfg, device, log_dir=None, rank=0):
    env = load_env_cls(task)(cfg=env_cfg, device=device, rank=rank)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), log_dir, device)
    except Exception:
        wrapped.close()
        raise
    return wrapped, runner


def restore(runner, env, checkpoint, device):
    """Released checkpoints include curriculum state; do not infer it from filenames."""
    infos = runner.load(str(checkpoint), map_location=device)
    if not infos or "env_state" not in infos:
        raise ValueError("Checkpoint must include infos.env_state for curriculum-safe resume")
    env.reset()
    return infos
