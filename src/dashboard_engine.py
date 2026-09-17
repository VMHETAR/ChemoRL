"""Offline research dashboard: reproducible toy rollouts and genuine Kernel SHAP.

No real images, clinical outcomes or dose recommendations are produced here.
The existing CNN is randomly initialized and frozen, not a trained CT encoder.
"""
import random

import numpy as np
import torch

from src.data.online_loader import OnlineLungCancerDataset
from src.models.pk_pd_simulator import ChemotherapyEnvironment
from src.models.rl_agent import ChemoRLAgent, DOSE_ACTIONS
from src.models.vision_encoder import CTScanVisionEncoder

FEATURES = ["Current diameter / 120", "Toy burden / 4", "Cycle / 6",
            "Baseline diameter / 120", "(Age − 60) / 20", "ECOG / 3"]


def build_experiment(seed=42, episodes=120, train_size=80, test_size=24):
    if min(episodes, train_size, test_size) < 1:
        raise ValueError("Experiment sizes must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    # Explicit synthetic mode: never attempts a dataset download.
    dataset = OnlineLungCancerDataset(train_size + test_size, source_url=None, seed=seed)
    encoder = CTScanVisionEncoder().eval()
    encoder.requires_grad_(False)
    agent = ChemoRLAgent()
    with torch.no_grad():
        images = torch.tensor(np.stack([p.ct_slice for p in dataset.patients]))
        embeddings = encoder(images).numpy()
    env = ChemotherapyEnvironment()
    background, learning = [], []
    for ep in range(episodes):
        i = ep % train_size
        obs, _ = env.reset(dataset.patients[i])
        z = embeddings[i]
        total = 0.0
        done = False
        while not done:
            background.append(obs.copy())
            action = agent.select_action(torch.tensor(z), torch.tensor(obs))
            nxt, reward, term, trunc, _ = env.step(DOSE_ACTIONS[action])
            done = term or trunc
            agent.memory.push(z, obs, action, reward, z, nxt, done)
            agent.update_model(32)
            total += reward
            obs = nxt
        learning.append(total)
        if ep % 10 == 0:
            agent.sync_target()
    agent.policy_net.eval()
    rng = np.random.default_rng(seed)
    bg = np.asarray(background)
    bg = bg[rng.choice(len(bg), min(24, len(bg)), replace=False)]
    runs = {}
    for label, fixed in [("Dueling DQN", None), ("Fixed action 1.00", 1.0),
                         ("Fixed action 0.35", 0.35), ("No action", 0.0)]:
        records = []
        for i in range(train_size, train_size + test_size):
            p, z = dataset.patients[i], embeddings[i]
            obs, _ = env.reset(p)
            states, actions, total = [], [], 0.0
            done = False
            while not done:
                states.append(obs.copy())
                a = agent.select_action(torch.tensor(z), torch.tensor(obs), eval_mode=True)
                dose = DOSE_ACTIONS[a] if fixed is None else fixed
                actions.append(a if fixed is None else -1)
                obs, reward, term, trunc, _ = env.step(dose)
                total += reward
                done = term or trunc
            records.append({"id": f"Virtual-{i - train_size + 1:03d}",
                            "image": p.ct_slice[0], "embedding": z,
                            "states": np.asarray(states), "actions": actions,
                            "history": [dict(row) for row in env.history],
                            "return": total,
                            "shrinkage": 100 * (1 - env.current_diameter_mm / p.baseline_diameter_mm),
                            "peak_burden": max(r["toxicity"] for r in env.history),
                            "completed": env.current_cycle == env.max_cycles})
        runs[label] = records
    return {"seed": seed, "episodes": episodes, "train_size": train_size,
            "test_size": test_size, "agent": agent, "background": bg,
            "runs": runs, "learning": learning}


def explain_state(experiment, subject=0, cycle=0, action=None):
    """Explain one fixed action Q-value, holding the random image embedding fixed.

    Kernel SHAP enumerates coalitions of six state features, against 24 sampled
    training states. This is not probability, causal effect, or image attribution.
    """
    import shap
    record = experiment["runs"]["Dueling DQN"][subject]
    state = record["states"][cycle]
    action = record["actions"][cycle] if action is None else int(action)
    if not 0 <= action < len(DOSE_ACTIONS):
        raise ValueError("Invalid action")
    model = experiment["agent"].policy_net
    z = record["embedding"]

    def predict(rows):
        rows = np.asarray(rows, dtype=np.float32)
        with torch.no_grad():
            vision = torch.tensor(np.repeat(z[None, :], len(rows), axis=0))
            return model(vision, torch.tensor(rows))[:, action].numpy().astype(float)

    explainer = shap.KernelExplainer(predict, experiment["background"])
    values = np.asarray(explainer.shap_values(state[None, :], nsamples=64,
                                             l1_reg=0.0, silent=True)).reshape(-1)
    base = float(explainer.expected_value)
    output = float(predict(state[None, :])[0])
    residual = output - base - float(values.sum())
    if not np.isfinite(values).all() or abs(residual) > 1e-4:
        raise ValueError("SHAP additivity check failed")
    return {"values": values, "base": base, "output": output, "residual": residual,
            "state": state, "action": action}
