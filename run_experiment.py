#!/usr/bin/env python3
"""
ChemoRL Master Experimentation & Clinical Benchmark Runner.
Executes training of Vision-Guided RL dosing policies, runs RECIST 1.1 & CTCAE v5.0 evaluation,
and outputs comparison against standard oncology baselines with publication figures.
"""

import sys
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from src.data.online_loader import OnlineLungCancerDataset, ONLINE_DATASET_SOURCES
from src.models.vision_encoder import CTScanVisionEncoder
from src.models.pk_pd_simulator import ChemotherapyEnvironment
from src.models.rl_agent import ChemoRLAgent
from src.models.generative_renderer import GenerativeDiseaseRenderer
from src.evaluation.benchmark import PolicyEvaluator
from src.utils.visualizer import plot_recist_waterfall, plot_policy_comparison_bar, plot_generative_progression_montage


def run_chemorl_experiment(
    train_patients: int = 150,
    test_patients: int = 50,
    episodes: int = 120,
    online_url: str = ONLINE_DATASET_SOURCES["nodulemnist_online"],
    device: str = "cpu",
    output_dir: str = "results"
):
    print("=" * 80)
    print(" ChemoRL: Vision-Based RL for Personalized Chemotherapy Dosing ")
    print("=" * 80)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Online / Streamed Patient Cohort
    print(f"\n[1/5] Loading patient cohort (Online link: {online_url}) ...")
    full_dataset = OnlineLungCancerDataset(
        num_patients=train_patients + test_patients,
        source_url=online_url,
        img_size=64,
        seed=42
    )

    train_cohort = full_dataset.patients[:train_patients]
    test_cohort = full_dataset.patients[train_patients:]
    print(f"Cohort Split: {len(train_cohort)} training patients, {len(test_cohort)} evaluation patients.")

    # 2. Instantiate Vision Encoder & RL Agent
    print("\n[2/5] Initializing Vision Encoder and Dueling Q-Network Agent ...")
    vision_encoder = CTScanVisionEncoder(in_channels=1, latent_dim=32).to(device)
    agent = ChemoRLAgent(vision_dim=32, clinical_dim=6, gamma=0.95, lr=1e-3, device=device)
    env = ChemotherapyEnvironment(max_cycles=6)

    # 3. Train RL Dosing Policy on Generative PK/PD Environment
    print(f"\n[3/5] Training ChemoRL Policy over {episodes} clinical simulation episodes ...")
    pbar = tqdm(range(episodes), desc="Training ChemoRL Agent")
    
    for ep in pbar:
        # Sample a patient from the training cohort
        patient = train_cohort[ep % len(train_cohort)]
        obs, info = env.reset(patient)
        
        # Extract initial visual representation z_t
        with torch.no_grad():
            img_t = torch.from_numpy(patient.ct_slice).float().unsqueeze(0).to(device)
            v_feat = vision_encoder(img_t).squeeze(0).cpu().numpy()

        done = False
        total_reward = 0.0

        while not done:
            action_idx = agent.select_action(
                torch.from_numpy(v_feat).float(),
                torch.from_numpy(obs).float()
            )
            dose_fraction = agent.get_dose_fraction(action_idx)

            next_obs, reward, terminated, truncated, sinfo = env.step(dose_fraction)
            done = terminated or truncated

            # Store transition in replay buffer
            agent.memory.push(
                v_feat,
                obs,
                action_idx,
                reward,
                v_feat, # Next vision representation
                next_obs,
                done
            )

            # Gradient update
            loss = agent.update_model(batch_size=32)
            obs = next_obs
            total_reward += reward

        if ep % 10 == 0:
            agent.sync_target()

        pbar.set_postfix({"Reward": f"{total_reward:.2f}", "Epsilon": f"{agent.epsilon:.3f}"})

    print("[Training Complete] Dueling DQN weights optimized.")

    # 4. Comparative Evaluation on Held-Out Test Cohort
    print(f"\n[4/5] Evaluating policies on {len(test_cohort)} unseen test patients ...")
    
    res_fixed = PolicyEvaluator.evaluate_fixed_dose(test_cohort, fixed_dose=1.0)
    res_mtd = PolicyEvaluator.evaluate_mtd_reactive(test_cohort)
    res_metronomic = PolicyEvaluator.evaluate_metronomic(test_cohort, low_dose=0.35)
    res_chemorl = PolicyEvaluator.evaluate_vision_rl(test_cohort, agent, vision_encoder)

    benchmarks = [res_chemorl, res_mtd, res_fixed, res_metronomic]

    # Print Clinical Benchmark Table
    print("\n" + "=" * 92)
    print(f"{'Strategy / Policy':<30} | {'ORR (%)':<8} | {'DCR (%)':<8} | {'G3+ Tox (%)':<12} | {'Mean Shrink (%)':<16}")
    print("-" * 92)
    for b in benchmarks:
        print(f"{b['policy_name']:<30} | {b['ORR_pct']:>6.1f}% | {b['DCR_pct']:>6.1f}% | {b['Severe_Toxicity_G3_G4_pct']:>10.1f}% | {b['Mean_Tumor_Reduction_pct']:>14.1f}%")
    print("=" * 92)

    # 5. Generate Figures and Plots
    print("\n[5/5] Exporting clinical visualization figures ...")
    # A. RECIST Waterfall Chart
    plot_recist_waterfall(res_chemorl, output_path=os.path.join(output_dir, "recist_waterfall.png"))
    
    # B. Policy Comparison Bar Chart
    plot_policy_comparison_bar(benchmarks, output_path=os.path.join(output_dir, "policy_comparison.png"))

    # C. Generative CT Disease Progression Montage for Sample Patient
    sample_patient = test_cohort[0]
    renderer = GenerativeDiseaseRenderer(img_size=64)
    # Simulate single patient under ChemoRL
    env.reset(sample_patient)
    slices = [sample_patient.ct_slice]
    diams = [sample_patient.baseline_diameter_mm]
    doses = [0.0]

    with torch.no_grad():
        img_t = torch.from_numpy(sample_patient.ct_slice).float().unsqueeze(0).to(device)
        v_feat = vision_encoder(img_t).squeeze(0)

    obs = env._get_observation()
    for cyc in range(6):
        act_idx = agent.select_action(v_feat, torch.from_numpy(obs).float(), eval_mode=True)
        dose = agent.get_dose_fraction(act_idx)
        obs, _, term, trunc, sinfo = env.step(dose)
        d_k = sinfo["diameter_mm"]
        rendered = renderer.render_cycle_slice(
            sample_patient.ct_slice,
            sample_patient.baseline_diameter_mm,
            d_k,
            cyc + 1
        )
        slices.append(rendered)
        diams.append(d_k)
        doses.append(dose)
        if term or trunc:
            break

    plot_generative_progression_montage(
        slices,
        diams,
        doses,
        output_path=os.path.join(output_dir, "generative_progression_ct.png")
    )

    print(f"\nAll evaluation metrics and figures generated in '{output_dir}/'.")
    return benchmarks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChemoRL Execution Script")
    parser.add_argument("--episodes", type=int, default=120, help="Number of training episodes")
    parser.add_argument("--train-patients", type=int, default=150, help="Number of training patient profiles")
    parser.add_argument("--test-patients", type=int, default=50, help="Number of evaluation patient profiles")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    run_chemorl_experiment(
        train_patients=args.train_patients,
        test_patients=args.test_patients,
        episodes=args.episodes,
        device=args.device
    )
