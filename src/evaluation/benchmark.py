"""
Comparative Policy Benchmark Suite.
Evaluates ChemoRL (Vision-RL) against Standard Fixed Full Dose, Reactive MTD,
and Metronomic Chemotherapy strategies on the exact same patient cohorts.
"""

import numpy as np
from typing import List, Dict, Any, Callable
from ..data.online_loader import PatientProfile
from ..models.pk_pd_simulator import ChemotherapyEnvironment
from .recist import evaluate_recist_1_1
from .ctcae import grade_toxicity_ctcae

class PolicyEvaluator:
    """Simulates and evaluates clinical policies across a patient cohort."""
    
    @staticmethod
    def evaluate_fixed_dose(cohort: List[PatientProfile], fixed_dose: float = 1.0) -> Dict[str, Any]:
        """Baseline 1: Fixed dose each cycle (e.g. 1.0x)."""
        env = ChemotherapyEnvironment()
        results = []

        for p in cohort:
            obs, info = env.reset(p)
            done = False
            min_diam = p.baseline_diameter_mm
            max_tox = 0.0

            while not done:
                _, _, term, trunc, sinfo = env.step(fixed_dose)
                done = term or trunc
                min_diam = min(min_diam, sinfo["diameter_mm"])
                max_tox = max(max_tox, sinfo["toxicity"])

            final_diam = env.current_diameter_mm
            recist = evaluate_recist_1_1(p.baseline_diameter_mm, min_diam, final_diam)
            ctcae = grade_toxicity_ctcae(max_tox)

            results.append({
                "patient_id": p.patient_id,
                "baseline_diam": p.baseline_diameter_mm,
                "final_diam": final_diam,
                "reduction_pct": ((final_diam - p.baseline_diameter_mm) / p.baseline_diameter_mm) * 100.0,
                "max_toxicity": max_tox,
                "recist": recist,
                "ctcae": ctcae,
                "history": env.history
            })

        return PolicyEvaluator._aggregate_metrics("Fixed Full Dose (1.0x)", results)

    @staticmethod
    def evaluate_mtd_reactive(cohort: List[PatientProfile]) -> Dict[str, Any]:
        """Baseline 2: Maximum Tolerated Dose (starts at 1.0x, cuts to 0.75x/0.5x if G3+ toxicity appears)."""
        env = ChemotherapyEnvironment()
        results = []

        for p in cohort:
            obs, info = env.reset(p)
            done = False
            current_dose = 1.0
            min_diam = p.baseline_diameter_mm
            max_tox = 0.0

            while not done:
                _, _, term, trunc, sinfo = env.step(current_dose)
                done = term or trunc
                min_diam = min(min_diam, sinfo["diameter_mm"])
                max_tox = max(max_tox, sinfo["toxicity"])

                # Reactive dose modification protocol
                if sinfo["toxicity"] >= 2.8: # Grade 3 toxicity
                    current_dose = max(0.25, current_dose - 0.25)

            final_diam = env.current_diameter_mm
            recist = evaluate_recist_1_1(p.baseline_diameter_mm, min_diam, final_diam)
            ctcae = grade_toxicity_ctcae(max_tox)

            results.append({
                "patient_id": p.patient_id,
                "baseline_diam": p.baseline_diameter_mm,
                "final_diam": final_diam,
                "reduction_pct": ((final_diam - p.baseline_diameter_mm) / p.baseline_diameter_mm) * 100.0,
                "max_toxicity": max_tox,
                "recist": recist,
                "ctcae": ctcae,
                "history": env.history
            })

        return PolicyEvaluator._aggregate_metrics("Reactive MTD (Protocol)", results)

    @staticmethod
    def evaluate_metronomic(cohort: List[PatientProfile], low_dose: float = 0.35) -> Dict[str, Any]:
        """Baseline 3: Metronomic Continuous Low-Dose (0.35x)."""
        env = ChemotherapyEnvironment()
        results = []

        for p in cohort:
            obs, info = env.reset(p)
            done = False
            min_diam = p.baseline_diameter_mm
            max_tox = 0.0

            while not done:
                _, _, term, trunc, sinfo = env.step(low_dose)
                done = term or trunc
                min_diam = min(min_diam, sinfo["diameter_mm"])
                max_tox = max(max_tox, sinfo["toxicity"])

            final_diam = env.current_diameter_mm
            recist = evaluate_recist_1_1(p.baseline_diameter_mm, min_diam, final_diam)
            ctcae = grade_toxicity_ctcae(max_tox)

            results.append({
                "patient_id": p.patient_id,
                "baseline_diam": p.baseline_diameter_mm,
                "final_diam": final_diam,
                "reduction_pct": ((final_diam - p.baseline_diameter_mm) / p.baseline_diameter_mm) * 100.0,
                "max_toxicity": max_tox,
                "recist": recist,
                "ctcae": ctcae,
                "history": env.history
            })

        return PolicyEvaluator._aggregate_metrics("Metronomic Low-Dose (0.35x)", results)

    @staticmethod
    def evaluate_vision_rl(cohort: List[PatientProfile], agent: Any, vision_encoder: Any) -> Dict[str, Any]:
        """Proposed: ChemoRL Vision-Guided Deep Reinforcement Learning Policy."""
        env = ChemotherapyEnvironment()
        results = []

        vision_encoder.eval()
        for p in cohort:
            obs, info = env.reset(p)
            done = False
            min_diam = p.baseline_diameter_mm
            max_tox = 0.0

            # Extract initial CT slice vision embedding
            import torch
            with torch.no_grad():
                img_t = torch.from_numpy(p.ct_slice).float().unsqueeze(0)
                v_feat = vision_encoder(img_t)

            while not done:
                c_feat = torch.from_numpy(obs).float().unsqueeze(0)
                action_idx = agent.select_action(v_feat, c_feat, eval_mode=True)
                dose = agent.get_dose_fraction(action_idx)

                next_obs, _, term, trunc, sinfo = env.step(dose)
                done = term or trunc
                obs = next_obs
                min_diam = min(min_diam, sinfo["diameter_mm"])
                max_tox = max(max_tox, sinfo["toxicity"])

            final_diam = env.current_diameter_mm
            recist = evaluate_recist_1_1(p.baseline_diameter_mm, min_diam, final_diam)
            ctcae = grade_toxicity_ctcae(max_tox)

            results.append({
                "patient_id": p.patient_id,
                "baseline_diam": p.baseline_diameter_mm,
                "final_diam": final_diam,
                "reduction_pct": ((final_diam - p.baseline_diameter_mm) / p.baseline_diameter_mm) * 100.0,
                "max_toxicity": max_tox,
                "recist": recist,
                "ctcae": ctcae,
                "history": env.history
            })

        return PolicyEvaluator._aggregate_metrics("ChemoRL (Vision-Guided RL)", results)

    @staticmethod
    def _aggregate_metrics(policy_name: str, patient_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates cohort-level clinical oncology benchmarks."""
        n = len(patient_results)
        cr_count = sum(1 for r in patient_results if r["recist"]["code"] == "CR")
        pr_count = sum(1 for r in patient_results if r["recist"]["code"] == "PR")
        sd_count = sum(1 for r in patient_results if r["recist"]["code"] == "SD")
        pd_count = sum(1 for r in patient_results if r["recist"]["code"] == "PD")
        
        g3_plus_count = sum(1 for r in patient_results if r["ctcae"]["is_severe_g3_plus"])
        g4_count = sum(1 for r in patient_results if r["ctcae"]["is_life_threatening"])

        mean_reduction = np.mean([r["reduction_pct"] for r in patient_results])
        mean_toxicity = np.mean([r["max_toxicity"] for r in patient_results])

        # Objective Response Rate (ORR = CR + PR)
        orr = ((cr_count + pr_count) / n) * 100.0
        # Disease Control Rate (DCR = CR + PR + SD)
        dcr = ((cr_count + pr_count + sd_count) / n) * 100.0
        # Severe Toxicity Rate
        severe_tox_rate = (g3_plus_count / n) * 100.0

        return {
            "policy_name": policy_name,
            "cohort_size": n,
            "ORR_pct": orr,
            "DCR_pct": dcr,
            "Severe_Toxicity_G3_G4_pct": severe_tox_rate,
            "Life_Threatening_G4_pct": (g4_count / n) * 100.0,
            "Mean_Tumor_Reduction_pct": -float(mean_reduction), # Inverted so positive means shrinkage
            "Mean_Max_Toxicity": float(mean_toxicity),
            "CR_count": cr_count,
            "PR_count": pr_count,
            "SD_count": sd_count,
            "PD_count": pd_count,
            "patient_details": patient_results
        }
