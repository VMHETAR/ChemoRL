"""
Generative Pharmacokinetic / Pharmacodynamic (PK/PD) & Disease Progression Simulator.
Models Gompertzian tumor volume dynamics, drug clearance, and cumulative organ toxicity
across multiple chemotherapy cycles (e.g. 6 cycles of 21-day regimens).
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from ..data.online_loader import PatientProfile

class ChemotherapyEnvironment:
    """
    Simulates patient disease dynamics under chemotherapy treatment.
    Gompertzian tumor response coupled with continuous CTCAE toxicity accumulation.
    """
    def __init__(
        self,
        max_cycles: int = 6,
        growth_rate_lambda: float = 0.12, # Gompertzian intrinsic growth constant
        carrying_capacity_mm: float = 120.0, # Asymptotic lethal tumor diameter
        toxicity_clearance_delta: float = 0.35, # Rate of bodily toxicity recovery between cycles
        lethal_toxicity_threshold: float = 4.0 # Cumulative toxicity triggering grade 4+ arrest
    ):
        self.max_cycles = max_cycles
        self.growth_rate_lambda = growth_rate_lambda
        self.carrying_capacity_mm = carrying_capacity_mm
        self.toxicity_clearance_delta = toxicity_clearance_delta
        self.lethal_toxicity_threshold = lethal_toxicity_threshold

        self.current_patient: Optional[PatientProfile] = None
        self.current_cycle: int = 0
        self.current_diameter_mm: float = 0.0
        self.current_toxicity: float = 0.0
        self.history: list = []

    def reset(self, patient: PatientProfile) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Initializes the simulation environment with a new patient profile."""
        self.current_patient = patient
        self.current_cycle = 0
        self.current_diameter_mm = float(patient.baseline_diameter_mm)
        self.current_toxicity = 0.0
        self.history = [{
            "cycle": 0,
            "diameter_mm": self.current_diameter_mm,
            "toxicity": 0.0,
            "dose": 0.0,
            "relative_change_pct": 0.0
        }]

        obs = self._get_observation()
        info = {
            "patient_id": patient.patient_id,
            "baseline_diameter_mm": patient.baseline_diameter_mm,
            "ecog_ps": patient.ecog_ps
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        """
        State observation vector:
        [current_diameter_norm, current_toxicity_norm, cycle_progress, baseline_norm, age_norm, ecog_norm]
        """
        assert self.current_patient is not None
        p = self.current_patient
        return np.array([
            self.current_diameter_mm / self.carrying_capacity_mm,
            self.current_toxicity / self.lethal_toxicity_threshold,
            float(self.current_cycle) / float(self.max_cycles),
            p.baseline_diameter_mm / self.carrying_capacity_mm,
            (p.age - 60.0) / 20.0,
            float(p.ecog_ps) / 3.0
        ], dtype=np.float32)

    def step(self, dose_fraction: float) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one chemotherapy treatment cycle (21 days).
        
        Args:
            dose_fraction: Dosage level relative to standard full dose (e.g., 0.0 to 1.25)
            
        Returns:
            next_state, reward, terminated, truncated, info
        """
        assert self.current_patient is not None
        p = self.current_patient
        self.current_cycle += 1

        prev_diameter = self.current_diameter_mm
        baseline_diam = p.baseline_diameter_mm

        # 1. Pharmacodynamic Tumor Reduction (Norton-Simon / Gompertzian balance)
        # Kill rate is proportional to dose, patient sensitivity, and active proliferating fraction
        kill_efficiency = 0.42 * p.chemo_sensitivity * dose_fraction
        
        # Gompertz natural regrowth between cycles
        if self.current_diameter_mm < self.carrying_capacity_mm:
            regrowth = self.growth_rate_lambda * self.current_diameter_mm * np.log(
                max(1.01, self.carrying_capacity_mm / max(1.0, self.current_diameter_mm))
            )
        else:
            regrowth = 0.0

        # Net diameter update
        reduction = self.current_diameter_mm * kill_efficiency
        new_diameter = max(0.0, self.current_diameter_mm - reduction + (0.25 * regrowth))
        self.current_diameter_mm = float(new_diameter)

        # 2. Cumulative Toxicity Accumulation & Recovery
        # Toxicity increases with dose and patient vulnerability, minus hepatic/renal clearance
        added_tox = dose_fraction * 1.35 * p.toxicity_susceptibility * (1.0 + 0.15 * p.ecog_ps)
        cleared_tox = self.current_toxicity * self.toxicity_clearance_delta * (p.drug_clearance_rate / 0.28)
        self.current_toxicity = max(0.0, self.current_toxicity + added_tox - cleared_tox)

        # 3. Clinical Response & Termination Checks
        rel_change_pct = ((self.current_diameter_mm - baseline_diam) / baseline_diam) * 100.0
        terminated = False
        truncated = self.current_cycle >= self.max_cycles

        # Fatal / toxic arrest condition
        is_fatal_toxicity = self.current_toxicity >= self.lethal_toxicity_threshold
        is_disease_progression_arrest = self.current_diameter_mm >= (baseline_diam * 1.5)

        if is_fatal_toxicity or is_disease_progression_arrest:
            terminated = True

        # 4. Multi-Objective Clinical Oncology Reward Function
        # Terms:
        # + Tumor diameter shrinkage relative to baseline
        # - Continuous quadratic toxicity penalty
        # - Severe CTCAE Grade 3+ penalty
        # - Fatal termination penalty
        tumor_shrinkage_reward = ((prev_diameter - self.current_diameter_mm) / baseline_diam) * 8.0
        continuous_tox_penalty = 1.2 * ((self.current_toxicity / 2.0) ** 2)
        
        grade3_penalty = 4.0 if self.current_toxicity >= 2.5 else 0.0
        fatal_penalty = 20.0 if terminated and is_fatal_toxicity else 0.0

        # RECIST Partial/Complete Response bonus at cycle completion
        recist_bonus = 0.0
        if rel_change_pct <= -30.0:
            recist_bonus = 3.0
        if self.current_diameter_mm <= 5.0: # Complete disappearance
            recist_bonus = 8.0

        reward = tumor_shrinkage_reward - continuous_tox_penalty - grade3_penalty - fatal_penalty + (recist_bonus if truncated else 0.0)

        # Record history
        step_record = {
            "cycle": self.current_cycle,
            "dose": dose_fraction,
            "diameter_mm": self.current_diameter_mm,
            "toxicity": self.current_toxicity,
            "relative_change_pct": rel_change_pct,
            "reward": reward
        }
        self.history.append(step_record)

        info = {
            "patient_id": p.patient_id,
            "cycle": self.current_cycle,
            "diameter_mm": self.current_diameter_mm,
            "toxicity": self.current_toxicity,
            "relative_change_pct": rel_change_pct,
            "is_fatal_toxicity": is_fatal_toxicity,
            "history": self.history
        }

        return self._get_observation(), float(reward), terminated, truncated, info
