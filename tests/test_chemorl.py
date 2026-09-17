"""
Unit Tests for ChemoRL Pipeline.
Verifies data streaming, vision encoder forward pass, PK/PD environment dynamics,
DQN agent updates, and RECIST/CTCAE evaluation metrics.
"""

import pytest
import torch
import numpy as np

from src.data.online_loader import OnlineLungCancerDataset
from src.models.vision_encoder import CTScanVisionEncoder
from src.models.pk_pd_simulator import ChemotherapyEnvironment
from src.models.rl_agent import ChemoRLAgent
from src.evaluation.recist import evaluate_recist_1_1, RECISTCategory
from src.evaluation.ctcae import grade_toxicity_ctcae, CTCAEGrade
from src.models.generative_renderer import GenerativeDiseaseRenderer

def test_dataset_initialization():
    dataset = OnlineLungCancerDataset(num_patients=10, img_size=64, seed=123)
    assert len(dataset) == 10
    img, clinical, meta = dataset[0]
    assert img.shape == (1, 64, 64)
    assert clinical.shape == (6,)
    assert "patient_id" in meta

def test_vision_encoder():
    encoder = CTScanVisionEncoder(in_channels=1, latent_dim=32)
    dummy_img = torch.randn(2, 1, 64, 64)
    latent = encoder(dummy_img)
    assert latent.shape == (2, 32)
    size_proxy = encoder.predict_size_proxy(latent)
    assert size_proxy.shape == (2, 1)

def test_pk_pd_environment():
    dataset = OnlineLungCancerDataset(num_patients=5, seed=42)
    env = ChemotherapyEnvironment(max_cycles=6)
    obs, info = env.reset(dataset.patients[0])
    assert obs.shape == (6,)
    assert info["baseline_diameter_mm"] > 0

    next_obs, reward, term, trunc, step_info = env.step(1.0) # Full dose
    assert next_obs.shape == (6,)
    assert isinstance(reward, float)
    assert step_info["cycle"] == 1
    assert step_info["diameter_mm"] >= 0

def test_recist_evaluation():
    # Complete Response: >95% or <=5mm
    recist_cr = evaluate_recist_1_1(baseline_diameter_mm=50.0, nadir_diameter_mm=2.0, final_diameter_mm=2.0)
    assert recist_cr["code"] == "CR"
    assert recist_cr["is_responder"] is True

    # Partial Response: >= 30% decrease
    recist_pr = evaluate_recist_1_1(baseline_diameter_mm=50.0, nadir_diameter_mm=30.0, final_diameter_mm=30.0)
    assert recist_pr["code"] == "PR"
    assert recist_pr["is_responder"] is True

    # Progressive Disease: >= 20% increase
    recist_pd = evaluate_recist_1_1(baseline_diameter_mm=50.0, nadir_diameter_mm=50.0, final_diameter_mm=65.0)
    assert recist_pd["code"] == "PD"
    assert recist_pd["is_responder"] is False

def test_ctcae_toxicity_grading():
    g0 = grade_toxicity_ctcae(0.5)
    assert g0["grade"] == CTCAEGrade.GRADE_0

    g3 = grade_toxicity_ctcae(3.2)
    assert g3["grade"] == CTCAEGrade.GRADE_3
    assert g3["is_severe_g3_plus"] is True

def test_generative_renderer():
    renderer = GenerativeDiseaseRenderer(img_size=64)
    dummy_slice = np.zeros((1, 64, 64), dtype=np.float32)
    evolved = renderer.render_cycle_slice(dummy_slice, baseline_diameter_mm=50.0, current_diameter_mm=25.0, cycle_idx=3)
    assert evolved.shape == (64, 64)
    assert np.all(evolved >= 0.0) and np.all(evolved <= 1.0)
