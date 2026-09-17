"""
RECIST 1.1 (Response Evaluation Criteria in Solid Tumors) Standards.
Evaluates tumor response into Complete Response (CR), Partial Response (PR),
Stable Disease (SD), or Progressive Disease (PD).
"""

from enum import Enum
from typing import Dict, Any

class RECISTCategory(str, Enum):
    CR = "Complete Response"
    PR = "Partial Response"
    SD = "Stable Disease"
    PD = "Progressive Disease"


def evaluate_recist_1_1(baseline_diameter_mm: float, nadir_diameter_mm: float, final_diameter_mm: float) -> Dict[str, Any]:
    """
    Evaluates RECIST 1.1 category.
    
    Args:
        baseline_diameter_mm: Baseline sum of longest diameters
        nadir_diameter_mm: Smallest sum of diameters recorded on study
        final_diameter_mm: End of study sum of diameters
        
    Returns:
        Dict with RECIST category, percentage change from baseline, and code.
    """
    pct_from_baseline = ((final_diameter_mm - baseline_diameter_mm) / baseline_diameter_mm) * 100.0
    pct_from_nadir = ((final_diameter_mm - nadir_diameter_mm) / max(1e-3, nadir_diameter_mm)) * 100.0

    if final_diameter_mm <= 5.0 or pct_from_baseline <= -95.0:
        category = RECISTCategory.CR
        code = "CR"
    elif pct_from_baseline <= -30.0:
        category = RECISTCategory.PR
        code = "PR"
    elif pct_from_nadir >= 20.0 and (final_diameter_mm - nadir_diameter_mm) >= 5.0:
        category = RECISTCategory.PD
        code = "PD"
    else:
        category = RECISTCategory.SD
        code = "SD"

    return {
        "category": category.value,
        "code": code,
        "pct_change_from_baseline": pct_from_baseline,
        "is_responder": code in ["CR", "PR"],
        "is_controlled": code in ["CR", "PR", "SD"]
    }
