"""
CTCAE v5.0 (Common Terminology Criteria for Adverse Events) Grading Module.
Translates continuous organ toxicity burden into standardized clinical grades (0 to 5).
"""

from enum import IntEnum
from typing import Dict, Any

class CTCAEGrade(IntEnum):
    GRADE_0 = 0  # No Adverse Event
    GRADE_1 = 1  # Mild / Asymptomatic
    GRADE_2 = 2  # Moderate / Local intervention indicated
    GRADE_3 = 3  # Severe / Medically significant / Dose reduction indicated
    GRADE_4 = 4  # Life-threatening consequences / Urgent cessation
    GRADE_5 = 5  # Death


def grade_toxicity_ctcae(toxicity_score: float) -> Dict[str, Any]:
    """
    Grades toxicity score according to CTCAE criteria.
    """
    if toxicity_score < 1.0:
        grade = CTCAEGrade.GRADE_0
        desc = "Grade 0: Normal / Tolerable"
    elif toxicity_score < 1.85:
        grade = CTCAEGrade.GRADE_1
        desc = "Grade 1: Mild (Nausea / Fatigue)"
    elif toxicity_score < 2.75:
        grade = CTCAEGrade.GRADE_2
        desc = "Grade 2: Moderate (Neutropenia / Mucositis)"
    elif toxicity_score < 3.80:
        grade = CTCAEGrade.GRADE_3
        desc = "Grade 3: Severe (Thrombocytopenia / Dose Reduction Required)"
    else:
        grade = CTCAEGrade.GRADE_4
        desc = "Grade 4: Life-Threatening Organ Failure"

    return {
        "grade": int(grade),
        "description": desc,
        "is_severe_g3_plus": int(grade) >= 3,
        "is_life_threatening": int(grade) >= 4
    }
