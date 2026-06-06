"""
LLM prompt templates for zero-shot IDH mutation prediction.
Mirrors the prompt structure described in the paper.
"""

from typing import List, Optional

SYSTEM_PROMPT = """\
You are an expert neuroradiologist and molecular neuropathologist with deep \
knowledge of glioma biology. You will be given a structured MRI-derived feature \
report for a patient with a confirmed brain glioma. Your task is to predict the \
IDH (isocitrate dehydrogenase) mutation status of the tumour using only the \
imaging-derived and clinical features provided — no additional information is \
available.

Key imaging correlates of IDH mutation status:
- IDH-MUTANT gliomas: typically frontal/anterior location, well-defined margins, \
  lower grade at presentation, less enhancement, higher T2/FLAIR signal relative \
  to T1-CE, lower enhancement ratios, larger oedema-to-tumour volume ratios.
- IDH-WILDTYPE (GBM): often parietal/temporal/multifocal, irregular margins, \
  high enhancement (T1-CE), ring-enhancing pattern, necrotic core, higher \
  enhancement ratios, older patient age.

Reason step by step before giving your final answer.
When citing features, use ONLY identifiers from the AVAILABLE FEATURES list provided."""

USER_PROMPT_TEMPLATE = """\
Below is the MRI feature report for this patient. Based on these features, \
predict whether the tumour is IDH-mutant or IDH-wildtype.

{narrative}

---
AVAILABLE FEATURES (use ONLY these identifiers when citing features):
{feature_vocabulary}

---
Respond ONLY in the following JSON format (no extra text):
{{
  "idh_status": "IDH-mutant" | "IDH-wildtype",
  "confidence": "high" | "medium" | "low",
  "reasoning": "<2-4 sentence explanation citing specific imaging features>",
  "decisive_feature": "<single most decisive feature identifier from AVAILABLE FEATURES>",
  "supporting_features": ["<feature_id>", ...],
  "contradicting_features": ["<feature_id>", ...],
  "findings": "<draft Findings paragraph for a radiology report>",
  "impression": "<one-sentence draft Impression for a radiology report>"
}}"""

COUNTERFACTUAL_PROMPT_TEMPLATE = """\
You previously predicted {original_prediction} for this case. What single minimal \
change to the features below would most likely flip your prediction to {opposite_prediction}?

{narrative}

---
AVAILABLE FEATURES:
{feature_vocabulary}

---
Respond ONLY in the following JSON format (no extra text):
{{
  "feature_to_change": "<feature identifier from AVAILABLE FEATURES>",
  "current_value": <current numeric value>,
  "new_value": <new numeric value that would flip the prediction>,
  "reasoning": "<why this change would flip the prediction>"
}}"""

OPPOSITE_RATIONALE_PROMPT_TEMPLATE = """\
Based on the MRI features below, write a radiology assessment arguing for \
{target_prediction}. Make the strongest possible case for {target_prediction} \
using only the features provided.

{narrative}

---
Respond ONLY in the following JSON format (no extra text):
{{
  "findings": "<Findings paragraph arguing for {target_prediction}>",
  "impression": "<one-sentence Impression arguing for {target_prediction}>"
}}"""

LABEL_FROM_RATIONALE_PROMPT_TEMPLATE = """\
Based on the following radiology report assessment, what is the IDH mutation status?

FINDINGS: {findings}

IMPRESSION: {impression}

Respond ONLY in the following JSON format (no extra text):
{{
  "idh_status": "IDH-mutant" | "IDH-wildtype",
  "confidence": "high" | "medium" | "low"
}}"""

_VOCAB_PLACEHOLDER = "(feature list not provided)"


def build_prompt(
    narrative: str,
    feature_vocabulary: Optional[List[str]] = None,
) -> tuple:
    """Return (system_prompt, user_prompt) for the OpenAI messages API."""
    vocab_str = (
        "\n".join(f"  - {f}" for f in feature_vocabulary)
        if feature_vocabulary
        else _VOCAB_PLACEHOLDER
    )
    return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(
        narrative=narrative,
        feature_vocabulary=vocab_str,
    )
