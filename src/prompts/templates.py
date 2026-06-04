"""
LLM prompt templates for zero-shot IDH mutation prediction.
Mirrors the prompt structure described in the paper.
"""

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

Reason step by step before giving your final answer."""

USER_PROMPT_TEMPLATE = """\
Below is the MRI feature report for this patient. Based on these features, \
predict whether the tumour is IDH-mutant or IDH-wildtype.

{narrative}

---
Respond ONLY in the following JSON format (no extra text):
{{
  "idh_status": "IDH-mutant" | "IDH-wildtype",
  "confidence": "high" | "medium" | "low",
  "reasoning": "<2-4 sentence explanation citing specific imaging features>"
}}"""


def build_prompt(narrative: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the OpenAI messages API."""
    return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(narrative=narrative)
