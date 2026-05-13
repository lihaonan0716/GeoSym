"""
Baseline extraction and formatting modules for the structured fusion experiment.

OCR + Rules: use generic OCR text extraction plus rule-based coordinate parsing.
Plain Table: reuse GeoSym-extracted coordinates in a plain, unstructured format.
"""

from evaluator.baselines.ocr_rule_baseline import OCRRuleBaseline
from evaluator.baselines.plain_table_baseline import PlainTableBaseline

__all__ = ["OCRRuleBaseline", "PlainTableBaseline"]
