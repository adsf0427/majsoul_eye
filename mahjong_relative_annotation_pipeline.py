"""Deprecated location. The precise fullwarp annotation pipeline moved to
``majsoul_eye.annotate.pipeline``.

This shim keeps ``import mahjong_relative_annotation_pipeline as P`` working —
including private helpers accessed as ``P._box_fill`` — by aliasing this module
name to the package module object. Remove once every caller imports from the
package directly (build_case_annotations / calibrate get repointed in a later
step; annotate_ai_session already imports from the package).
"""
import sys as _sys

from majsoul_eye.annotate import pipeline as _pipeline

# Make ``import mahjong_relative_annotation_pipeline`` yield the real module, so
# attribute access (public and _underscore) resolves against the moved code.
_sys.modules[__name__] = _pipeline
