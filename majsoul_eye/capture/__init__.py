"""Capture tooling: record Mahjong Soul ground-truth from Akagi's MITM stream.

⚠️ DEV-ONLY / Akagi-COUPLED. Everything under ``capture/`` is *training
infrastructure* — it imports and monkeypatches Akagi (AGPLv3 + Commons Clause).
The shipped recognizer (``majsoul_eye`` core, the vision models) must NEVER
import this package, so the deployed product stays free of Akagi code. See
``docs/DESIGN.md`` §7 (license).
"""
