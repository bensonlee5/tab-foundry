from __future__ import annotations

from hypothesis import settings


HYPOTHESIS_CI = settings(deadline=None, max_examples=25)
HYPOTHESIS_EXTENDED = settings(deadline=None, max_examples=35)
HYPOTHESIS_STRESS = settings(deadline=None, max_examples=40)
HYPOTHESIS_TINY = settings(deadline=None, max_examples=10)
