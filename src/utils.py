"""Utilities: plotting, bookkeeping, schedules, simple helpers."""
from __future__ import annotations
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# global plotting style
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 14
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
