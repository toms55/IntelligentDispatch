"""Project-wide constants."""
from datetime import timedelta

nem_interval = timedelta(minutes=5)  # NEM dispatch/settlement interval
nem_interval_hours = nem_interval / timedelta(hours=1)  # 1/12, for scaling per-hour rates
