def calc_rot_time(start_angle, target_angle, sec_per_rad=1.2, min_time=0.8, max_time=None):
    duration = max(min_time, abs(target_angle - start_angle) * sec_per_rad)
    if max_time is not None:
        return min(duration, max_time)
    return duration
