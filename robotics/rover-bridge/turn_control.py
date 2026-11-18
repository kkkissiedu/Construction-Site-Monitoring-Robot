"""
turn_control.py — closed-loop turn controller: gyro feedback with saturation, spin and arc modes.

Pure Python (no ROS, no hardware), so it can be simulated and tested on a laptop.
"""

# %% Imports
import math


# %% Config
FULL_SCALE      = 255      # motor command range, the PWM resolution the loop saturates at
TURN_THRESHOLD  = 0.05     # rad/s, a requested turn rate below this counts as "not turning"
IDLE_RESET      = 0.5      # s without a turn request before an arc latch is released
ALPHA_FILTER    = 0.2      # smoothing of the accelerometer's angular-acceleration estimate
ACC_DAMPING     = 0.1      # s, scales the accelerometer term relative to kp


# %% Helpers
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fit(left: float, right: float) -> tuple:
    """Bring both commands into range by shifting them together, so the turn (their difference) is kept."""
    top = max(left, right)
    if top > FULL_SCALE:
        left, right = left - (top - FULL_SCALE), right - (top - FULL_SCALE)

    bottom = min(left, right)
    if bottom < -FULL_SCALE:
        left, right = left + (-FULL_SCALE - bottom), right + (-FULL_SCALE - bottom)

    return clamp(left, -FULL_SCALE, FULL_SCALE), clamp(right, -FULL_SCALE, FULL_SCALE)


# %% Controller
class TurnController:
    """Turn-rate loop on the gyro.

    command = kinematic feed-forward from (linear, angular)
              + kp * e + ki * integral(e)                      e = angular - gyro_z   (rad/s)
              - acc_weight * kp * ACC_DAMPING * alpha          alpha = accel_y / imu_offset

    The left side gets the feed-forward minus the correction and the right side plus it, so a left
    turn (positive) speeds the right side up. Every output is limited to +-FULL_SCALE, and the
    integrator stops growing while the output is saturated, so no wheel-speed estimate is needed.

    Modes: "open" (feed-forward only), "spin" (wheels counter-rotate), "arc" (the inner wheel never
    reverses) and "auto": spin first, and if the output sits at full scale for stall_time seconds
    without the gyro reaching stall_ratio of the requested rate, fall back to an arc until the turn
    request ends.
    """

    def __init__(self, track: float, max_speed: float, mode: str = "auto", kp: float = 50.0, ki: float = 120.0,
                 acc_weight: float = 0.0, imu_offset: float = 0.07, arc_ratio: float = 1.5,
                 stall_time: float = 1.0, stall_ratio: float = 0.5) -> None:
        self.track       = track
        self.max_speed   = max_speed
        self.mode        = mode
        self.kp          = kp
        self.ki          = ki
        self.acc_weight  = acc_weight
        self.imu_offset  = imu_offset
        self.arc_ratio   = arc_ratio
        self.stall_time  = stall_time
        self.stall_ratio = stall_ratio
        self.reset()

    def reset(self) -> None:
        self.integral    = 0.0
        self.stalled_for = 0.0
        self.idle_for    = 0.0
        self.alpha       = 0.0
        self.arc_latched = False

    def update(self, dt: float, linear: float, angular: float, gyro_z=None, accel_y=None) -> tuple:
        """Return (left, right, info). linear in m/s, angular and gyro_z in rad/s (left positive), accel_y in m/s^2."""
        info    = {"mode": "open", "saturated": False, "error": 0.0, "switched": False}
        turning = abs(angular) > TURN_THRESHOLD

        if not turning:
            self.idle_for += dt
            if self.idle_for >= IDLE_RESET:
                self.arc_latched = False
        else:
            self.idle_for = 0.0

        if not turning and abs(linear) < 1e-3:
            self.integral    = 0.0
            self.stalled_for = 0.0
            self.alpha       = 0.0
            return 0, 0, info

        # geometry: in an arc the inner wheel is kept from reversing
        use_arc = turning and (self.mode == "arc" or (self.mode == "auto" and self.arc_latched))
        v       = linear
        if use_arc:
            floor = self.arc_ratio * abs(angular) * self.track / 2.0
            if abs(v) < floor:
                v = math.copysign(floor, linear if abs(linear) > 1e-3 else 1.0)

        scale     = FULL_SCALE / self.max_speed
        left_ff   = (v - angular * self.track / 2.0) * scale
        right_ff  = (v + angular * self.track / 2.0) * scale

        # feedback
        closed = gyro_z is not None and self.mode != "open"
        error  = 0.0
        corr   = 0.0
        if closed:
            error = angular - gyro_z
            if accel_y is not None and self.acc_weight > 0.0:
                self.alpha += ALPHA_FILTER * (accel_y / self.imu_offset - self.alpha)
            else:
                self.alpha = 0.0
            corr = self.kp * error + self.ki * self.integral - self.acc_weight * self.kp * ACC_DAMPING * self.alpha
            info["mode"]  = "arc" if use_arc else ("spin" if turning else "hold")
            info["error"] = error

        left_raw  = left_ff - corr
        right_raw = right_ff + corr
        peak      = max(abs(left_raw), abs(right_raw))
        saturated = peak >= FULL_SCALE

        if closed and ((not saturated) or error * corr < 0.0):          # anti-windup
            limit = FULL_SCALE / max(self.ki, 1e-6)
            self.integral = clamp(self.integral + error * dt, -limit, limit)

        if closed and self.mode == "auto" and turning and not use_arc:  # spin that cannot get going
            progress = gyro_z * angular > 0.0 and abs(gyro_z) >= self.stall_ratio * abs(angular)
            if peak >= 0.98 * FULL_SCALE and not progress:
                self.stalled_for += dt
                if self.stalled_for >= self.stall_time:
                    self.arc_latched = True
                    self.integral    = 0.0
                    self.stalled_for = 0.0
                    info["switched"] = True
            else:
                self.stalled_for = 0.0

        if use_arc:                    # clip the outer wheel and keep the inner one rolling the same way as the body
            left  = clamp(left_raw,  -FULL_SCALE, FULL_SCALE)
            right = clamp(right_raw, -FULL_SCALE, FULL_SCALE)
            sign  = 1.0 if v >= 0.0 else -1.0
            if angular > 0.0:
                left = max(left * sign, 0.0) * sign
            else:
                right = max(right * sign, 0.0) * sign
        else:
            left, right = fit(left_raw, right_raw)

        info["saturated"] = saturated
        return round(left), round(right), info
