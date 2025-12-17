# env/uav_env.py
import gym
from gym import spaces
import numpy as np
import math

class UAVEnv(gym.Env):
    """
    UAV environment for obstacle avoidance and target tracking.
    Observation: sequences of 20-d vectors [self-state(5), target-state(4), relative(2), radar(9)] -> total 20
    Action: 2-dim continuous [-1,1] (linear_accel_scale, angular_accel_scale)
    - Generates obstacles randomly (excluding a forbidden spawn box for the pursuer).
    - Provides set_num_obstacles() to regenerate obstacles (useful for curriculum learning).
    - Reward is continuous and composed of potential-based shaping + smooth zone reward + smooth obstacle penalty + other terms.
    """

    def __init__(self,
                 observation_seq_len=1,
                 p_drop=0.0,
                 env_size=(150.0, 150.0),
                 num_obstacles=10,
                 obstacle_min_radius=1.0,
                 obstacle_max_radius=3.0,
                 max_speed=2.0,
                 max_angular_speed=math.pi,
                 max_linear_accel=0.5,
                 max_angular_accel=math.pi/6,
                 dt=0.1,
                 target_speed=1.0,
                 seed=None,
                 test_mode=False,
                 shaping_gamma=1.0,
                 auto_generate_obstacles=True,
                 regen_obstacles_per_episode=False,
                 pursuer_spawn_box=None,
                 min_agent_spawn_separation_frac=0.12,
                 ):
        super(UAVEnv, self).__init__()

        # perception / environment params
        self.r_s = 12.0  # radar max perception radius (used for normalization)
        self.N = observation_seq_len
        self._base_p_drop = float(p_drop)

        # environment geometry
        self.width, self.height = float(env_size[0]), float(env_size[1])
        self.D_max = math.hypot(self.width, self.height)

        # obstacle generation parameters
        self.num_obstacles = int(num_obstacles)
        self.obstacle_min_radius = float(obstacle_min_radius)
        self.obstacle_max_radius = float(obstacle_max_radius)

        # motion limits
        self.max_speed = float(max_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.max_linear_accel = float(max_linear_accel)
        self.max_angular_accel = float(max_angular_accel)
        self.dt = float(dt)

        # target params
        self.target_speed = float(target_speed)

        # shaping gamma for potential-based shaping (match RL gamma)
        self.shaping_gamma = float(shaping_gamma)

        # store obstacles list as [(ox, oy, r), ...]
        self.obstacles = []

        # state vector: [x, y, yaw, v, w, tx, ty, tvx, tvy]
        self.state = np.zeros(9, dtype=np.float32)

        # last target info (for dropout)
        self.last_target_info = np.zeros(4, dtype=np.float32)
        self.last_rel = np.zeros(2, dtype=np.float32)

        # observation history (N x 20)
        self.history = []

        # RNGs
        self.test_mode = bool(test_mode)
        self._seed = seed
        self.rng = np.random.RandomState(seed)
        # test_mode should provide seed for reproducibility
        if self.test_mode and seed is None:
            raise ValueError("UAVEnv: test_mode=True requires explicit seed for reproducibility.")

        # action / observation spaces
        self.action_space = spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.ones((self.N, 20)),
                                            high=np.ones((self.N, 20)),
                                            dtype=np.float32)

        # default pursuer spawn box (absolute coordinates)
        # [x0, y0, x1, y1]
        if pursuer_spawn_box is None:
            self.pursuer_spawn_box = [15.0, 15.0, 20.0, 20.0]
        else:
            # sanitize provided value -> convert to list of floats and clamp to env bounds later at reset()
            self.pursuer_spawn_box = [
                float(pursuer_spawn_box[0]), float(pursuer_spawn_box[1]),
                float(pursuer_spawn_box[2]), float(pursuer_spawn_box[3])
            ]

        # internal flag: if obstacles empty, we'll auto-generate on reset
        self._auto_generate_obstacles = bool(auto_generate_obstacles)
        self._regen_obstacles_per_episode = bool(regen_obstacles_per_episode)

        self.min_agent_spawn_separation_frac = float(min_agent_spawn_separation_frac)

        # initialize obstacles now (if desired)
        self.set_num_obstacles(self.num_obstacles, forbidden_box=self.pursuer_spawn_box)

        # initialize environment (populates history)
        self.reset()

    # ----------------------------
    # Public API for obstacle control
    # ----------------------------
    def set_num_obstacles(self, n, forbidden_box=None, min_radius=None, max_radius=None, max_attempts=2000):
        """
        Regenerate obstacle list with n obstacles, avoiding forbidden_box if provided.
        forbidden_box: [x0, y0, x1, y1] (absolute coords) to avoid placing obstacle centers inside.
        """
        self.num_obstacles = int(n)
        if min_radius is not None:
            self.obstacle_min_radius = float(min_radius)
        if max_radius is not None:
            self.obstacle_max_radius = float(max_radius)

        self.obstacles = self._generate_random_obstacles(
            n=self.num_obstacles,
            forbidden_box=forbidden_box,
            min_r=self.obstacle_min_radius,
            max_r=self.obstacle_max_radius,
            max_attempts=max_attempts
        )
        return self.obstacles

    # ----------------------------
    # Internal helpers: sampling & validation
    # ----------------------------
    def _point_in_box(self, x, y, box):
        x0, y0, x1, y1 = box
        return (x >= x0) and (x <= x1) and (y >= y0) and (y <= y1)

    def _valid_obstacle_center(self, cx, cy, radius, obstacles, forbidden_box, margin=0.5):
        # check bounds
        if not (radius <= cx <= (self.width - radius) and radius <= cy <= (self.height - radius)):
            return False
        # check forbidden_box
        if forbidden_box is not None and self._point_in_box(cx, cy, forbidden_box):
            return False
        # check overlap with other obstacles
        for (ox, oy, orad) in obstacles:
            if math.hypot(cx - ox, cy - oy) <= (radius + orad + margin):
                return False
        return True

    def _generate_random_obstacles(self, n, forbidden_box=None, min_r=3.0, max_r=8.0, max_attempts=2000):
        """
        Generate up to n circular obstacles; avoids forbidden_box for centers.
        Tries max_attempts per obstacle; relaxes constraints if can't place.
        Returns list of (ox, oy, oradius).
        """
        obstacles = []
        attempts = 0
        placed = 0
        while placed < n and attempts < max_attempts:
            attempts += 1
            radius = float(self.rng.uniform(min_r, max_r))
            cx = float(self.rng.uniform(radius, self.width - radius))
            cy = float(self.rng.uniform(radius, self.height - radius))
            if self._valid_obstacle_center(cx, cy, radius, obstacles, forbidden_box, margin=0.5):
                obstacles.append((cx, cy, radius))
                placed += 1
            # if attempts high and cannot place, relax min_r slightly
            if attempts % 500 == 0 and placed < n:
                min_r = max(1.0, min_r * 0.9)
                max_r = max(min_r + 0.5, max_r * 0.95)
        # If unable to place enough, return whatever placed (caller can handle)
        return obstacles

    def _valid_spawn_point(self, x, y, obstacles, min_agent_sep, agent_margin=1.0):
        # bounds
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            return False
        # obstacles check: ensure distance > obstacle_radius + agent_margin
        for (ox, oy, r) in obstacles:
            if math.hypot(x - ox, y - oy) <= (r + agent_margin):
                return False
        return True

    def _sample_in_box(self, box):
        x0, y0, x1, y1 = box
        x = float(self.rng.uniform(x0, x1))
        y = float(self.rng.uniform(y0, y1))
        return x, y

    def _sample_spawn_positions(self, pursuer_box=None, min_agent_sep_frac=0.12, max_attempts=2000):
        """
        Sample pursuer and evader positions.
        - pursuer_box: [x0,y0,x1,y1] absolute coordinates; if None, default small area used.
        - evader sampled anywhere in full region but must not overlap obstacles and must be separated from pursuer.
        Returns: (px,py), (ex,ey)
        """
        if pursuer_box is None:
            pursuer_box = self.pursuer_spawn_box

        # min separation based on diagonal
        D = math.hypot(self.width, self.height)
        min_agent_sep = float(min_agent_sep_frac) * D

        # sample pursuer within box (guaranteed)
        attempts = 0
        p_pos = None
        while attempts < max_attempts:
            attempts += 1
            px, py = self._sample_in_box(pursuer_box)
            # Also ensure pursuer not inside obstacle
            if self._valid_spawn_point(px, py, self.obstacles, min_agent_sep=0.0, agent_margin=0.01):
                p_pos = (px, py)
                break
        if p_pos is None:
            # fallback: pick center of pursuer_box
            px = 0.5 * (pursuer_box[0] + pursuer_box[2])
            py = 0.5 * (pursuer_box[1] + pursuer_box[3])
            p_pos = (px, py)

        # sample evader in full area ensuring not overlapping obstacles and separated from pursuer
        attempts = 0
        e_pos = None
        while attempts < max_attempts:
            attempts += 1
            ex = float(self.rng.uniform(0.0, self.width))
            ey = float(self.rng.uniform(0.0, self.height))
            if not self._valid_spawn_point(ex, ey, self.obstacles, min_agent_sep=0.0, agent_margin=0.01):
                continue
            # check separation from pursuer
            if math.hypot(ex - p_pos[0], ey - p_pos[1]) < min_agent_sep:
                continue
            e_pos = (ex, ey)
            break
        if e_pos is None:
            # fallback: place evader near far edge
            ex = min(self.width - 1.0, max(1.0, p_pos[0] + D * 0.5))
            ey = min(self.height - 1.0, max(1.0, p_pos[1]))
            e_pos = (ex, ey)

        return p_pos, e_pos

    # ----------------------------
    # Gym API
    # ----------------------------
    def seed(self, seed=None):
        """Sets the random seed for reproducibility (only affects this env instance)."""
        self._seed = seed
        import random as _py_random
        _py_random.seed(seed)  # optional: python random seed (keeps env deterministic for python random)
        self.rng = np.random.RandomState(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        """
        Reset env:
        - If obstacles list empty or auto-generate allowed, generate obstacles (excluding pursuer spawn box).
        - Sample pursuer in [15,20]x[15,20], sample evader anywhere (not overlapping obstacles).
        - Initialize velocities to zero (paper setup).
        - Build initial observation and history (repeated N times).
        """
        if seed is not None:
            self.seed(seed)

        # Ensure pursuer spawn box boundaries are valid relative to env size
        px0, py0, px1, py1 = self.pursuer_spawn_box
        # clamp spawn box to env bounds (safety)
        self.pursuer_spawn_box = [
            max(0.0, min(px0, self.width)), max(0.0, min(py0, self.height)),
            max(0.0, min(px1, self.width)), max(0.0, min(py1, self.height))
        ]

        # If obstacles absent or auto-generation enabled, (maybe) regenerate obstacles.
        # New behavior:
        #  - If regen_obstacles_per_episode == True -> always regenerate (with same n)
        #  - Else if obstacles empty and auto_generate_obstacles True -> generate once (old behavior)
        if self._auto_generate_obstacles:
            if self._regen_obstacles_per_episode:
                # each episode regenerates new obstacles (with same n)
                self.obstacles = self._generate_random_obstacles(
                    n=self.num_obstacles,
                    forbidden_box=self.pursuer_spawn_box,
                    min_r=self.obstacle_min_radius,
                    max_r=self.obstacle_max_radius,
                    max_attempts=2000
                )
            else:
                if (not self.obstacles):
                    self.obstacles = self._generate_random_obstacles(
                        n=self.num_obstacles,
                        forbidden_box=self.pursuer_spawn_box,
                        min_r=self.obstacle_min_radius,
                        max_r=self.obstacle_max_radius,
                        max_attempts=2000
                    )

        # Sample spawn positions
        (px, py), (ex, ey) = self._sample_spawn_positions(pursuer_box=self.pursuer_spawn_box,
                                                         min_agent_sep_frac=self.min_agent_spawn_separation_frac,
                                                         max_attempts=2000)

        # Set pursuer state
        self.state[0] = px
        self.state[1] = py
        self.state[2] = float(self.rng.uniform(-math.pi, math.pi))  # yaw
        self.state[3] = 0.0  # linear vel
        self.state[4] = 0.0  # angular vel

        # Set evader state (position and velocity)
        self.state[5] = ex
        self.state[6] = ey
        angle = float(self.rng.uniform(-math.pi, math.pi))
        self.state[7] = self.target_speed * math.cos(angle)
        self.state[8] = self.target_speed * math.sin(angle)


        # Reset history and populate N times with initial obs
        self.history = []
        initial_obs = self._get_observation(noise=(not self.test_mode), init=True)
        for _ in range(self.N):
            self.history.append(initial_obs.copy())

        # initialize last_dist to current physical distance so first r_shape = 0
        x, y, _, _, _, tx, ty, _, _ = self.state
        self.last_dist = math.hypot(tx - x, ty - y)
        # recompute D_max in case env size changed
        self.D_max = math.hypot(self.width, self.height)

        return np.array(self.history, dtype=np.float32)

    def step(self, action):
        """
        Execute one time step:
        - apply action (clipped), update pursuer dynamics
        - move evader according to its simple policy (avoid close obstacles per original paper)
        - compute observation, rewards, done, info
        Returns (obs_seq, reward, done, False, info)
        """
        # ----------------------------
        # 1) Apply action (scale to acceleration)
        # ----------------------------
        a_lin = float(np.clip(action[0], -1.0, 1.0)) * self.max_linear_accel
        a_ang = float(np.clip(action[1], -1.0, 1.0)) * self.max_angular_accel

        # update pursuer velocities and pose
        self.state[3] = float(np.clip(self.state[3] + a_lin * self.dt, 0.0, self.max_speed))
        self.state[4] = float(np.clip(self.state[4] + a_ang * self.dt, -self.max_angular_speed, self.max_angular_speed))
        self.state[2] = self._wrap_angle(self.state[2] + self.state[4] * self.dt)
        self.state[0] += self.state[3] * math.cos(self.state[2]) * self.dt
        self.state[1] += self.state[3] * math.sin(self.state[2]) * self.dt

        # ----------------------------
        # 2) Evader behavior & integrate
        # ----------------------------
        ev_dists = self._get_evader_observation()
        if any(d < self.r_s for d in ev_dists):
            ang = self.rng.uniform(0, 2 * math.pi)
            self.state[7] = self.target_speed * math.cos(ang)
            self.state[8] = self.target_speed * math.sin(ang)
        # integrate evader
        self.state[5] += float(self.state[7] * self.dt)
        self.state[6] += float(self.state[8] * self.dt)
        # handle boundary bounce
        if not (0 <= self.state[5] <= self.width):
            self.state[5] = float(np.clip(self.state[5], 0.0, self.width))
            self.state[7] = -self.state[7]
        if not (0 <= self.state[6] <= self.height):
            self.state[6] = float(np.clip(self.state[6], 0.0, self.height))
            self.state[8] = -self.state[8]

        # ----------------------------
        # 3) Observation sequence update
        # ----------------------------
        obs = self._get_observation(noise=(not self.test_mode), init=False)
        self.history.pop(0)
        self.history.append(obs.copy())
        obs_seq = np.array(self.history, dtype=np.float32)

        # ----------------------------
        # 4) Geometric computations for reward
        # ----------------------------
        x, y, yaw, v, w, tx, ty, tvx, tvy = self.state
        dx = tx - x
        dy = ty - y
        d_curr = math.hypot(dx, dy)                 # current physical distance (d')
        bearing = math.atan2(dy, dx)
        angle_diff = self._angle_diff(bearing, yaw)

        # previous distance (for shaping)
        d_prev = getattr(self, 'last_dist', d_curr)

        # ----------------------------
        # 5) Initialize rewards & flags
        # ----------------------------
        done = False
        info = {}
        # sub-reward placeholders
        rT = 0.0
        r_shape = 0.0
        r_zone = 0.0
        rTheta = 0.0
        rV = 0.0
        rdir = 0.0
        robs = 0.0

        # ----------------------------
        # 6) constants & hyperparams
        # ----------------------------
        # termination
        K_T = 50.0

        # shape weight
        k_shape = 81.05

        # zone radii (paper): Ri=12, Ro=25
        Ri, Ro = 12.0, 25.0

        # zone smooth params
        zone_min, zone_max = 0.18, 0.30  # S_zone target range
        inner_mag_min, inner_mag_max = 0.18, 0.20  # S_inner target range, will be negated
        alpha = 0.9      # sigmoid steepness for Ri/Ro boundaries
        beta = 0.9       # steepness for inner penalty

        # angle/velocity/direction weights
        k_theta = 0.04
        v_t = 0.8
        k_v = 0.02
        d_s = 10.0
        k_dir = 0.04

        # obstacle avoidance (exp kernel)
        k_obs = 0.06
        lambda_obs = 1.05   # decay length for exp kernel
        rclip = 0.2  # clip of obstacle avoidance reward

        eps = 1e-6

        # ----------------------------
        # 7) Terminal conditions
        # ----------------------------
        for (ox, oy, rad) in self.obstacles:
            if math.hypot(x - ox, y - oy) <= rad:
                done = True
                info['collision'] = True
                break
        if not done and not (0 <= x <= self.width and 0 <= y <= self.height):
            done = True
            info['out_of_bounds'] = True

        if done:
            rT = -K_T

        # ----------------------------
        # 8) potential-based shaping (smooth distance-driven)
        #    Phi(s) = - d / D_max
        #    r_shape = k_shape * (gamma * Phi(s') - Phi(s)) = k_shape * (d_prev - gamma*d_curr) / D_max
        # ----------------------------
        if  d_curr > Ro:
            r_shape = k_shape * (d_prev - self.shaping_gamma * d_curr) / (self.D_max + eps)
        # print(f"[DEBUG] D_max is {self.D_max:.4f}, d_prev is {d_prev}, d_curr is {d_curr}, ref is {(d_prev - self.shaping_gamma * d_curr)}, r_shape is {r_shape}")

        # ----------------------------
        # 9) zone reward (smooth) -- uses current distance d_curr only
        #    raw_zone(d)  = sigma(alpha*(d - Ri)) - sigma(alpha*(d - Ro))    ∈ [0, 1]
        #    raw_inner(d) = sigma(beta*(Ri - d))                            ∈ [0, 1]
        #    S_zone ∈ [zone_min, zone_max]
        #    S_inner ∈ [-inner_mag_max, -inner_mag_min]
        # ----------------------------
        # numerically stable sigmoid
        def sigma(x):
            # clamp x if extremely negative/positive is not strictly necessary here but kept simple
            return 1.0 / (1.0 + math.exp(-x))

        # --- raw responses in [0,1] ---
        raw_zone = sigma(alpha * (d_curr - Ri)) - sigma(alpha * (d_curr - Ro))
        # clamp numerical noise
        raw_zone = float(max(0.0, min(1.0, raw_zone)))

        raw_inner = sigma(beta * (Ri - d_curr))
        raw_inner = float(max(0.0, min(1.0, raw_inner)))

        # --- boundaries for activation ---
        raw_zone_boundary = sigma(alpha * (Ro - Ri)) - 0.5
        # clamp boundary into [0,1] for safety (though expected ~0.0..0.5)
        raw_zone_boundary = float(max(0.0, min(1.0, raw_zone_boundary)))
        raw_inner_boundary = 0.5

        # --- compute S_zone ---
        if raw_zone < raw_zone_boundary:
            S_zone = 0.0
        else:
            denom = (1.0 - raw_zone_boundary)
            if denom <= 1e-9:
                z_norm = 1.0
            else:
                z_norm = (raw_zone - raw_zone_boundary) / denom
            z_norm = max(0.0, min(1.0, z_norm))
            S_zone = zone_min + (zone_max - zone_min) * z_norm

        # --- compute S_inner (negative penalty) ---
        if d_curr > Ri:
            S_inner = 0.0
        else:
            # d_curr <= Ri -> map raw_inner in [0.5,1] -> i_norm in [0,1]
            denom_i = (1.0 - raw_inner_boundary)
            if denom_i <= 1e-9:
                i_norm = 1.0
            else:
                i_norm = (raw_inner - raw_inner_boundary) / denom_i
            i_norm = max(0.0, min(1.0, i_norm))
            inner_mag = inner_mag_min + (inner_mag_max - inner_mag_min) * i_norm
            S_inner = - float(inner_mag)

        # final zone component
        r_zone = float(S_zone + S_inner)

        # ----------------------------
        # 10) heading / angle reward (smooth)
        # ----------------------------
        # use cosine so in [-1,1], positive when facing target
        rTheta = k_theta * math.cos(angle_diff)

        # ----------------------------
        # 11) speed reward: penalize too slow
        # ----------------------------
        if v < v_t:
            rV = -k_v * ((v_t - v) / (v_t + eps))
        else:
            rV = 0.0

        # ----------------------------
        # 12) direction reward (from middle ray, index 4 of 9 in obs tail)
        # ----------------------------
        d_rays = obs[-9:]
        # d5 is denormalized (actual meters)
        d5 = float(d_rays[4] * self.r_s)
        rdir = k_dir * math.tanh((d5 - d_s) / (d_s + eps))

        # ----------------------------
        # 13) obstacle avoidance: smooth exp kernel over 9 rays
        # ----------------------------
        sum_term = 0.0
        for di_norm in d_rays:
            di = float(di_norm * self.r_s)
            # exp kernel: when di is small -> big penalty, when di large -> near zero
            sum_term += math.exp(- (di / (lambda_obs + eps)))
            # print(f"[DEBUG] di is {di}, di_norm is {di_norm}, sum_term is {sum_term}")
        ro = - k_obs * sum_term
        # print(f"[DEBUG] ro is {ro}")
        robs = max(-rclip, ro)

        # ----------------------------
        # 14) aggregate
        # ----------------------------
        reward = rT + r_shape + r_zone + rTheta + rV + rdir + robs

        # ----------------------------
        # 15) info fields (backwards-compatible + extended)
        # ----------------------------
        info['rT'] = rT
        info['r_shape'] = r_shape
        info['r_zone'] = r_zone
        info['rTheta'] = rTheta
        info['rV'] = rV
        info['rdir'] = rdir
        info['robs'] = robs
        info['total'] = reward
        info['dist'] = d_curr


        # ----------------------------
        # 16) update last_dist & return tuple
        # ----------------------------
        self.last_dist = d_curr
        return obs_seq, float(reward), bool(done), False, info

    # ----------------------------
    # Observation math
    # ----------------------------
    def _get_observation(self, noise=False, init=False):
        """
        Build the 20-dim observation vector:
        [x_n, y_n, yaw_n, v_n, w_n, tx_n, ty_n, tvx_n, tvy_n, dist_n, angle_n, 9 * ray_n]
        where ray_n are normalized distances in [0,1] for 9 rays in front 180 deg.
        Handles dropout of target info with probability p_drop (uses last_target_info).
        """
        x, y, yaw, v, w, tx, ty, tvx, tvy = self.state
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        angle_diff = self._angle_diff(bearing, yaw)


        current_p_drop = self._base_p_drop

        # dropout
        if (not init) and (self.rng.uniform(0.0, 1.0) < current_p_drop):
            # reuse last observed target info (as in paper)
            tx_o, ty_o, tvx_o, tvy_o = self.last_target_info
            dist_o, angle_o = self.last_rel
            # use last values instead
            tx_n = (2.0 * tx_o / self.width) - 1.0
            ty_n = (2.0 * ty_o / self.height) - 1.0
            tvx_n = tvx_o / (self.target_speed + 1e-6)
            tvy_n = tvy_o / (self.target_speed + 1e-6)
            dist_n = dist_o / math.hypot(self.width, self.height)
            angle_n = angle_o / math.pi
        else:
            tx_n = (2.0 * tx / self.width) - 1.0
            ty_n = (2.0 * ty / self.height) - 1.0
            tvx_n = tvx / (self.target_speed + 1e-6)
            tvy_n = tvy / (self.target_speed + 1e-6)
            dist_n = dist / math.hypot(self.width, self.height)
            angle_n = angle_diff / math.pi
            # update last observed
            self.last_target_info = np.array([tx, ty, tvx, tvy], dtype=np.float32)
            self.last_rel = np.array([dist, angle_diff], dtype=np.float32)

        # normalize own state
        x_n = (2.0 * x / self.width) - 1.0
        y_n = (2.0 * y / self.height) - 1.0
        yaw_n = yaw / math.pi
        v_n = v / (self.max_speed + 1e-6)
        w_n = w / (self.max_angular_speed + 1e-6)

        # cast 9 rays within 180 degrees ahead (as in original)
        ray_angles = np.linspace(-math.pi / 2, math.pi / 2, 11)[1:-1]  # 9 values
        ray_distances = []
        for rel_angle in ray_angles:
            ra = yaw + rel_angle
            min_t = self.r_s
            # boundary intersections
            if abs(math.cos(ra)) > 1e-6:
                t_x = ((self.width - x) / math.cos(ra)) if (math.cos(ra) > 0) else (-x / math.cos(ra))
            else:
                t_x = np.inf
            if abs(math.sin(ra)) > 1e-6:
                t_y = ((self.height - y) / math.sin(ra)) if (math.sin(ra) > 0) else (-y / math.sin(ra))
            else:
                t_y = np.inf
            for t in (t_x, t_y):
                if 0 < t < min_t:
                    min_t = t
            # obstacle intersections
            for (ox, oy, r_obs) in self.obstacles:
                dx_o = x - ox
                dy_o = y - oy
                a = 1.0
                b = 2.0 * (dx_o * math.cos(ra) + dy_o * math.sin(ra))
                c = dx_o * dx_o + dy_o * dy_o - r_obs * r_obs
                disc = b * b - 4.0 * a * c
                if disc >= 0.0:
                    sqrt_d = math.sqrt(disc)
                    t1 = (-b + sqrt_d) / (2.0 * a)
                    t2 = (-b - sqrt_d) / (2.0 * a)
                    for t in (t1, t2):
                        if 0 < t < min_t:
                            min_t = t
            ray_distances.append(min_t)
        ray_distances = np.array(ray_distances, dtype=np.float32)
        ray_n = ray_distances / (self.r_s + 1e-6)

        # compose obs
        obs = np.array([
            x_n, y_n, yaw_n, v_n, w_n,
            tx_n, ty_n, tvx_n, tvy_n,
            dist_n, angle_n
        ] + ray_n.tolist(), dtype=np.float32)

        if noise:
            obs += self.rng.normal(0.0, 0.05, size=obs.shape)
        obs = np.clip(obs, -1.0, 1.0)


        return obs

    def _get_evader_observation(self):
        """
        360 deg 8-ray observation for evader (actual distances, not normalized)
        """
        tx, ty = self.state[5], self.state[6]
        max_dist = self.r_s // 3
        dists = []
        ray_angles = np.linspace(0, 2 * math.pi, 8, endpoint=False)
        for ra in ray_angles:
            min_t = max_dist
            # boundaries
            if math.cos(ra) > 1e-6:
                t_x = (self.width - tx) / math.cos(ra)
            elif math.cos(ra) < -1e-6:
                t_x = -tx / math.cos(ra)
            else:
                t_x = np.inf
            if math.sin(ra) > 1e-6:
                t_y = (self.height - ty) / math.sin(ra)
            elif math.sin(ra) < -1e-6:
                t_y = -ty / math.sin(ra)
            else:
                t_y = np.inf
            for t in (t_x, t_y):
                if 0 < t < min_t:
                    min_t = t
            for (ox, oy, r) in self.obstacles:
                dx_o = tx - ox
                dy_o = ty - oy
                a = math.cos(ra) ** 2 + math.sin(ra) ** 2
                b = 2.0 * (dx_o * math.cos(ra) + dy_o * math.sin(ra))
                c = dx_o * dx_o + dy_o * dy_o - r * r
                disc = b * b - 4.0 * a * c
                if disc >= 0.0:
                    t1 = (-b + math.sqrt(disc)) / (2.0 * a)
                    t2 = (-b - math.sqrt(disc)) / (2.0 * a)
                    for t in (t1, t2):
                        if 0 < t < min_t:
                            min_t = t
            dists.append(min(min_t, self.r_s))
        return dists

    def get_obstacles(self):
        return list(self.obstacles)

    # ----------------------------
    # utility math
    # ----------------------------
    def _angle_diff(self, target_angle, current_angle):
        diff = target_angle - current_angle
        return (diff + math.pi) % (2 * math.pi) - math.pi

    def _wrap_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi