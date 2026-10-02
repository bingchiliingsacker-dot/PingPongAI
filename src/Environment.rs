use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Constants (previously magic numbers / per-instance fields)
// ---------------------------------------------------------------------------
const WIDTH: i32 = 800;
const HEIGHT: i32 = 600;
const PADDLE_W: i32 = 10;
const PADDLE_H: i32 = 100;
const PADDLE_HALF_H: i32 = PADDLE_H / 2;
const BALL_HALF: i32 = 15; // ball is 30x30
const BLUE_X: i32 = 60;
const RED_X: i32 = 730;
const MAX_STEPS: i32 = 500_000;
const OBS_SIZE: usize = 6;
const STATE_STRIDE: usize = 2 * OBS_SIZE; // red obs + blue obs per env

// ---------------------------------------------------------------------------
// Tiny, fast, per-environment RNG (xorshift64*), seeded via splitmix64.
// Replaces creating a thread RNG on every call to rnd()/new()/reset().
// ---------------------------------------------------------------------------
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Self(z | 1) // state must be non-zero
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform integer in [lo, hi] (inclusive).
    #[inline]
    fn range_incl(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (hi - lo + 1) as u64;
        lo + ((self.next_u64() >> 32) % span) as i32
    }

    /// Returns +1 or -1 with equal probability.
    #[inline]
    fn sign(&mut self) -> i32 {
        if self.next_u64() >> 63 == 0 {
            1
        } else {
            -1
        }
    }
}

/// Same AABB test as the old `colliderect`, but on scalars (no Vec<i32> rects).
#[inline(always)]
fn paddle_hit(paddle_x: i32, paddle_cy: i32, ball_cx: i32, ball_cy: i32) -> bool {
    paddle_x < ball_cx + BALL_HALF
        && paddle_x + PADDLE_W > ball_cx - BALL_HALF
        && (ball_cy - paddle_cy).abs() < BALL_HALF + PADDLE_HALF_H
}

// ---------------------------------------------------------------------------
// Single environment (plain Rust struct, not exposed to Python)
// ---------------------------------------------------------------------------
struct Env {
    rng: Rng,
    ball_cx: i32,
    ball_cy: i32,
    red_cy: i32,
    blue_cy: i32,
    dx: i32,
    dy: i32,
    steps: i32,
    truncated: bool,
}

impl Env {
    fn new(seed: u64) -> Self {
        let mut env = Self {
            rng: Rng::new(seed),
            ball_cx: 0,
            ball_cy: 0,
            red_cy: 0,
            blue_cy: 0,
            dx: 1,
            dy: 1,
            steps: 0,
            truncated: false,
        };
        env.reset();
        env
    }

    fn reset(&mut self) {
        self.ball_cx = WIDTH / 2;
        self.ball_cy = self.rng.range_incl(50, HEIGHT - 50);
        self.red_cy = HEIGHT / 2;
        self.blue_cy = HEIGHT / 2;
        self.dx = self.rng.sign();
        self.dy = self.rng.sign();
        self.steps = 0;
        self.truncated = false;
    }

    /// Writes [red_obs(6), blue_obs(6)] into `out` (len 12). No allocation.
    #[inline]
    fn write_obs(&self, out: &mut [f32]) {
        let w = WIDTH as f32;
        let h = HEIGHT as f32;

        // Red sees the normal orientation.
        out[0] = self.ball_cx as f32 / w;
        out[1] = self.ball_cy as f32 / h;
        out[2] = self.red_cy as f32 / h;
        out[3] = self.dx as f32;
        out[4] = self.dy as f32;
        out[5] = (self.ball_cy - self.red_cy) as f32 / h;

        // Blue sees a horizontally mirrored world.
        out[6] = (WIDTH - self.ball_cx) as f32 / w;
        out[7] = self.ball_cy as f32 / h;
        out[8] = self.blue_cy as f32 / h;
        out[9] = -self.dx as f32;
        out[10] = self.dy as f32;
        out[11] = (self.ball_cy - self.blue_cy) as f32 / h;
    }

    /// Returns (reward_red, reward_blue, done). Game logic identical to the
    /// original implementation.
    #[inline]
    fn step(&mut self, action_red: i8, action_blue: i8) -> (f32, f32, bool) {
        let mut r_red = 0.0f32;
        let mut r_blue = 0.0f32;
        let mut done = false;

        let old_dist_red = (self.ball_cy - self.red_cy).abs();
        let old_dist_blue = (self.ball_cy - self.blue_cy).abs();

        // Paddle collisions
        if self.dx == -1 && paddle_hit(BLUE_X, self.blue_cy, self.ball_cx, self.ball_cy) {
            self.dx = 1;
            r_blue += 1.0;
        }
        if self.dx == 1 && paddle_hit(RED_X, self.red_cy, self.ball_cx, self.ball_cy) {
            self.dx = -1;
            r_red += 1.0;
        }

        // Wall collisions
        if self.ball_cy >= HEIGHT - BALL_HALF {
            self.ball_cy = HEIGHT - BALL_HALF;
            self.dy = -1;
        } else if self.ball_cy <= BALL_HALF {
            self.ball_cy = BALL_HALF;
            self.dy = 1;
        }

        // Move ball
        self.ball_cx += self.dx;
        self.ball_cy += self.dy;

        // Move paddles (0 = up, 1 = down, 2 = stay)
        let min_c = PADDLE_HALF_H;
        let max_c = HEIGHT - PADDLE_HALF_H;

        if action_red == 0 && self.red_cy > min_c {
            self.red_cy -= 1;
        } else if action_red == 1 && self.red_cy < max_c {
            self.red_cy += 1;
        }

        if action_blue == 0 && self.blue_cy > min_c {
            self.blue_cy -= 1;
        } else if action_blue == 1 && self.blue_cy < max_c {
            self.blue_cy += 1;
        }

        // Movement shaping (only while ball travels toward that paddle)
        let new_dist_red = (self.ball_cy - self.red_cy).abs();
        let new_dist_blue = (self.ball_cy - self.blue_cy).abs();

        if self.dx == 1 {
            r_red += if new_dist_red < old_dist_red { 0.01 } else { -0.01 };
        }
        if self.dx == -1 {
            r_blue += if new_dist_blue < old_dist_blue { 0.01 } else { -0.01 };
        }

        // Score
        if self.ball_cx >= 720 {
            r_blue += 10.0;
            r_red -= 10.0;
            done = true;
        } else if self.ball_cx <= 80 {
            r_red += 10.0;
            r_blue -= 10.0;
            done = true;
        }

        // Step limit
        self.steps += 1;
        if !done && self.steps >= MAX_STEPS {
            done = true;
            r_red += 1.0;
            r_blue += 1.0;
            self.truncated = true;
        }

        (r_red, r_blue, done)
    }
}

// ---------------------------------------------------------------------------
// Vectorised simulation exposed to Python. All buffers are preallocated and
// results are handed to Python as NumPy arrays (no nested Vec -> list copies).
// ---------------------------------------------------------------------------
#[pyclass]
pub struct Simulation {
    envs: Vec<Env>,
    dones: Vec<bool>,
    truncs: Vec<bool>,
    states: Vec<f32>,  // [n, 2, 6] flattened, persistent
    rewards: Vec<f32>, // [n, 2] flattened, persistent
}

impl Simulation {
    fn states_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        let n = self.envs.len();
        Array3::from_shape_vec((n, 2, OBS_SIZE), self.states.clone())
            .expect("shape matches buffer")
            .into_pyarray(py)
    }
}

#[pymethods]
impl Simulation {
    #[new]
    #[pyo3(signature = (num_envs, seed=None))]
    fn new(num_envs: usize, seed: Option<u64>) -> Self {
        let base_seed = seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x1234_5678)
        });

        let envs: Vec<Env> = (0..num_envs)
            .map(|i| Env::new(base_seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9))))
            .collect();

        let mut states = vec![0.0f32; num_envs * STATE_STRIDE];
        for (env, out) in envs.iter().zip(states.chunks_exact_mut(STATE_STRIDE)) {
            env.write_obs(out);
        }

        Self {
            envs,
            dones: vec![false; num_envs],
            truncs: vec![false; num_envs],
            states,
            rewards: vec![0.0; num_envs * 2],
        }
    }

    /// Returns (states[n,2,6] f32, rewards[n,2] f32, dones[n] bool, truncateds[n] bool)
    fn step_all<'py>(
        &mut self,
        py: Python<'py>,
        actions_red: PyReadonlyArray1<'py, i8>,
        actions_blue: PyReadonlyArray1<'py, i8>,
    ) -> PyResult<(
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<bool>>,
    )> {
        let n = self.envs.len();
        let red = actions_red.as_slice()?;
        let blue = actions_blue.as_slice()?;

        if red.len() != n || blue.len() != n {
            return Err(PyValueError::new_err(format!(
                "expected {n} actions per agent, got red={}, blue={}",
                red.len(),
                blue.len()
            )));
        }

        self.rewards.fill(0.0);

        for i in 0..n {
            if self.dones[i] {
                continue; // finished envs keep their last state, reward 0
            }

            let env = &mut self.envs[i];
            let (r_red, r_blue, done) = env.step(red[i], blue[i]);

            self.rewards[2 * i] = r_red;
            self.rewards[2 * i + 1] = r_blue;
            self.dones[i] = done;
            self.truncs[i] = env.truncated;

            env.write_obs(&mut self.states[i * STATE_STRIDE..(i + 1) * STATE_STRIDE]);
        }

        let rewards = Array2::from_shape_vec((n, 2), self.rewards.clone())
            .expect("shape matches buffer")
            .into_pyarray(py);

        Ok((
            self.states_array(py),
            rewards,
            PyArray1::from_slice(py, &self.dones),
            PyArray1::from_slice(py, &self.truncs),
        ))
    }

    /// Resets every environment and returns states[n,2,6].
    fn reset_all<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        self.dones.fill(false);
        self.truncs.fill(false);

        for (env, out) in self
            .envs
            .iter_mut()
            .zip(self.states.chunks_exact_mut(STATE_STRIDE))
        {
            env.reset();
            env.write_obs(out);
        }

        self.states_array(py)
    }
}
