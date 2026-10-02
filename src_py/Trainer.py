import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src_py import AI
from ping_pong_env_rs import Simulation


# ---------------- config ----------------
NUM_ENVS = 256                 # more envs mostly adds an idle tail
EPOCHS = 300
MAX_STEPS_PER_EPOCH = 1500     # cut off the long tail of the slowest envs
PUSH_FRACTION = 0.25           # store ~1 in 4 transitions -> buffer spans many epochs
BATCH_SIZE = 256
UPDATES_PER_EPOCH = 2000
GAMMA = 0.999                  # ~300-tick horizon between a mistake and its penalty
LR = 5e-4
TARGET_SYNC = 500
MEMORY_SIZE = 500_000
EPS_START, EPS_END, EPS_DECAY_EPOCHS = 1.0, 0.05, 150
NUM_ACTIONS = 3
STATE_SIZE = 6

EVAL_EVERY = 10
EVAL_ENVS = 200
EVAL_CAP = 3000
NO_RETURN_LENGTH = 320         # ball crosses the field in 320 ticks if nobody hits it

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | Environments: {NUM_ENVS}")

rng = np.random.default_rng()


def fmt(x):
    return "n/a" if x is None else f"{x:.5f}"


# ---------------- replay buffer ----------------
class ReplayBuffer:
    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.states = torch.empty((capacity, state_size), dtype=torch.float32, device=device)
        self.actions = torch.empty(capacity, dtype=torch.long, device=device)
        self.rewards = torch.empty(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.empty((capacity, state_size), dtype=torch.float32, device=device)
        self.terminals = torch.empty(capacity, dtype=torch.float32, device=device)
        self.pos = 0
        self.size = 0

    def push(self, states, actions, rewards, next_states, terminals):
        n = states.shape[0]
        if n == 0:
            return

        idx = (torch.arange(n, device=device) + self.pos) % self.capacity

        self.states[idx] = states
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_states[idx] = next_states
        self.terminals[idx] = terminals

        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, n):
        idx = torch.randint(0, self.size, (n,), device=device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.terminals[idx],
        )

    def __len__(self):
        return self.size


# ---------------- agent ----------------
class Agent:
    def __init__(self):
        self.net = AI.NeuralNetwork().to(device)

        self.target = copy.deepcopy(self.net)
        self.target.eval()

        self.opt = optim.Adam(
            self.net.parameters(),
            lr=LR,
            fused=(device.type == "cuda"),
        )

        self.loss_fn = nn.SmoothL1Loss()
        self.mem = ReplayBuffer(MEMORY_SIZE, STATE_SIZE)
        self.updates = 0

    def act(self, states, eps):
        with torch.inference_mode():
            greedy = self.net(states).argmax(dim=1)
            if eps <= 0.0:
                return greedy

            n = states.shape[0]
            random_actions = torch.randint(0, NUM_ACTIONS, (n,), device=states.device)
            explore = torch.rand(n, device=states.device) < eps
            return torch.where(explore, random_actions, greedy)

    def learn(self, num_updates):
        if len(self.mem) < BATCH_SIZE:
            return None

        loss_sum = torch.zeros((), device=device)
        q_sum = torch.zeros((), device=device)

        for _ in range(num_updates):
            s, a, r, ns, terminal = self.mem.sample(BATCH_SIZE)

            q = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q = self.target(ns).max(dim=1).values
                target = r + GAMMA * next_q * (1.0 - terminal)

            loss = self.loss_fn(q, target)

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
            self.opt.step()

            loss_sum += loss.detach()
            q_sum += q.detach().mean()

            self.updates += 1
            if self.updates % TARGET_SYNC == 0:
                self.target.load_state_dict(self.net.state_dict())

        return (loss_sum / num_updates).item(), (q_sum / num_updates).item()


# ---------------- setup ----------------
sim = Simulation(NUM_ENVS)

red = Agent()
blue = Agent()


def pick_actions(states, eps):
    red_actions = red.act(states[:, 0], eps)
    blue_actions = blue.act(states[:, 1], eps)
    host = torch.stack((red_actions, blue_actions)).to(torch.int8).cpu().numpy()
    return red_actions, blue_actions, host


# ---------------- greedy evaluation ----------------
def evaluate():
    """Greedy (eps=0) games on a separate set of envs.
    Mean game length ~320 means nobody ever returns the ball; higher = longer rallies."""
    env = Simulation(EVAL_ENVS)
    states = torch.from_numpy(env.reset_all()).to(device)
    active = np.ones(EVAL_ENVS, dtype=bool)
    lengths = np.zeros(EVAL_ENVS)

    for t in range(1, EVAL_CAP + 1):
        _, _, host = pick_actions(states, 0.0)
        next_np, _, dones_np, _ = env.step_all(host[0], host[1])

        lengths[active & dones_np] = t
        active = ~dones_np
        states = torch.from_numpy(next_np).to(device)

        if not active.any():
            break

    lengths[active] = EVAL_CAP
    return lengths.mean(), int(active.sum())


# ---------------- training ----------------
for epoch in range(EPOCHS):
    eps = max(
        EPS_END,
        EPS_START - (EPS_START - EPS_END) * epoch / EPS_DECAY_EPOCHS,
    )

    states = torch.from_numpy(sim.reset_all()).to(device)
    active = np.ones(NUM_ENVS, dtype=bool)

    step = 0
    wins_red = wins_blue = 0
    length_sum = 0

    while active.any() and step < MAX_STEPS_PER_EPOCH:
        step += 1

        # Random subset of active envs whose transitions we store.
        keep = active & (rng.random(NUM_ENVS) < PUSH_FRACTION)
        idx = torch.from_numpy(np.flatnonzero(keep)).to(device)

        red_actions, blue_actions, host = pick_actions(states, eps)

        next_np, rewards_np, dones_np, trunc_np = sim.step_all(host[0], host[1])

        terminal_np = (dones_np & ~trunc_np).astype(np.float32)

        # One host->device transfer.
        packed = torch.from_numpy(
            np.concatenate(
                (next_np.reshape(NUM_ENVS, -1), rewards_np, terminal_np[:, None]),
                axis=1,
            )
        ).to(device)

        next_states = packed[:, : 2 * STATE_SIZE].reshape(NUM_ENVS, 2, STATE_SIZE)
        rewards = packed[:, 2 * STATE_SIZE : 2 * STATE_SIZE + 2]
        terminal = packed[:, 2 * STATE_SIZE + 2]

        red.mem.push(
            states[idx, 0], red_actions[idx], rewards[idx, 0],
            next_states[idx, 0], terminal[idx],
        )
        blue.mem.push(
            states[idx, 1], blue_actions[idx], rewards[idx, 1],
            next_states[idx, 1], terminal[idx],
        )

        finished = active & dones_np
        if finished.any():
            decided = finished & ~trunc_np
            red_won = decided & (rewards_np[:, 0] > 0)

            wins_red += int(red_won.sum())
            wins_blue += int(decided.sum() - red_won.sum())
            length_sum += step * int(finished.sum())

        states = next_states
        active = ~dones_np

    unfinished = int(active.sum())
    n_finished = NUM_ENVS - unfinished
    mean_len = length_sum / n_finished if n_finished else float("nan")

    red_res = red.learn(UPDATES_PER_EPOCH) or (None, None)
    blue_res = blue.learn(UPDATES_PER_EPOCH) or (None, None)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | eps {eps:.2f} | steps {step} | "
        f"red {wins_red}, blue {wins_blue}, cut off {unfinished} | "
        f"mean game len {mean_len:.0f} | "
        f"loss red {fmt(red_res[0])}, blue {fmt(blue_res[0])} | "
        f"mean Q red {fmt(red_res[1])}, blue {fmt(blue_res[1])} | "
        f"buffer {len(red.mem)}"
    )

    if (epoch + 1) % EVAL_EVERY == 0:
        mean_len_eval, cut = evaluate()
        print(
            f"   >> greedy eval: mean game length {mean_len_eval:.0f} "
            f"(no returns = {NO_RETURN_LENGTH}), still running at cap: {cut}"
        )

    if (epoch + 1) % 50 == 0:
        torch.save(red.net.state_dict(), "../red_model.pth")
        torch.save(blue.net.state_dict(), "../blue_model.pth")


# ---------------- final save ----------------
torch.save(red.net.state_dict(), "../red_model.pth")
torch.save(blue.net.state_dict(), "../blue_model.pth")

print("\nTraining complete! Models saved.")
