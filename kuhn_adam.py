import torch
import torch.optim as optim
from functools import reduce


# Set random seed for reproducibility
torch.manual_seed(42)

# Define constants for actions
CHECK, BET, CALL, FOLD = 0, 1, 2, 3
moves = ['CHECK', 'BET', 'CALL', 'FOLD']

# Define the dimensions
d_dim = 3  # Player 1's cards (J, Q, K)
e_dim = 3  # Player 2's cards (J, Q, K)
m_dim = 4  # Player 1's action
t_dim = 4  # Player 2's action
n_dim = 4  # Player 1's second action

# Initialize the fixed tensors with uniform probabilities
d = torch.ones(d_dim) / d_dim
e = torch.ones(e_dim) / e_dim

# Initialize the parameters to be optimized
M1 = torch.zeros(d_dim, m_dim, requires_grad=True)
T = torch.zeros(e_dim, m_dim, t_dim, requires_grad=True)
M2 = torch.zeros(d_dim, m_dim, t_dim, n_dim, requires_grad=True)

# Initialize R with penalties for illegal moves
R = torch.full((d_dim, e_dim, m_dim, t_dim, n_dim), 0.)


legal = {
    CHECK: {
        CHECK: {},
        BET: {
            FOLD: {},
            CALL: {}
        }
    },
    BET: {
        FOLD: {},
        CALL: {}
    }
}

# Mask illegal moves by setting the rewards to a bad value
BAD = -3.
s = slice(None)
def set_bad(state):
    # By default we set every move in the position to BAD for the current player
    R[:, :, *state, ...] = BAD * (-1)**len(state)
    # Then we find the legal moves and recurse on them
    for a in reduce((lambda d, key: d[key]), state, legal):
        set_bad(state + (a,))
set_bad(())


# Fill in the payoffs for legal moves
for p1_card in range(3):
    for p2_card in range(3):
        if p1_card != p2_card:  # Ensure different cards
            p1_wins = 1. if p1_card > p2_card else -1.

            R[p1_card, p2_card, CHECK, CHECK, :] = 1 * p1_wins
            R[p1_card, p2_card, CHECK, BET, FOLD] = -1
            R[p1_card, p2_card, CHECK, BET, CALL] = 2 * p1_wins

            R[p1_card, p2_card, BET, FOLD, :] = 1.
            R[p1_card, p2_card, BET, CALL, :] = 2 * p1_wins

        else:
            # Ignore the case that should never happen
            R[p1_card, p2_card, :, :, :] = 0.

# Define the objective function
def objective(d, e, M1, T1, M2, R):
    pM1 = M1.softmax(dim=-1)# .detach() + (M1 - M1.detach())
    pT1 = T1.softmax(dim=-1)# .detach() + (T1 - T1.detach())
    pM2 = M2.softmax(dim=-1)# .detach() + (M2 - M2.detach())
    return torch.einsum('d,e,dm,emt,dmtn,demtn->', d, e, pM1, pT1, pM2, R)

# Set up separate optimizers for Player 1 and Player 2
optimizer_p1 = optim.Adam([M1, M2])
optimizer_p2 = optim.Adam([T])

# Number of optimization steps
warmup_steps = 1000
n_steps = 10000

# Optimization loop with decreasing learning rate
for step in range(1, n_steps + 1):
    # Update learning rate
    lr = 1/warmup_steps if step < warmup_steps else 1/step
    for opt in [optimizer_p1, optimizer_p2]:
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    # Optimize Player 1's strategy (maximize)
    optimizer_p1.zero_grad()
    obj = objective(d, e, M1, T, M2, R)
    (-obj).backward()  # Maximize
    optimizer_p1.step()

    # Optimize Player 2's strategy (minimize)
    optimizer_p2.zero_grad()
    obj = objective(d, e, M1, T, M2, R)
    obj.backward()  # Minimize
    optimizer_p2.step()

    if (step + 1) % 100 == 0:
        print(f"Step {step + 1}/{n_steps}, Objective: {obj.item():.4f}")

# Print final results
final_obj = objective(d, e, M1, T, M2, R)
print(f"\nFinal Objective: {final_obj.item():.4f}")

print("\nOptimized M1 (Player 1's initial strategy):")
for card in range(d_dim):
    print("JQK"[card])
    ps = M1[card].softmax(dim=-1)
    for m in legal.keys():
        print(f"  {moves[m]}: {ps[m].item():.4f}")

print("\nOptimized T (Player 2's strategy):")
for card in range(e_dim):
    print("JQK"[card])
    for m1 in legal.keys():
        ps = T[card, m1].softmax(dim=-1)
        for t1 in legal[m1].keys():
            print(f"  {moves[m1]} -> {moves[t1]}: {ps[t1].item():.4f}")

print("\nOptimized M2 (Player 1's response strategy):")
for card in range(d_dim):
    print("JQK"[card])
    for m1 in legal.keys():
        for t1 in legal[m1].keys():
            ps = M2[card, m1, t1].softmax(dim=-1)
            for m2 in legal[m1][t1].keys():
                print(f"  {moves[m1]} -> {moves[t1]} -> {moves[m2]}: {ps[m2].item():.4f}")
