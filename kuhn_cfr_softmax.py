import torch
import torch.nn.functional as F
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
# d = torch.ones(d_dim) / d_dim
# e = torch.ones(e_dim) / e_dim
# Actually it's not an outer product, since the two cards are dependent. In particular it's not possible to give both players the same card.
de = torch.zeros(d_dim, e_dim)
for i in range(d_dim):
    for j in range(e_dim):
        if i != j:
            de[i, j] = 1. / (d_dim + e_dim)

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



# Number of CFR iterations
n_iterations = 10000

# Initialize strategy sums and regret sums
M1_regret = torch.zeros(d_dim, m_dim, requires_grad=True)
T_regret = torch.zeros(e_dim, m_dim, t_dim, requires_grad=True)
M2_regret = torch.zeros(d_dim, m_dim, t_dim, n_dim, requires_grad=True)

M1_sum = torch.zeros_like(M1_regret)
T_sum = torch.zeros_like(T_regret)
M2_sum = torch.zeros_like(M2_regret)

lr = 1

# CFR training loop
for iteration in range(1, n_iterations + 1):

    M1 = M1_regret.softmax(dim=-1)
    T = T_regret.softmax(dim=-1)
    M2 = M2_regret.softmax(dim=-1)

    # Compute expected values. These are the same as the gradients
    # with respect to the probabilities rather than the logits.
    # M1_regret += lr * torch.einsum('de,emt,dmtn,demtn->dm', de, T, M2, R)
    # T_regret -= lr * torch.einsum('de,dm,dmtn,demtn->emt', de, M1, M2, R)
    # M2_regret += lr * torch.einsum('de,dm,emt,demtn->dmtn', de, M1, T, R)

    # M1.retain_grad()
    # T.retain_grad()
    # M2.retain_grad()
    M1_regret.grad = None
    M2_regret.grad = None
    T_regret.grad = None
    obj = torch.einsum('de,dm,emt,dmtn,demtn->', de, M1, T, M2, R)
    obj.backward()
    with torch.no_grad():
        # Very interesting that this works, but M1_regrets.grad doesn't.
        # M1_regret += lr * M1.grad
        # T_regret -= lr * T.grad  # Minimize the loss
        # M2_regret += lr * M2.grad

        # It's because the softmax gradient of softmax(x) @ t wrt. x
        # is p * (t - p @ t), where p = softmax(x) and t is the target.
        # So if we divide the gradient by the probabilities, we get the
        # correct gradient with respect to the logits.
        M1_regret += lr * M1_regret.grad / (M1+1e-8)
        T_regret -= lr * T_regret.grad / (T+1e-8)  # Minimize the loss
        M2_regret += lr * M2_regret.grad / (M2+1e-8)

    # Update strategy sums
    M1_sum += M1
    T_sum += T
    M2_sum += M2

    if iteration % 1000 == 0:
        avg_M1 = M1_sum / iteration
        avg_T = T_sum / iteration
        avg_M2 = M2_sum / iteration
        obj = torch.einsum('de,dm,emt,dmtn,demtn->', de, avg_M1, avg_T, avg_M2, R)
        print(f"Iteration {iteration}/{n_iterations}, Objective: {obj.item():.4f}")

# Compute final average strategies
avg_M1 = M1_sum / n_iterations
avg_T = T_sum / n_iterations
avg_M2 = M2_sum / n_iterations

# Print final results
final_obj = torch.einsum('de,dm,emt,dmtn,demtn->', de, avg_M1, avg_T, avg_M2, R)
print(f"\nFinal Objective: {final_obj.item():.4f}")


print("\nOptimized M1 (Player 1's initial strategy):")
for card in range(d_dim):
    print("JQK"[card])
    for m in legal.keys():
        print(f"  {moves[m]}: {avg_M1[card, m].item():.4f}")

print("\nOptimized T (Player 2's strategy):")
for card in range(e_dim):
    print("JQK"[card])
    for m1 in legal.keys():
        for t1 in legal[m1].keys():
            print(f"  {moves[m1]} -> {moves[t1]}: {avg_T[card, m1, t1].item():.4f}")

print("\nOptimized M2 (Player 1's response strategy):")
for card in range(d_dim):
    print("JQK"[card])
    for m1 in legal.keys():
        for t1 in legal[m1].keys():
            for m2 in legal[m1][t1].keys():
                print(f"  {moves[m1]} -> {moves[t1]} -> {moves[m2]}: {avg_M2[card, m1, t1, m2].item():.4f}")
