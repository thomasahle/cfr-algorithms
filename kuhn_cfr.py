import torch
import math
import torch.nn.functional as F
from functools import reduce
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("strategy", type=str, choices=["cfr", "cfr+", "hedge", "normal-hedge"])
parser.add_argument("-n", type=int, default=10000)
parser.add_argument("-cards", type=int, default=3)
args = parser.parse_args()

# Set random seed for reproducibility
# torch.manual_seed(42)
n_cards = args.cards
card_names = [str(i) for i in range(11-n_cards+3, 11)] + ["J", "Q", "K"]

# Define constants for actions
CHECK, BET, CALL, FOLD = 0, 1, 2, 3
moves = ["CHECK", "BET", "CALL", "FOLD"]

# Define the dimensions
d_dim = n_cards  # Player 1's cards (J, Q, K)
e_dim = n_cards  # Player 2's cards (J, Q, K)
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
            de[i, j] = 1.0 / (d_dim + e_dim)

# Initialize R with penalties for illegal moves
R = torch.empty((d_dim, e_dim, m_dim, t_dim, n_dim))

legal = {CHECK: {CHECK: {}, BET: {FOLD: {}, CALL: {}}}, BET: {FOLD: {}, CALL: {}}}

# Mask illegal moves by setting the rewards to a bad value
BAD = -3.0
s = slice(None)


def set_bad(state):
    # By default we set every move in the position to BAD for the current player
    R[:, :, *state, ...] = BAD * (-1) ** len(state)
    # Then we find the legal moves and recurse on them
    for a in reduce((lambda d, key: d[key]), state, legal):
        set_bad(state + (a,))


set_bad(())

# Fill in the payoffs for legal moves
for p1_card in range(n_cards):
    for p2_card in range(n_cards):
        if p1_card != p2_card:  # Ensure different cards
            p1_wins = 1.0 if p1_card > p2_card else -1.0

            R[p1_card, p2_card, CHECK, CHECK, :] = 1 * p1_wins
            R[p1_card, p2_card, CHECK, BET, FOLD] = -1
            R[p1_card, p2_card, CHECK, BET, CALL] = 2 * p1_wins

            R[p1_card, p2_card, BET, FOLD, :] = 1.0
            R[p1_card, p2_card, BET, CALL, :] = 2 * p1_wins


# Number of CFR iterations
n_iterations = args.n

# Initialize strategy sums and regret sums
M1_regret = torch.zeros(d_dim, m_dim)
T_regret = torch.zeros(e_dim, m_dim, t_dim)
M2_regret = torch.zeros(d_dim, m_dim, t_dim, n_dim)

M1_r2 = torch.zeros(d_dim, m_dim)
T_r2 = torch.zeros(e_dim, m_dim, t_dim)
M2_r2 = torch.zeros(d_dim, m_dim, t_dim, n_dim)

M1_sum = torch.zeros_like(M1_regret)
T_sum = torch.zeros_like(T_regret)
M2_sum = torch.zeros_like(M2_regret)

def calc_exploitability(strat_M1, strat_T, strat_M2):
    U = torch.einsum('de,dm,emt,dmtn,demtn->', de, strat_M1, strat_T, strat_M2, R)
    expl1 = torch.einsum('de,emt,demtn->dmtn', de, strat_T, R).max(-1).values.sum(-1).max(-1).values.sum()
    expl2 = torch.einsum('de,dm,dmtn,demtn->emt', de, strat_M1, strat_M2, R).min(-1).values.sum()
    return expl1-U, U-expl2

def get_strat(regret, r2, cnt):
    if args.strategy == "cfr":
        regret = regret.clamp(min=1e-8)
        return regret / regret.sum(dim=-1, keepdim=True)
    elif args.strategy == "cfr+":
        regret = regret.clamp_(min=1e-8)  # Note the in-place operation
        return regret / regret.sum(dim=-1, keepdim=True)
    elif args.strategy == "hedge":
        std = (r2.mean() + 1).sqrt()
        C = math.log(cnt + regret.shape[-1])  # Like AlphaZero
        return (regret * C / std).softmax(dim=-1)
    elif args.strategy == "normal-hedge":
        C = math.log(cnt + regret.shape[-1])  # Like AlphaZero
        regret = regret.clamp(min=1e-8)
        ps = regret * torch.exp(r2/cnt/C/2)
        return ps / ps.sum(dim=-1, keepdim=True)

# CFR training loop
for iteration in range(1, n_iterations + 1):
    # CFR+ (Optional)
    M1 = get_strat(M1_regret, M1_r2, iteration)
    T = get_strat(T_regret, T_r2, iteration)
    M2 = get_strat(M2_regret, M2_r2, iteration)

    # Compute expected values. These are the same as the gradients
    # of the strategy with respect to each strategy tensor
    EV_M1 = torch.einsum("de,emt,dmtn,demtn->dm", de, T, M2, R)
    EV_T = torch.einsum("de,dm,dmtn,demtn->emt", de, M1, M2, R)
    EV_M2 = torch.einsum("de,dm,emt,demtn->dmtn", de, M1, T, R)

    # Compute counterfactual regrets
    # cf = D - E[D]
    CF_M1 = EV_M1 - (EV_M1 * M1).sum(dim=-1, keepdim=True)
    CF_T = EV_T - (EV_T * T).sum(dim=-1, keepdim=True)
    CF_M2 = EV_M2 - (EV_M2 * M2).sum(dim=-1, keepdim=True)

    # What happens without normalization?
    # It seems to work with softmax only. I guess because softmax
    # doesn't care about additive constants.
    # CF_M1 = CF_M1
    # CF_T = CF_T
    # CF_M2 = EV_M2

    # Update regret sums
    M1_regret += CF_M1
    T_regret -= CF_T  # Note the negative sign for player 2
    M2_regret += CF_M2

    # TODO: Could do "linear" or "quadratic" cfr here, by multiplying
    # the new CF's by iteration or iteration^2

    # Update strategy sums
    M1_sum += M1
    T_sum += T
    M2_sum += M2

    # Update variances
    if args.strategy in ("hedge", "normal-hedge"):
        M1_r2 += CF_M1 ** 2
        T_r2 += CF_T ** 2
        M2_r2 += CF_M2 ** 2

    if iteration % 100 == 0:
        avg_M1 = M1_sum / iteration
        avg_T = T_sum / iteration
        avg_M2 = M2_sum / iteration
        obj = torch.einsum("de,dm,emt,dmtn,demtn->", de, avg_M1, avg_T, avg_M2, R)
        expl1, expl2 = calc_exploitability(avg_M1, avg_T, avg_M2)
        mean = (expl1 + expl2)/2
        print(
            f"Iteration {iteration}/{args.n}"
            f", Objective: {obj:.4f}"
            f", Expl: ({expl1:.4f} + {expl2:.4f})/2 = {mean:.4f}"
        )
        if mean < 0.01:
            break

# Compute final average strategies
avg_M1 = M1_sum / M1_sum.sum(dim=-1, keepdim=True)
avg_T = T_sum / T_sum.sum(dim=-1, keepdim=True)
avg_M2 = M2_sum / M2_sum.sum(dim=-1, keepdim=True)


# Print final results
final_obj = torch.einsum("de,dm,emt,dmtn,demtn->", de, avg_M1, avg_T, avg_M2, R)
print(f"\nFinal Objective: {final_obj.item():.4f}")


print("\nOptimized M1 (Player 1's initial strategy):")
for card in range(d_dim):
    print(card_names[card])
    for m in legal.keys():
        print(f"  {moves[m]}: {avg_M1[card, m].item():.4f}")

print("\nOptimized T (Player 2's strategy):")
for card in range(e_dim):
    print(card_names[card])
    for m1 in legal.keys():
        for t1 in legal[m1].keys():
            print(f"  {moves[m1]} -> {moves[t1]}: {avg_T[card, m1, t1].item():.4f}")

print("\nOptimized M2 (Player 1's response strategy):")
for card in range(d_dim):
    print(card_names[card])
    for m1 in legal.keys():
        for t1 in legal[m1].keys():
            for m2 in legal[m1][t1].keys():
                print(
                    f"  {moves[m1]} -> {moves[t1]} -> {moves[m2]}: {avg_M2[card, m1, t1, m2].item():.4f}"
                )
