import torch
import math
import torch.nn.functional as F
from functools import reduce
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-strat", type=str, choices=["cfr", "cfr+"], default="cfr")
parser.add_argument("-n", type=int, default=10000)
parser.add_argument("-cards", type=int, default=3)
args = parser.parse_args()

# Set random seed for reproducibility
# torch.manual_seed(42)
n_cards = args.cards
card_names = [str(i) for i in range(11 - n_cards + 3, 11)] + ["J", "Q", "K"]

# Define constants for actions
CHECK, BET, CALL, FOLD = 0, 1, 2, 3
moves = ["CHECK", "BET", "CALL", "FOLD"]

# Define the dimensions
d_dim = n_cards  # Player 1's cards (J, Q, K)
e_dim = n_cards  # Player 2's cards (J, Q, K)
m_dim = 4  # Player 1's action
t_dim = 4  # Player 2's action
n_dim = 4  # Player 1's second action

# The prior distribution is a jont matrix, since it's not possible
# to give both players the same card.
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

# Two players zero sum
R = torch.stack([R, -R], dim=-1)  # (d, e, m, t, n, 2)

# If we want to support multi players, liars dice, say,
# we could use this
n_players = 2

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
    U = torch.einsum("de,dm,emt,dmtn,demtnp->p", de, strat_M1, strat_T, strat_M2, R)
    expl1 = (
        torch.einsum("de,emt,demtn->dmtn", de, strat_T, R[..., 0])
        .max(-1)
        .values.sum(-1)
        .max(-1)
        .values.sum()
    )
    expl2 = (
        torch.einsum("de,dm,dmtn,demtn->emt", de, strat_M1, strat_M2, R[..., 1])
        .max(-1)
        .values.sum()
    )
    return expl1 - U[0], expl2 - U[1]


def get_strat(regret):
    if args.strat == "cfr":
        regret = regret.clamp(min=1e-8)
        return regret / regret.sum(dim=-1, keepdim=True)
    elif args.strat == "cfr+":
        regret = regret.clamp_(min=1e-8)  # Note the in-place operation
        return regret / regret.sum(dim=-1, keepdim=True)


def cfrd(beta, player, tensors, reward):
    """
    Counterfactual Regret Minimization with Decomposition (CFR-D)
    Args:
        beta: The probability of reaching the current state.
        player: The first player to go.
        tensors: The policies for each player in order of play.
        reward: The reward tensor for the current state.
    Returns:
        The utility tensor depending on the hidden inputs from
        the players, for both players. Shape is (d, e, p).
    """
    if len(tensors) == 0:
        return reward  # (d, e, p) shaped

    regrets, tail = tensors[0], tensors[1:]

    strat = get_strat(regrets)  # (d, m)
    expand_strat = strat[:, None, :] if player == 0 else strat[None, :, :]  # (d, e, m)

    util = []
    for i in range(regrets.shape[-1]):
        # If I wanted real Bayesian ranges, I should normalize here.
        # Either on the d- or e-dim, depending on the player.
        # Then I would multiply by it in the base-case, and as cfrd returns.
        # Just slice everything
        util.append(
            cfrd(
                beta * expand_strat[:, :, i],
                1 - player,
                [t[:, i] for t in tail],
                reward[:, :, i],
            )
        )

    ret = torch.stack(util, dim=-1)  # (d, e, p, m) shaped

    # Select utilities relevant to the player
    util = ret[:, :, player]  # (d, e, m) shaped
    # Mix in info from the left (beta) and
    # sum away the opponent's probs, we don't need them
    util = (beta[:, :, None] * util).sum(dim=1 - player)  # (d, m) shaped
    # This basically comes from GD
    node_util = (strat * util).sum(dim=-1)  # (d,) shaped
    regrets += util - node_util.unsqueeze(-1)

    # Combine strat into return value and contract over action dim
    return (expand_strat[:, :, None] * ret).sum(dim=-1)


# CFR training loop
for iteration in range(1, args.n + 1):
    cfrd(de, 0, [M1_regret, T_regret, M2_regret], R)

    # Keep track of the average strategies. It's also possible to do this inside
    # the cfrd loop, but one has to be careful to scale things right.
    for s_sum, regrets in [(M1_sum, M1_regret), (T_sum, T_regret), (M2_sum, M2_regret)]:
        s_sum += get_strat(regrets)

    if iteration % 100 == 0:
        avg_M1 = M1_sum / iteration
        avg_T = T_sum / iteration
        avg_M2 = M2_sum / iteration
        obj = torch.einsum(
            "de,dm,emt,dmtn,demtn->", de, avg_M1, avg_T, avg_M2, R[..., 0]
        )
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
final_obj = torch.einsum("de,dm,emt,dmtn,demtn->", de, avg_M1, avg_T, avg_M2, R[..., 0])
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
