import torch
import math, random
import torch.nn.functional as F
from functools import reduce
import argparse
from functools import reduce



parser = argparse.ArgumentParser()
parser.add_argument("strategy", type=str, choices=["cfr", "cfr+", "normal-hedge", "hedge"])
parser.add_argument("-n", type=int, default=10000)
parser.add_argument("-bs", type=int, default=1000)
parser.add_argument("-cards", type=int, default=3)
args = parser.parse_args()



CHECK, BET, CALL, FOLD = 0, 1, 2, 3
moves = ["CHECK", "BET", "CALL", "FOLD"]

n_cards = args.cards
card_names = [str(i) for i in range(11-n_cards+3, 11)] + ["J", "Q", "K"]

d_dim = n_cards  # Player 1's cards (J, Q, K)
e_dim = n_cards  # Player 2's cards (J, Q, K)
m_dim = 4  # Player 1's action
t_dim = 4  # Player 2's action
n_dim = 4  # Player 1's second action

# Initialize the fixed tensors with uniform probabilities
de = torch.zeros(d_dim, e_dim)
for i in range(d_dim):
    for j in range(e_dim):
        if i != j:
            de[i, j] = 1.0 / (d_dim + e_dim)

# Initialize the valid card combinations
valid_pairs = [(d, e) for d in range(d_dim) for e in range(e_dim) if d != e]
valid_pairs = torch.tensor(valid_pairs)  # shape: (num_pairs, 2)
num_pairs = valid_pairs.shape[0]

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

# TODO: Maybe it's better to do masking when computing the strategies,
# rather than trying to force it all through the regrets.


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
batch_size = args.bs  # Adjust batch size as needed
#lr = 0.1  # Learning rate

# Initialize regret sums and strategy sums
M1_regret = torch.zeros(d_dim, m_dim)
T_regret = torch.zeros(e_dim, m_dim, t_dim)
M2_regret = torch.zeros(d_dim, m_dim, t_dim, n_dim)

M1_r2 = torch.zeros(d_dim, m_dim)
T_r2 = torch.zeros(e_dim, m_dim, t_dim)
M2_r2 = torch.zeros(d_dim, m_dim, t_dim, n_dim)

M1_sum = torch.zeros_like(M1_regret)
T_sum = torch.zeros_like(T_regret)
M2_sum = torch.zeros_like(M2_regret)

def index_add(tensor, values, *batch_indices):
    # Like torch.index_add, but supports multiple index dimensions
    flat_index = 0
    for d, i in zip(tensor.shape, batch_indices):
        flat_index = flat_index * d + i
    tensor.flatten(0, -2).index_add_(0, flat_index, values)

def get_strat(regrets, r2, cnt):
    #if random.random() < .5:
    if args.strategy in ("cfr", "cfr+"):
        #lb = (r2.mean(dim=-1) + 1e-8).sqrt().unsqueeze(-1)
        #print(lb)
        #lb = (r2/(cnt+1).unsqueeze(-1)).mean() + 1
        #lb = (regrets.clamp(min=1)/(1+cnt.unsqueeze(-1))).mean().sqrt()
        lb = 1
        #lb = R.std() / 2
        positive_regrets = regrets.clamp(min=lb)
        #lb = positive_regrets.mean(dim=-1, keepdim=True).sqrt()
        #lb = r2.mean(dim=-1, keepdim=True).sqrt() / (cnt+1).sqrt().unsqueeze(-1)
        #lb = 0
        #return positive_regrets / positive_regrets.sum(dim=-1, keepdim=True)
        return positive_regrets / positive_regrets.sum(dim=-1, keepdim=True)
    elif args.strategy == "hedge":
        #else:
        # We use scaling formula from Chaudhuri et al. 2009
        # "A Parameter-free Hedging Algorithm"
        std = (r2.mean(dim=-1) + 1e-8).sqrt().unsqueeze(-1)
        C = math.sqrt(math.log(regrets.shape[-1]))
        return (regrets * C / std).softmax(dim=-1)

    elif args.strategy == "normal-hedge":
        C = math.sqrt(math.log(regrets.shape[-1]))
        #C = 1
        var = r2.mean(dim=-1, keepdims=True) + 1
        C = math.log(regrets.shape[-1])
        # But if anything, this is just a sharpened version of cfr, so why should
        # I expect it to work better in those hard MC situations?
        ps = regrets.clamp(min=1) * torch.exp(r2 / var / C)
        return ps / ps.sum(dim=-1, keepdim=True)

def calc_exploitability(strat_M1, strat_T, strat_M2):
    U = torch.einsum('de,dm,emt,dmtn,demtn->', de, strat_M1, strat_T, strat_M2, R)
    expl1 = torch.einsum('de,emt,demtn->dmtn', de, strat_T, R).max(-1).values.sum(-1).max(-1).values.sum()
    expl2 = torch.einsum('de,dm,dmtn,demtn->emt', de, strat_M1, strat_M2, R).min(-1).values.sum()
    return expl1-U, U-expl2


# CFR training loop
for T in range(1, n_iterations + 1):
    # Sample a batch of sum (d, e) pairs
    indices = torch.randint(high=num_pairs, size=(batch_size,))
    sampled_pairs = valid_pairs[indices]
    sampled_d = sampled_pairs[:, 0]
    sampled_e = sampled_pairs[:, 1]

    # TODO: This is not CFR+, since we are clamping regrets in each iteration,
    # rather than storing them clamped.
    # But maybe that's actually a good thing for Monte Carlo, since we have some
    # pretty bad estimates of the regrets in the beginning.

    # Sampling m1 actions using regret-matching
    M1_strategy = get_strat(
            M1_regret[sampled_d],
            M1_r2[sampled_d],
            M1_sum[sampled_d].sum(-1)
            )
    index_add(M1_sum, M1_strategy, sampled_d)
    m1 = torch.multinomial(M1_strategy, num_samples=1).squeeze(-1)

    # Sampling t actions using regret-matching
    T_strategy = get_strat(T_regret[sampled_e, m1], T_r2[sampled_e, m1], T_sum[sampled_e, m1].sum(-1))
    index_add(T_sum, T_strategy, sampled_e, m1)  # T_sum[sampled_e, m1] += strategy
    t = torch.multinomial(T_strategy, num_samples=1).squeeze(-1)  # shape: (batch_size,)

    # Sampling m2 actions using regret-matching
    M2_strategy = get_strat(M2_regret[sampled_d, m1, t], M2_r2[sampled_d, m1, t], M2_sum[sampled_d, m1, t].sum(-1))
    index_add(M2_sum, M2_strategy, sampled_d, m1, t)  # M2_sum[sampled_d, m1, t] += strategy
    m2 = torch.multinomial(M2_strategy, num_samples=1).squeeze(-1)

    U = R[sampled_d, sampled_e, m1, t, m2].unsqueeze(-1)  # shape: (batch_size, 1)

    # Compute counterfactual regrets and update M1_regret
    CF_M1 = U * (torch.eye(m_dim)[m1] / (M1_strategy+1e-8) - 1)
    index_add(M1_regret, CF_M1, sampled_d)

    # Compute counterfactual regrets and update T_regret
    CF_T = U * (torch.eye(t_dim)[t] / (T_strategy+1e-8) - 1)
    index_add(T_regret, -CF_T, sampled_e, m1)  # Note: Negative sign for T_regret

    # Compute counterfactual regrets and update M2_regret
    CF_M2 = U * (torch.eye(n_dim)[m2] / (M2_strategy+1e-8) - 1)
    index_add(M2_regret, CF_M2, sampled_d, m1, t)

    #if args.strategy == "hedge":
    index_add(M1_r2, CF_M1**2, sampled_d)
    index_add(T_r2, CF_T**2, sampled_e, m1)
    index_add(M2_r2, CF_M2**2, sampled_d, m1, t)

    if args.strategy == "cfr+":
        M1_regret.clamp_(min=0)
        T_regret.clamp_(min=0)
        M2_regret.clamp_(min=0)

    #if T % (n_iterations//10) == 0:
    if T % (100000 // args.bs) == 0:
        #print(M1_cnt)
        #print(M1_sum.sum(-1))
        #assert torch.all(M1_cnt == M1_sum.sum(dim=-1))

        # We don't actually need those counts. We can just normalize
        # the sum directly.
        avg_M1 = M1_sum / (1+M1_sum.sum(dim=-1, keepdim=True))
        avg_T = T_sum / (1+T_sum.sum(dim=-1, keepdim=True))
        avg_M2 = M2_sum / (1+M2_sum.sum(dim=-1, keepdim=True))
        #avg_M1 = M1_sum / M1_cnt.unsqueeze(-1)
        #avg_T = T_sum / T_cnt.unsqueeze(-1)
        #avg_M2 = M2_sum / M2_cnt.unsqueeze(-1)

        expl1, expl2 = calc_exploitability(avg_M1, avg_T, avg_M2)

        obj_samples = U.mean().item()
        obj = torch.einsum("de,dm,emt,dmtn,demtn->", de, avg_M1, avg_T, avg_M2, R)
        #obj_current = torch.einsum("de,dm,emt,dmtn,demtn->", de, get_strat(M1_regret,T), get_strat(T_regret,T), get_strat(M2_regret,T), R)
        # obj_current = 0
        print(f"Iteration {T}/{n_iterations}"
              f", Estimated: {obj_samples:.4f}"
              # f", Current: {obj_current:.4f}"
              f", True: {obj:.4f}"
              f", Expl: ({expl1:.4f}, {expl2:.4f})"
              )

import sys
sys.exit(0)

# Say I have a value t in [0, t_dim], utility U
# I want to add U

# Strat: 0.2, 0.3, 0.5; Us: 1, 2, 3
# I want to add:
# r[0] += 1 - E[U] = 1 - (0.2 * 1 + 0.3 * 2 + 0.5 * 3)
# r[1] += 2 - E[U] = 2 - (0.2 * 1 + 0.3 * 2 + 0.5 * 3)
# ...
# Say t is sampled according to the strat, and U is the utility at the sample
# then E[U] can be estimated from just U.
# But the leading term should maybe be U/strat[i] * I[i = t]

# Thoughts:
# I guess what some algorithms do is using sampling to get a smaller tree, and then run full
# CFR on that. However, this still creates the issue that we might have a good estimate of
# the counterfactual values in the top node (since many samples passed through it), but not
# in the lower nodes.
# Maybe what ReBel is doing is updating only the root strategy when search at the root.
# But then, during the game, using a similar search strategy in the other nodes, at which
# point we actually _do_ have good information on the counterfactual values, because those
# nodes are now the root.
# What is the thing about public belief states? When I'm searching the lower nodes, should I
# assume I don't know what my own dice are?

# Compute final average strategies

# avg_M1 = get_strat(M1_regret, n_iterations)
# avg_T = get_strat(T_regret, n_iterations)
# avg_M2 = get_strat(M2_regret, n_iterations)

avg_M1 = M1_sum / (1+M1_sum.sum(dim=-1, keepdim=True))
avg_T = T_sum / (1+T_sum.sum(dim=-1, keepdim=True))
avg_M2 = M2_sum / (1+M2_sum.sum(dim=-1, keepdim=True))

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
