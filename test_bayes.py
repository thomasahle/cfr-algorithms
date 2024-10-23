import torch
torch.manual_seed(1)

def sample(probs, num_samples=1):
    # Returns matrix of shape (num_samples, *probs.shape).
    # The last dim of the input is assumed to be class probabilities,
    # and the last dim of the output is one-hot.
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
        added_dim = True
    else:
        added_dim = False
    samples = torch.multinomial(probs.flatten(0, -2), num_samples, replacement=True)
    *in_dims, out_dim = probs.shape
    if added_dim:
        in_dims.pop(0)
    samples = samples.T.reshape([num_samples] + in_dims)
    return torch.eye(out_dim)[samples]

d1, d2, d3 = 2, 3, 4
# C = torch.softmax(torch.randn(d1), dim=-1)
C = torch.ones(d1).softmax(dim=-1)
M1 = torch.randn(d1,d2).softmax(dim=-1)
M2 = torch.randn(d1,d2,d3).softmax(dim=-1)
R = torch.randn(d1,d2,d3)

EV = torch.einsum('i,ij,ijk,ijk->',C,M1,M2,R)
print(f'Expected value: {EV}')

num_samples = 1000_000
C_samples = sample(C, num_samples)

C_EV = torch.einsum('bi,ij,ijk,ijk->b', C_samples, M1, M2, R).mean()
print(f'EV with C-samples: {C_EV}')

M_samples = sample(torch.einsum('bi,ij->bj', C_samples, M1), 1).squeeze(0)
M_EV = torch.einsum('bi,bj,ijk,ijk->b', C_samples, M_samples, M2, R).mean()
print(f'EV with M and C-samples: {M_EV}')


wrong_M_EV = torch.einsum('i,bj,ijk,ijk->b', C, M_samples, M2, R).mean()
print(f'EV with wrong conditioning: {wrong_M_EV}')

c_cond0 = torch.einsum('i,ij,bj->bi', C, M1, M_samples)
ev_c_cond0 = torch.einsum('bi,bj,ijk,ijk->b', c_cond0, M_samples, M2, R)
print(f'EV without normalization: {ev_c_cond0.mean()}')

ev_c_cond1 = ev_c_cond0 / c_cond0.sum(dim=-1)  # Could just say .sum()
print(f'EV with correct normalization: {ev_c_cond1.mean()}')
