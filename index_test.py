import unittest
import torch

def index_add(tensor, cnts, values, *batch_indices):
    flat_index = 0
    for d, i in zip(tensor.shape, batch_indices):
        flat_index = flat_index * d + i
    tensor.flatten(0, -2).index_add_(0, flat_index, values)
    if cnts is not None:
        cnts.flatten().index_add_(0, flat_index, torch.ones_like(flat_index))


def test_index_add():
    # Create a tensor of shape [2, 3, 4], initialized to zeros
    tensor = torch.zeros(2, 3, 4, dtype=int)

    # Create cnts tensor initialized to zeros
    cnts = torch.zeros(2, 3, dtype=int)

    # Define batch_indices (indices along the first two dimensions)
    batch_index1 = torch.tensor([0, 1, 0])
    batch_index2 = torch.tensor([2, 0, 2])

    # Values to add, with shape matching the remaining dimensions
    values = torch.tensor([
        [1, 2, 3, 4],    # To be added at position [0, 2, :]
        [5, 6, 7, 8],    # To be added at position [1, 0, :]
        [9, 10, 11, 12]  # To be added at position [0, 2, :] again
    ])

    # Call the index_add function
    index_add(tensor, cnts, values, batch_index1, batch_index2)

    # Manually compute the expected tensor
    expected_tensor = torch.zeros(2, 3, 4, dtype=int)
    expected_tensor[0, 2, :] += torch.tensor([1, 2, 3, 4])    # First addition
    expected_tensor[1, 0, :] += torch.tensor([5, 6, 7, 8])    # Second addition
    expected_tensor[0, 2, :] += torch.tensor([9, 10, 11, 12]) # Third addition

    # Manually compute the expected cnts
    expected_cnts = torch.zeros(2, 3, dtype=int)
    expected_cnts[0, 2] += 2  # [0, 2] was updated twice
    expected_cnts[1, 0] += 1  # [1, 0] was updated once

    # Verify that the tensor matches the expected tensor
    assert torch.equal(tensor, expected_tensor), "Tensor update is incorrect."

    # Verify that the cnts matches the expected cnts
    assert torch.equal(cnts, expected_cnts), "Counts update is incorrect."

    print("Test passed successfully.")

# Run the test
test_index_add()
