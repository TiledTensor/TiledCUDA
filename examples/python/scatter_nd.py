import torch

import random
from functools import reduce
from operator import mul

torch.ops.load_library("build/libtiledcuda.so")


def compute_output_shape(index_dims, input_dims):
    end_size = index_dims[-1]
    out_shape = index_dims[:-1]
    for i in range(len(input_dims) - end_size):
        out_shape.append(input_dims[len(index_dims) + i])
    return out_shape


def test_scatter_nd():
    data_shape = [7, 8, 9, 10]
    data_numel = reduce(mul, data_shape)
    data = torch.empty(data_shape, dtype=torch.float32,
                       device='cuda').fill_(5.0)
    scatter_data = data.flatten()

    indices_shape = [5, 2]
    indices_numel = reduce(mul, indices_shape)
    indices = torch.empty(indices_shape, dtype=torch.int64, device='cuda')

    for i in range(indices_shape[0]):
        # indices[i * indices_shape[1]] = random.randint(0, data_shape[0] - 1)
        # indices[i * indices_shape[1] +
        #         1] = random.randint(0, data_shape[1] - 1)
        indices[i][0] = random.randint(0, data_shape[0] - 1)
        indices[i][1] = random.randint(0, data_shape[1] - 1)

    scatter_indices = indices.flatten()

    slice_size = 1
    end_size = indices_shape[-1]
    for i in range(end_size, len(data_shape)):
        slice_size *= data_shape[i]

    update_shape = compute_output_shape(indices_shape, data_shape)
    # update_numel = reduce(mul, update_shape)
    updates = torch.empty(update_shape, dtype=torch.float32,
                          device='cuda').fill_(10.0)
    scatter_updates = updates.flatten()

    torch.ops.tiledcuda.scatter_nd(scatter_data, scatter_updates,
                                   scatter_indices)

    return scatter_data


if __name__ == "__main__":
    print(torch.ops.tiledcuda.scatter_nd)
    data = test_scatter_nd()
    # Print data
    print(data)
