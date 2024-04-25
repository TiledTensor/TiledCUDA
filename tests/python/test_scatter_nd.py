import torch
import random
import sys
import unittest

sys.path.append('./')


class TestGemm(unittest.TestCase):

    def _compute_output_shape(self, index_dims, input_dims):
        end_size = index_dims[-1]
        out_shape = index_dims[:-1]
        for i in range(len(input_dims) - end_size):
            out_shape.append(input_dims[len(index_dims) + i])
        return out_shape

    def setUp(self):
        torch.manual_seed(1234)

    def test_scatter_nd(self):
        data_shape = [7, 8, 9, 10]
        data = torch.empty(data_shape, dtype=torch.float32,
                           device='cuda').fill_(5.0)
        scatter_data = data.flatten()

        indices_shape = [5, 2]
        indices = torch.empty(indices_shape, dtype=torch.int64, device='cuda')

        for i in range(indices_shape[0]):
            indices[i][0] = random.randint(0, data_shape[0] - 1)
            indices[i][1] = random.randint(0, data_shape[1] - 1)

        scatter_indices = indices.flatten()

        update_shape = self._compute_output_shape(indices_shape, data_shape)
        updates = torch.empty(update_shape, dtype=torch.float32,
                              device='cuda').fill_(10.0)
        scatter_updates = updates.flatten()

        import pytiledcuda
        pytiledcuda.scatter_nd(scatter_data, scatter_indices, scatter_updates)

        # Implement `scatter_nd` in Python.
        data[indices[:, 0], indices[:, 1]] = updates

        flattened_data = data.flatten()

        # Print data
        print(scatter_data)
        print(flattened_data)

        assert torch.allclose(scatter_data, flattened_data)


if __name__ == "__main__":
    unittest.main()
