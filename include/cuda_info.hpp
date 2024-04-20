#pragma once

namespace tiledcuda {

// Returns the maximum number of threads per multiprocessor for the given GPU.
int GetGPUMaxThreadsPerMultiProcessor(int id);

}  // namespace tiledcuda