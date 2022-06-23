// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>

#include "../ssb/ssb_gpu_utils.h"
#include "cub/test/test_util.h"
#include "utils/gpu_utils.h"

using namespace std;
using namespace cub;

float fullAggGPU(uint* d_value, int num_items, CachingDeviceAllocator& g_allocator) {
  SETUP_TIMING();

  float time_full_agg;
  unsigned long long *d_out;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(unsigned long long)));

  // Allocate temporary storage
  void            *d_temp_storage = NULL;
  size_t          temp_storage_bytes = 0;

  CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_value, d_out, num_items));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage, temp_storage_bytes));

  TIME_FUNC(CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
      d_value, d_out, num_items)), time_full_agg);

  unsigned long long result;
  CubDebugExit(cudaMemcpy(&result, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

  cout << result << endl;

  CLEANUP(d_out);
  CLEANUP(d_temp_storage);

  return time_full_agg;
}

bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main(int argc, char** argv)
{
    uint num_items          = 1 << 28;
    int num_trials          = 3;
    bool full_agg           = true;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("t", num_trials);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items>] "
            "[--t=<num trials>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

/*    uint *d_value;*/
    /*CubDebugExit(g_allocator.DeviceAllocate((void**)&d_value, sizeof(uint) * num_items));*/
    string column_name = "test";
    int len = 1<<28;
    uint* h_value = loadColumn<uint>(column_name, len);
    uint* d_value = loadColumnToGPU<uint>(h_value, len, g_allocator);

    SETUP_TIMING();

    for (int i = 0; i < num_trials; i++) {
        // Full Aggregation.
        float time_taken, global_memory_bandwidth, l2_cache_bandwidth, shared_memory_bandwidth;

        time_taken = fullAggGPU(d_value, num_items, g_allocator); // Time is in ms
        global_memory_bandwidth = ((num_items * 4) / time_taken) / 1000000.0;

/*        TIME_FUNC((l2_test<<<1024,128>>>(d_value, d_dummy)), time_taken);*/
        /*l2_cache_bandwidth = 1024 * ((128 * 128 * 8 * 10 * 4) / time_taken) / 1000000.0;*/

        /*TIME_FUNC((shared_test<<<1024,128>>>(d_value, d_dummy)), time_taken);*/
        /*shared_memory_bandwidth = ((1024 * 128 * 4003 * 4) / time_taken) / 1000000.0;*/

/*        device_copy_scalar(d_dummy, d_dummy2, num_items);*/
        /*device_copy_vector(d_dummy, d_dummy2, num_items);*/

        cout<< "{"
            << "\"time_taken\":" << time_taken << ","
            << "\"global_memory_bandwidth\":" << global_memory_bandwidth
/*            << "\"l2_cache_bandwidth\":" << l2_cache_bandwidth*/
            /*<< "\"shared_memory_bandwidth\":" << shared_memory_bandwidth*/
            << "}" << endl;
    }

    CLEANUP(d_value);

    return 0;
}
