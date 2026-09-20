extern "C" __global__ void keepalive_compute(float *output, int iterations) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (index % 127 + 1) * 0.001f;
    #pragma unroll 1
    for (int step = 0; step < iterations; ++step) {
        value = __fmaf_rn(value, 1.0000001f, 0.0000001f);
    }
    output[index] = value;
}
