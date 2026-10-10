// Day 1 手工作业：vec_add.cu —— 第一个 CUDA kernel
// 目标：c[i] = a[i] + b[i]，N = 1<<20（1048576 个 float）
//
// 【类比你的 ISP 背景】
//   Grid/Block/Thread ≈ 一张大图切给 NPE 集群里 N 个 core，每个 core 处理一个 tile
//   warp（32 线程锁步执行）≈ SIMD 向量单元：32 个 lane 同一条指令，divergence 就是 lane 被 mask 掉
//   编译命令：nvcc -O2 -arch=sm_120 vec_add.cu -o vec_add
//   跑 nsys：  nsys profile -o vec_add ./vec_add

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N (1 << 20)
#define CUDA_CHECK(x) do { \
    cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(1); \
    } \
} while (0)

// kernel：每个线程算一个元素
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    // TODO(任务 1)：用 blockDim.x / blockIdx.x / threadIdx.x 算出全局线程编号 i
    int i = /* blockIdx.x * blockDim.x + threadIdx.x ? */;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    float *a, *b, *c;
    size_t bytes = N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));

    // 填 host 数据：a[i]=i, b[i]=1.0 → 期望 c[i]=i+1（方便逐元素校验）
    float* ha = (float*)malloc(bytes);
    float* hb = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { ha[i] = (float)i; hb[i] = 1.0f; }

    // TODO(你的任务 2)：把 ha/hb 拷到显存（cudaMemcpyHostToDevice，方向参数别写反）
    // CUDA_CHECK(cudaMemcpy(...));

    // TODO(你的任务 3)：启动 kernel
    // 思考：block 开多大？（提示：warp=32 的倍数；从 256 开始）
    // grid 开多大？（提示：ceil(N / block)，向上取整怎么写？）
    // vec_add<<<grid, block>>>(a, b, c, N);

    // TODO(你的任务 4)：同步 + 拷回 + 校验
    // 校验方式：至少抽查 c[0]==1, c[N/2]==N/2+1, c[N-1]==N（不要只打印"跑完了"）

    // 清理：cudaFree × 3, free × 3

    printf("NOT RUN (skeleton)\n");
    return 0;
}
