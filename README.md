# Lab 4.1
## 如何运行
1. 克隆该仓库 `git clone https://github.com/xuehaonan27/cailab41.git`
2. 进入目录 `cd cailab41`
3. 运行编译脚本（见下）
4. 运行程序来观察时间表现和图片效果

## 运行编译脚本
```shell
bash run_cpp.sh <mode> <opt> <d_or_t>
```
1. mode: `scalar` 或 `vector`。其中`scalar`表示不使用任何向量化，因此不能运行MMX/SSE2/AVX2/AVX512，只能运行标量程序（plain）；`vector`表示使用向量化，来保证编译MMX/SSE2/AVX2/AVX512。其中标量程序（plain）可能被编译器做一些自动向量化。
2. opt: `O0`, `O1`, `O2`, `O3` 或 `Ofast`。表示优化等级。
3. d_or_t: `demo` 或 `timing`。`demo`表示快速运行来展示图片效果（展示程序的正确性）。`timing`将运行计算100次，取平均时间来进行精准的计时。

## 运行程序
```shell
# 下列程序在scalar和vector下都可以运行
# Compile target file without vectorization or with vectorization (automatically by compiler)
./build/lab41 part2 plain  ./demo/dem1.yuv ./output/part2/dem1/plain/
./build/lab41 part2 plain  ./demo/dem2.yuv ./output/part2/dem2/plain/
./build/lab41 part3 plain  ./demo/dem1.yuv ./demo/dem2.yuv ./output/part3/plain/

# 下列程序只有在vector下可以运行
# Compile target file with vectorization
./build/lab41 part2 mmx    ./demo/dem1.yuv ./output/part2/dem1/mmx/
./build/lab41 part2 sse2   ./demo/dem1.yuv ./output/part2/dem1/sse2/
./build/lab41 part2 avx2   ./demo/dem1.yuv ./output/part2/dem1/avx2/
./build/lab41 part2 avx512 ./demo/dem1.yuv ./output/part2/dem1/avx512/

./build/lab41 part2 mmx    ./demo/dem2.yuv ./output/part2/dem2/mmx/
./build/lab41 part2 sse2   ./demo/dem2.yuv ./output/part2/dem2/sse2/
./build/lab41 part2 avx2   ./demo/dem2.yuv ./output/part2/dem2/avx2/
./build/lab41 part2 avx512 ./demo/dem2.yuv ./output/part2/dem2/avx512/

./build/lab41 part3 mmx    ./demo/dem1.yuv ./demo/dem2.yuv ./output/part3/mmx/
./build/lab41 part3 sse2   ./demo/dem1.yuv ./demo/dem2.yuv ./output/part3/sse2/
./build/lab41 part3 avx2   ./demo/dem1.yuv ./demo/dem2.yuv ./output/part3/avx2/
./build/lab41 part3 avx512 ./demo/dem1.yuv ./demo/dem2.yuv ./output/part3/avx512/
```
