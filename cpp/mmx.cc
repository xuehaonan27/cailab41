#include "mmx.hh"

#include "misc.hh"
#include <mmintrin.h> // mmintrin is for MMX

// void solve_mmx(
//     const unsigned char *y_data,
//     const unsigned char *u_data,
//     const unsigned char *v_data,
//     unsigned char **y_result,
//     unsigned char **u_result,
//     unsigned char **v_result)
// {
//     const __m64 zero = _mm_setzero_si64();
//     const __m64 one = _mm_set1_pi32(1);
//     const __m64 factor_yuv_r = _mm_set1_pi32(static_cast<int>(0.299 * 256));
//     const __m64 factor_yuv_g = _mm_set1_pi32(static_cast<int>(0.587 * 256));
//     const __m64 factor_yuv_b = _mm_set1_pi32(static_cast<int>(0.114 * 256));
//     const __m64 factor_yuv_u = _mm_set1_pi32(static_cast<int>(-0.147 * 256));
//     const __m64 factor_yuv_v = _mm_set1_pi32(static_cast<int>(0.615 * 256));
//     const __m64 factor_yuv_g2 = _mm_set1_pi32(static_cast<int>(-0.289 * 256));
//     const __m64 factor_yuv_v2 = _mm_set1_pi32(static_cast<int>(-0.515 * 256));
//     const __m64 factor_yuv_u2 = _mm_set1_pi32(static_cast<int>(-0.100 * 256));

//     for (int image_idx = 0; image_idx < 84; image_idx++)
//     {
//         unsigned char alpha = static_cast<unsigned char>(1 + image_idx * 3);
//         const __m64 alpha_mmx = _mm_set1_pi32(alpha);

//         for (int j = 0; j < HEIGHT; j++)
//         {
//             for (int i = 0; i < WIDTH; i += 8) // process 8 bytes a time
//             {
//                 size_t y_index = j * WIDTH + i;
//                 size_t uv_index = size_t(j / 2) * size_t(WIDTH / 2) + size_t(i / 2); // Cannot be changed

//                 // https://community.intel.com/t5/Intel-ISA-Extensions/load-store-intrinsics-in-MMX-technology-m64/m-p/824668
//                 __m64 y_vec = *((const __m64 *)(y_data + y_index)); // Load 8 y values
//                 __m64 u_vec = *((const __m64 *)(u_data + uv_index)); // Load
//                 __m64 v_vec = *((const __m64 *)(v_data + uv_index));

//                 u_vec = _mm_unpackhi_pi8(u_vec, u_vec);

//                 // YUV to ARGB
//                 __m64 r_inter = _mm_mullo_pi16()
//                 float r = y + 1.140 * v;



//                 float g = y - 0.394 * u - 0.581 * v;
//                 float b = y + 2.032 * u;

//                 // Alpha mix
//                 float r2 = alpha * r / 256;
//                 float g2 = alpha * g / 256;
//                 float b2 = alpha * b / 256;

//                 float y2 = 0.299 * r + 0.587 * g + 0.114 * b;
//                 float u2 = -0.147 * r - 0.289 * g + 0.436 * b;
//                 float v2 = 0.615 * r - 0.515 * g - 0.100 * b;

//                 *(unsigned char *)((char *)y_result + image_idx * Y_SIZE + y_index) = (unsigned char)(y2);
//                 *(unsigned char *)((char *)u_result + image_idx * U_SIZE + uv_index) = (unsigned char)(u2);
//                 *(unsigned char *)((char *)v_result + image_idx * V_SIZE + uv_index) = (unsigned char)(v2);
//             }
//         }
//     }
// }
