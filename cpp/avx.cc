#include "avx.hh"

#include "misc.hh"
#include <immintrin.h> // immintrin is for AVX

void solve_avx512(
    const unsigned char *y_data,
    const unsigned char *u_data,
    const unsigned char *v_data,
    unsigned char **y_result,
    unsigned char **u_result,
    unsigned char **v_result)
{
    // 常量向量，用于计算
    __m512 k_r_factor = _mm512_set1_ps(1.140f);
    __m512 k_g_factor_u = _mm512_set1_ps(-0.394f);
    __m512 k_g_factor_v = _mm512_set1_ps(-0.581f);
    __m512 k_b_factor = _mm512_set1_ps(2.032f);

    __m512 k_y_factor = _mm512_set1_ps(0.299f);
    __m512 k_u_factor = _mm512_set1_ps(-0.147f);
    __m512 k_v_factor = _mm512_set1_ps(0.615f);

    __m512 k_g_factor = _mm512_set1_ps(0.587f);
    __m512 k_b_factor_y = _mm512_set1_ps(0.114f);
    __m512 k_u_to_b_factor = _mm512_set1_ps(0.436f);
    __m512 k_r_to_y_factor = _mm512_set1_ps(-0.515f);
    __m512 k_b_to_y_factor = _mm512_set1_ps(-0.100f);

    for (int image_idx = 0; image_idx < 84; image_idx++)
    {
        unsigned char alpha = 1 + image_idx * 3;
        __m512 alpha_percentage = _mm512_set1_ps(float(alpha) / 256.0f);

        for (int j = 0; j < HEIGHT; j += 2)
        {
            for (int i = 0; i < WIDTH; i += 16)
            {
                // 计算索引
                size_t y_index1 = j * WIDTH + i;
                size_t y_index2 = (j + 1) * WIDTH + i;
                size_t uv_index = (j / 2) * (WIDTH / 2) + (i / 2);

size_t y_index = j * WIDTH + i;
                // 加载 Y 数据（两行）
                __m512i y_line1 = _mm512_loadu_si512(&y_data[y_index]);
                __m512i y_line2 = _mm512_loadu_si512(&y_data[y_index + WIDTH]);

                // 加载 U 和 V 数据
                __m512i u_data_vec = _mm512_loadu_si512(&u_data[uv_index]);
                __m512i v_data_vec = _mm512_loadu_si512(&v_data[uv_index]);

                // 解码为浮点数
                __m512 y_flt1 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi8(y_line1, _mm512_setzero_si512()));
                __m512 y_flt2 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi8(y_line2, _mm512_setzero_si512()));
                __m512 u_flt = _mm512_cvtepi32_ps(_mm512_unpacklo_epi8(u_data_vec, _mm512_setzero_si512()));
                __m512 v_flt = _mm512_cvtepi32_ps(_mm512_unpacklo_epi8(v_data_vec, _mm512_setzero_si512()));

                // 加载Y分量（两行，16个像素）
                __m512i y_vec1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(y_data + y_index1)));
                __m512i y_vec2 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(y_data + y_index2)));

                // 加载U和V分量（每行8个元素，需要扩展为16个元素以匹配Y分量）
                __m256i u_half = _mm256_loadu_si256((const __m256i *)(u_data + uv_index));
                __m256i v_half = _mm256_loadu_si256((const __m256i *)(v_data + uv_index));

                // 广播U和V分量以扩展到16个像素
                __m512i u_vec = _mm512_inserti64x4(_mm512_castsi256_si512(u_half), u_half, 1);
                __m512i v_vec = _mm512_inserti64x4(_mm512_castsi256_si512(v_half), v_half, 1);

                // 转换为浮点数
                __m512 y_f1 = _mm512_cvtepi32_ps(y_vec1);
                __m512 y_f2 = _mm512_cvtepi32_ps(y_vec2);
                __m512 u_f = _mm512_cvtepi32_ps(u_vec);
                __m512 v_f = _mm512_cvtepi32_ps(v_vec);

                // YUV 转 ARGB
                __m512 r1 = _mm512_add_ps(y_f1, _mm512_mul_ps(k_r_factor, v_f));
                __m512 g1 = _mm512_add_ps(y_f1, _mm512_add_ps(_mm512_mul_ps(k_g_factor_u, u_f), _mm512_mul_ps(k_g_factor_v, v_f)));
                __m512 b1 = _mm512_add_ps(y_f1, _mm512_mul_ps(k_b_factor, u_f));

                __m512 r2 = _mm512_add_ps(y_f2, _mm512_mul_ps(k_r_factor, v_f));
                __m512 g2 = _mm512_add_ps(y_f2, _mm512_add_ps(_mm512_mul_ps(k_g_factor_u, u_f), _mm512_mul_ps(k_g_factor_v, v_f)));
                __m512 b2 = _mm512_add_ps(y_f2, _mm512_mul_ps(k_b_factor, u_f));

                // Alpha 混合
                __m512 r1_a = _mm512_mul_ps(r1, alpha_percentage);
                __m512 g1_a = _mm512_mul_ps(g1, alpha_percentage);
                __m512 b1_a = _mm512_mul_ps(b1, alpha_percentage);

                __m512 r2_a = _mm512_mul_ps(r2, alpha_percentage);
                __m512 g2_a = _mm512_mul_ps(g2, alpha_percentage);
                __m512 b2_a = _mm512_mul_ps(b2, alpha_percentage);

                // RGB 转 YUV
                __m512 y1_new = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(k_y_factor, r1_a), _mm512_mul_ps(k_g_factor, g1_a)), _mm512_mul_ps(k_b_factor_y, b1_a));
                __m512 u1_new = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(k_u_factor, r1_a), _mm512_mul_ps(k_g_factor, g1_a)), _mm512_mul_ps(k_u_to_b_factor, b1_a));
                __m512 v1_new = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(k_v_factor, r1_a), _mm512_mul_ps(k_r_to_y_factor, g1_a)), _mm512_mul_ps(k_b_to_y_factor, b1_a));

                __m512 y2_new = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(k_y_factor, r2_a), _mm512_mul_ps(k_g_factor, g2_a)), _mm512_mul_ps(k_b_factor_y, b2_a));
                __m512 u2_new = u1_new; // U和V在两行之间共享
                __m512 v2_new = v1_new;

                // 转换为整数
                __m512i y1_new_i = _mm512_cvtps_epi32(y1_new);
                __m512i y2_new_i = _mm512_cvtps_epi32(y2_new);
                __m512i u_new_i = _mm512_cvtps_epi32(u1_new);
                __m512i v_new_i = _mm512_cvtps_epi32(v1_new);

                // 存储结果
                _mm_storeu_si128((__m128i *)((char *)y_result + image_idx * Y_SIZE + y_index1), _mm512_cvtusepi32_epi8(y1_new_i));
                _mm_storeu_si128((__m128i *)((char *)y_result + image_idx * Y_SIZE + y_index2), _mm512_cvtusepi32_epi8(y2_new_i));
                _mm_storeu_si128((__m128i *)((char *)u_result + image_idx * U_SIZE + uv_index), _mm512_cvtusepi32_epi8(u_new_i));
                _mm_storeu_si128((__m128i *)((char *)v_result + image_idx * V_SIZE + uv_index), _mm512_cvtusepi32_epi8(v_new_i));
            }
        }
    }
}
