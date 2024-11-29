#include "avx.hh"

#include "misc.hh"
#include <cstdint>
#include <immintrin.h> // immintrin is for AVX

#include <cstdio>

#define VECTOR_SIZE 32 // An AVX512 vector register could hold 32 uint16_t

static __m512i u16_512_16 = _mm512_set1_epi16(16);
static __m512i u16_512_298 = _mm512_set1_epi16(298);
static __m512i u16_512_409 = _mm512_set1_epi16(409);
static __m512i u16_512_128 = _mm512_set1_epi16(128);
static __m512i u16_512_8 = _mm512_set1_epi16(8);
static __m512i u16_512_100 = _mm512_set1_epi16(100);
static __m512i u16_512_208 = _mm512_set1_epi16(208);
static __m512i u16_512_516 = _mm512_set1_epi16(516);
static __m512i u16_512_0xff = _mm512_set1_epi16(0xff);
static __m512i u16_512_255 = _mm512_set1_epi16(255);
static __m512i u16_512_0 = _mm512_set1_epi16(0);
static __m512i u16_512_66 = _mm512_set1_epi16(66);
static __m512i u16_512_129 = _mm512_set1_epi16(129);
static __m512i u16_512_25 = _mm512_set1_epi16(25);
static __m512i u16_512_minus38 = _mm512_set1_epi16(-38);
static __m512i u16_512_74 = _mm512_set1_epi16(74);
static __m512i u16_512_112 = _mm512_set1_epi16(112);
static __m512i u16_512_94 = _mm512_set1_epi16(94);
static __m512i u16_512_18 = _mm512_set1_epi16(18);
static __m512i u16_512_256 = _mm512_set1_epi16(256);
static __m512i load_permute_mask = _mm512_set_epi16(
    15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9,
    8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
static __m256i shuffle_mask = _mm256_set_epi8(
    31, 29, 27, 25, 23, 21, 19, 17, 30, 28, 26, 24, 22, 20, 18, 16,
    15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
const int imm8 = 0xD8;

void solve_avx512(
    const uint8_t *y_data,
    const uint8_t *u_data,
    const uint8_t *v_data,
    uint8_t **y_result,
    uint8_t **u_result,
    uint8_t **v_result)
{
    for (int image_idx = 0; image_idx < 84; image_idx++)
    {
        const uint8_t alpha = 1 + image_idx * 3;
        __m512i alpha_vec = _mm512_set1_epi16(alpha);

        for (int j = 0; j < HEIGHT; j += 2) // read 2 Y data lines a time
        {
            for (int i = 0; i < WIDTH; i += VECTOR_SIZE)
            {
                size_t y_index_1 = j * WIDTH + i;
                size_t y_index_2 = (j + 1) * WIDTH + i;
                size_t uv_index = size_t(j / 2) * size_t(WIDTH / 2) + size_t(i / 2);

                // Load data
                __m512i y_vec_1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(y_data + y_index_1)));
                __m512i y_vec_2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(y_data + y_index_2)));
                __m128i u_vec_8 = _mm_loadu_si128((__m128i *)(u_data + uv_index));
                __m128i v_vec_8 = _mm_loadu_si128((__m128i *)(v_data + uv_index));
                __m256i u_vec_16 = _mm256_cvtepu8_epi16(u_vec_8);
                __m256i v_vec_16 = _mm256_cvtepu8_epi16(v_vec_8);
                __m512i u_vec = _mm512_permutexvar_epi16(load_permute_mask, _mm512_castsi256_si512(u_vec_16));
                __m512i v_vec = _mm512_permutexvar_epi16(load_permute_mask, _mm512_castsi256_si512(v_vec_16));

                // YUV -> ARGB
                __m512i u_minus_128 = _mm512_sub_epi16(u_vec, u16_512_128);
                __m512i v_minus_128 = _mm512_sub_epi16(v_vec, u16_512_128);
                __m512i yuv2rgb_component_1_yvec1 = _mm512_mullo_epi16(u16_512_298, _mm512_sub_epi16(y_vec_1, u16_512_16));
                __m512i yuv2rgb_component_1_yvec2 = _mm512_mullo_epi16(u16_512_298, _mm512_sub_epi16(y_vec_2, u16_512_16));
                __m512i yuv2rgb_component_2 = _mm512_mullo_epi16(u16_512_409, v_minus_128); // 409 * ((v) - 128)
                __m512i yuv2rgb_component_3 = _mm512_mullo_epi16(u16_512_100, u_minus_128); // 100 * ((u) - 128)
                __m512i yuv2rgb_component_4 = _mm512_mullo_epi16(u16_512_208, v_minus_128); // 208 * ((v) - 128)
                __m512i yuv2rgb_component_5 = _mm512_mullo_epi16(u16_512_516, u_minus_128); // 516 * ((u) - 128)
                __m512i r_yvec1_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            yuv2rgb_component_1_yvec1,
                            yuv2rgb_component_2),
                        u16_512_128),
                    8);
                __m512i r_yvec2_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            yuv2rgb_component_1_yvec2,
                            yuv2rgb_component_2),
                        u16_512_128),
                    8);
                __m512i g_yvec1_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_sub_epi16(
                            _mm512_sub_epi16(
                                yuv2rgb_component_1_yvec1,
                                yuv2rgb_component_3),
                            yuv2rgb_component_4),
                        u16_512_128),
                    8);
                __m512i g_yvec2_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_sub_epi16(
                            _mm512_sub_epi16(
                                yuv2rgb_component_1_yvec2,
                                yuv2rgb_component_3),
                            yuv2rgb_component_4),
                        u16_512_128),
                    8);
                __m512i b_yvec1_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            yuv2rgb_component_1_yvec1,
                            yuv2rgb_component_5),
                        u16_512_128),
                    8);
                __m512i b_yvec2_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            yuv2rgb_component_1_yvec2,
                            yuv2rgb_component_5),
                        u16_512_128),
                    8);

                __m512i r_yvec1_clamped = _mm512_max_epi16(_mm512_min_epi16(r_yvec1_unclamped, u16_512_255), u16_512_0);
                __m512i r_yvec2_clamped = _mm512_max_epi16(_mm512_min_epi16(r_yvec2_unclamped, u16_512_255), u16_512_0);
                __m512i g_yvec1_clamped = _mm512_max_epi16(_mm512_min_epi16(g_yvec1_unclamped, u16_512_255), u16_512_0);
                __m512i g_yvec2_clamped = _mm512_max_epi16(_mm512_min_epi16(g_yvec2_unclamped, u16_512_255), u16_512_0);
                __m512i b_yvec1_clamped = _mm512_max_epi16(_mm512_min_epi16(b_yvec1_unclamped, u16_512_255), u16_512_0);
                __m512i b_yvec2_clamped = _mm512_max_epi16(_mm512_min_epi16(b_yvec2_unclamped, u16_512_255), u16_512_0);
                __m512i r_yvec1 = _mm512_and_si512(r_yvec1_clamped, u16_512_0xff);
                __m512i r_yvec2 = _mm512_and_si512(r_yvec2_clamped, u16_512_0xff);
                __m512i g_yvec1 = _mm512_and_si512(g_yvec1_clamped, u16_512_0xff);
                __m512i g_yvec2 = _mm512_and_si512(g_yvec2_clamped, u16_512_0xff);
                __m512i b_yvec1 = _mm512_and_si512(b_yvec1_clamped, u16_512_0xff);
                __m512i b_yvec2 = _mm512_and_si512(b_yvec2_clamped, u16_512_0xff);

                // Alpha mix
                __m512i r2_yvec1 = _mm512_srli_epi16(_mm512_mullo_epi16(alpha_vec, r_yvec1), 8);
                __m512i g2_yvec1 = _mm512_srli_epi16(_mm512_mullo_epi16(alpha_vec, g_yvec1), 8);
                __m512i b2_yvec1 = _mm512_srli_epi16(_mm512_mullo_epi16(alpha_vec, b_yvec1), 8);
                __m512i r2_yvec2 = _mm512_srli_epi16(_mm512_mullo_epi16(alpha_vec, r_yvec2), 8);
                __m512i g2_yvec2 = _mm512_srli_epi16(_mm512_mullo_epi16(alpha_vec, g_yvec2), 8);
                __m512i b2_yvec2 = _mm512_srli_epi16(_mm512_mullo_epi16(alpha_vec, b_yvec2), 8);

                // ARGB -> YUV
                __m512i y2_yvec1 =
                    _mm512_add_epi16(
                        _mm512_srli_epi16(
                            _mm512_add_epi16(
                                _mm512_add_epi16(
                                    _mm512_add_epi16(
                                        _mm512_mullo_epi16(u16_512_66, r2_yvec1),
                                        _mm512_mullo_epi16(u16_512_129, g2_yvec1)),
                                    _mm512_mullo_epi16(u16_512_25, b2_yvec1)),
                                u16_512_128),
                            8),
                        u16_512_16);
                __m512i y2_yvec2 =
                    _mm512_add_epi16(
                        _mm512_srli_epi16(
                            _mm512_add_epi16(
                                _mm512_add_epi16(
                                    _mm512_add_epi16(
                                        _mm512_mullo_epi16(u16_512_66, r2_yvec2),
                                        _mm512_mullo_epi16(u16_512_129, g2_yvec2)),
                                    _mm512_mullo_epi16(u16_512_25, b2_yvec2)),
                                u16_512_128),
                            8),
                        u16_512_16);

                __m512i u2_yvec1 =
                    _mm512_add_epi16(
                        _mm512_srli_epi16(
                            _mm512_add_epi16(
                                _mm512_add_epi16(
                                    _mm512_sub_epi16(
                                        _mm512_mullo_epi16(u16_512_minus38, r2_yvec1),
                                        _mm512_mullo_epi16(u16_512_74, g2_yvec1)),
                                    _mm512_mullo_epi16(u16_512_112, b2_yvec1)),
                                u16_512_128),
                            8),
                        u16_512_128);

                __m512i v2_yvec1 =
                    _mm512_add_epi16(
                        _mm512_srli_epi16(
                            _mm512_add_epi16(
                                _mm512_sub_epi16(
                                    _mm512_sub_epi16(
                                        _mm512_mullo_epi16(u16_512_112, r2_yvec1),
                                        _mm512_mullo_epi16(u16_512_94, g2_yvec1)),
                                    _mm512_mullo_epi16(u16_512_18, b2_yvec1)),
                                u16_512_128),
                            8),
                        u16_512_128);

                __m256i y2_yvec1_packed = _mm512_cvtepi16_epi8(y2_yvec1);
                __m256i y2_yvec2_packed = _mm512_cvtepi16_epi8(y2_yvec2);
                __m256i u2_yvec1_packed_repeated = _mm512_cvtepi16_epi8(u2_yvec1);
                __m256i v2_yvec1_packed_repeated = _mm512_cvtepi16_epi8(v2_yvec1);
                __m256i u2_yvec1_shuffled = _mm256_shuffle_epi8(u2_yvec1_packed_repeated, shuffle_mask);
                __m256i v2_yvec1_shuffled = _mm256_shuffle_epi8(v2_yvec1_packed_repeated, shuffle_mask);
                __m256i u2_yvec1_permuted = _mm256_permute4x64_epi64(u2_yvec1_shuffled, imm8);
                __m256i v2_yvec1_permuted = _mm256_permute4x64_epi64(v2_yvec1_shuffled, imm8);
                __m128i u2_yvec1_packed = _mm256_castsi256_si128(u2_yvec1_permuted);
                __m128i v2_yvec1_packed = _mm256_castsi256_si128(v2_yvec1_permuted);

                // Store value
                _mm256_storeu_epi8((uint8_t *)y_result + image_idx * Y_SIZE + y_index_1, y2_yvec1_packed);
                _mm256_storeu_epi8((uint8_t *)y_result + image_idx * Y_SIZE + y_index_2, y2_yvec2_packed);
                _mm_storeu_epi8((uint8_t *)u_result + image_idx * U_SIZE + uv_index, u2_yvec1_packed);
                _mm_storeu_epi8((uint8_t *)v_result + image_idx * V_SIZE + uv_index, v2_yvec1_packed);
            }
        }
    }
}

void test()
{
    __m256i input = _mm256_set_epi8(
        0xFF, 0xFF, 0xEE, 0xEE, 0xDD, 0xDD, 0xCC, 0xCC, 0xBB, 0xBB, 0xAA, 0xAA, 0x99, 0x99, 0x88, 0x88,
        0x77, 0x77, 0x66, 0x66, 0x55, 0x55, 0x44, 0x44, 0x33, 0x33, 0x22, 0x22, 0x11, 0x11, 0x00, 0x00);
    // __m256i output = _mm256_set_epi8(
    //     0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00,
    //     0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00);
    __m256i shuffle_mask = _mm256_set_epi8(
        31, 29, 27, 25, 23, 21, 19, 17, 30, 28, 26, 24, 22, 20, 18, 16,
        15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
    __m128i output = _mm_set_epi8(0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00);
    __m256i calc = _mm256_shuffle_epi8(input, shuffle_mask);

    char *l256_ptr = (char *)&calc;
    for (int i = 0; i < 32; i++)
    {
        printf("%02X ", l256_ptr[i] & 0xFF);
    }
    printf("\n");
    // 3120
    // 11011000
    // 0xD8
    const int imm8 = 0xD8;
    __m256i permuted = _mm256_permute4x64_epi64(calc, imm8);

    char *l256_ptr_1 = (char *)&permuted;
    for (int i = 0; i < 32; i++)
    {
        printf("%02X ", l256_ptr_1[i] & 0xFF);
    }
    printf("\n");

    __m128i calced = _mm256_castsi256_si128(permuted);
    char *low128_ptr = (char *)&calced;
    for (int i = 0; i < 16; i++)
    {
        printf("%02X ", low128_ptr[i] & 0xFF);
    }
    printf("\n");
    char *low128_ptr_1 = (char *)&output;
    for (int i = 0; i < 16; i++)
    {
        printf("%02X ", low128_ptr_1[i] & 0xFF);
    }
    printf("\n");
    int eq_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(calced, output));
    printf("eq_mask = %x\n", eq_mask);
    if (eq_mask == 0xFFFF)
    {
        printf("Equal");
    }
    else
    {
        printf("Not Equal");
    }
}

void solve_avx512_part3(
    const uint8_t *p1_y_data,
    const uint8_t *p1_u_data,
    const uint8_t *p1_v_data,
    const uint8_t *p2_y_data,
    const uint8_t *p2_u_data,
    const uint8_t *p2_v_data,
    uint8_t **y_result,
    uint8_t **u_result,
    uint8_t **v_result)
{
    for (int image_idx = 0; image_idx < 84; image_idx++)
    {
        const uint8_t alpha = 1 + image_idx * 3;
        const __m512i alpha_vec = _mm512_set1_epi16(alpha);
        for (int j = 0; j < HEIGHT; j += 2)
        {
            for (int i = 0; i < WIDTH; i += VECTOR_SIZE)
            {
                size_t y_index_1 = j * WIDTH + i;
                size_t y_index_2 = y_index_1 + WIDTH;
                size_t uv_index = size_t(j / 2) * size_t(WIDTH / 2) + size_t(i / 2);

                // Load data
                __m512i p1_y_vec_1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(p1_y_data + y_index_1)));
                __m512i p1_y_vec_2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(p1_y_data + y_index_2)));
                __m512i p2_y_vec_1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(p2_y_data + y_index_1)));
                __m512i p2_y_vec_2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i *)(p2_y_data + y_index_2)));

                __m128i p1_u_vec_8 = _mm_loadu_si128((__m128i *)(p1_u_data + uv_index));
                __m128i p1_v_vec_8 = _mm_loadu_si128((__m128i *)(p1_v_data + uv_index));
                __m128i p2_u_vec_8 = _mm_loadu_si128((__m128i *)(p2_u_data + uv_index));
                __m128i p2_v_vec_8 = _mm_loadu_si128((__m128i *)(p2_v_data + uv_index));
                __m256i p1_u_vec_16 = _mm256_cvtepu8_epi16(p1_u_vec_8);
                __m256i p1_v_vec_16 = _mm256_cvtepu8_epi16(p1_v_vec_8);
                __m256i p2_u_vec_16 = _mm256_cvtepu8_epi16(p2_u_vec_8);
                __m256i p2_v_vec_16 = _mm256_cvtepu8_epi16(p2_v_vec_8);
                __m512i p1_u_vec = _mm512_permutexvar_epi16(load_permute_mask, _mm512_castsi256_si512(p1_u_vec_16));
                __m512i p1_v_vec = _mm512_permutexvar_epi16(load_permute_mask, _mm512_castsi256_si512(p1_v_vec_16));
                __m512i p2_u_vec = _mm512_permutexvar_epi16(load_permute_mask, _mm512_castsi256_si512(p2_u_vec_16));
                __m512i p2_v_vec = _mm512_permutexvar_epi16(load_permute_mask, _mm512_castsi256_si512(p2_v_vec_16));

                // YUV -> ARGB
                __m512i p1_u_minus_128 = _mm512_sub_epi16(p1_u_vec, u16_512_128);
                __m512i p1_v_minus_128 = _mm512_sub_epi16(p1_v_vec, u16_512_128);
                __m512i p1_yuv2rgb_component_1_yvec1 = _mm512_mullo_epi16(u16_512_298, _mm512_sub_epi16(p1_y_vec_1, u16_512_16));
                __m512i p1_yuv2rgb_component_1_yvec2 = _mm512_mullo_epi16(u16_512_298, _mm512_sub_epi16(p1_y_vec_2, u16_512_16));
                __m512i p1_yuv2rgb_component_2 = _mm512_mullo_epi16(u16_512_409, p1_v_minus_128); // 409 * ((v) - 128)
                __m512i p1_yuv2rgb_component_3 = _mm512_mullo_epi16(u16_512_100, p1_u_minus_128); // 100 * ((u) - 128)
                __m512i p1_yuv2rgb_component_4 = _mm512_mullo_epi16(u16_512_208, p1_v_minus_128); // 208 * ((v) - 128)
                __m512i p1_yuv2rgb_component_5 = _mm512_mullo_epi16(u16_512_516, p1_u_minus_128); // 516 * ((u) - 128)
                __m512i p1_r_yvec1_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            p1_yuv2rgb_component_1_yvec1,
                            p1_yuv2rgb_component_2),
                        u16_512_128),
                    8);
                __m512i p1_r_yvec2_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            p1_yuv2rgb_component_1_yvec2,
                            p1_yuv2rgb_component_2),
                        u16_512_128),
                    8);
                __m512i p1_g_yvec1_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_sub_epi16(
                            _mm512_sub_epi16(
                                p1_yuv2rgb_component_1_yvec1,
                                p1_yuv2rgb_component_3),
                            p1_yuv2rgb_component_4),
                        u16_512_128),
                    8);
                __m512i p1_g_yvec2_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_sub_epi16(
                            _mm512_sub_epi16(
                                p1_yuv2rgb_component_1_yvec2,
                                p1_yuv2rgb_component_3),
                            p1_yuv2rgb_component_4),
                        u16_512_128),
                    8);
                __m512i p1_b_yvec1_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            p1_yuv2rgb_component_1_yvec1,
                            p1_yuv2rgb_component_5),
                        u16_512_128),
                    8);
                __m512i p1_b_yvec2_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            p1_yuv2rgb_component_1_yvec2,
                            p1_yuv2rgb_component_5),
                        u16_512_128),
                    8);

                __m512i p1_r_yvec1_clamped = _mm512_max_epi16(_mm512_min_epi16(p1_r_yvec1_unclamped, u16_512_255), u16_512_0);
                __m512i p1_r_yvec2_clamped = _mm512_max_epi16(_mm512_min_epi16(p1_r_yvec2_unclamped, u16_512_255), u16_512_0);
                __m512i p1_g_yvec1_clamped = _mm512_max_epi16(_mm512_min_epi16(p1_g_yvec1_unclamped, u16_512_255), u16_512_0);
                __m512i p1_g_yvec2_clamped = _mm512_max_epi16(_mm512_min_epi16(p1_g_yvec2_unclamped, u16_512_255), u16_512_0);
                __m512i p1_b_yvec1_clamped = _mm512_max_epi16(_mm512_min_epi16(p1_b_yvec1_unclamped, u16_512_255), u16_512_0);
                __m512i p1_b_yvec2_clamped = _mm512_max_epi16(_mm512_min_epi16(p1_b_yvec2_unclamped, u16_512_255), u16_512_0);
                __m512i p1_r_yvec1 = _mm512_and_si512(p1_r_yvec1_clamped, u16_512_0xff);
                __m512i p1_r_yvec2 = _mm512_and_si512(p1_r_yvec2_clamped, u16_512_0xff);
                __m512i p1_g_yvec1 = _mm512_and_si512(p1_g_yvec1_clamped, u16_512_0xff);
                __m512i p1_g_yvec2 = _mm512_and_si512(p1_g_yvec2_clamped, u16_512_0xff);
                __m512i p1_b_yvec1 = _mm512_and_si512(p1_b_yvec1_clamped, u16_512_0xff);
                __m512i p1_b_yvec2 = _mm512_and_si512(p1_b_yvec2_clamped, u16_512_0xff);

                // YUV -> ARGB
                __m512i p2_u_minus_128 = _mm512_sub_epi16(p2_u_vec, u16_512_128);
                __m512i p2_v_minus_128 = _mm512_sub_epi16(p2_v_vec, u16_512_128);
                __m512i p2_yuv2rgb_component_1_yvec1 = _mm512_mullo_epi16(u16_512_298, _mm512_sub_epi16(p2_y_vec_1, u16_512_16));
                __m512i p2_yuv2rgb_component_1_yvec2 = _mm512_mullo_epi16(u16_512_298, _mm512_sub_epi16(p2_y_vec_2, u16_512_16));
                __m512i p2_yuv2rgb_component_2 = _mm512_mullo_epi16(u16_512_409, p2_v_minus_128); // 409 * ((v) - 128)
                __m512i p2_yuv2rgb_component_3 = _mm512_mullo_epi16(u16_512_100, p2_u_minus_128); // 100 * ((u) - 128)
                __m512i p2_yuv2rgb_component_4 = _mm512_mullo_epi16(u16_512_208, p2_v_minus_128); // 208 * ((v) - 128)
                __m512i p2_yuv2rgb_component_5 = _mm512_mullo_epi16(u16_512_516, p2_u_minus_128); // 516 * ((u) - 128)
                __m512i p2_r_yvec1_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            p2_yuv2rgb_component_1_yvec1,
                            p2_yuv2rgb_component_2),
                        u16_512_128),
                    8);
                __m512i p2_r_yvec2_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            p2_yuv2rgb_component_1_yvec2,
                            p2_yuv2rgb_component_2),
                        u16_512_128),
                    8);
                __m512i p2_g_yvec1_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_sub_epi16(
                            _mm512_sub_epi16(
                                p2_yuv2rgb_component_1_yvec1,
                                p2_yuv2rgb_component_3),
                            p2_yuv2rgb_component_4),
                        u16_512_128),
                    8);
                __m512i p2_g_yvec2_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_sub_epi16(
                            _mm512_sub_epi16(
                                p2_yuv2rgb_component_1_yvec2,
                                p2_yuv2rgb_component_3),
                            p2_yuv2rgb_component_4),
                        u16_512_128),
                    8);
                __m512i p2_b_yvec1_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            p2_yuv2rgb_component_1_yvec1,
                            p2_yuv2rgb_component_5),
                        u16_512_128),
                    8);
                __m512i p2_b_yvec2_unclamped = _mm512_srli_epi16(
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            p2_yuv2rgb_component_1_yvec2,
                            p2_yuv2rgb_component_5),
                        u16_512_128),
                    8);

                __m512i p2_r_yvec1_clamped = _mm512_max_epi16(_mm512_min_epi16(p2_r_yvec1_unclamped, u16_512_255), u16_512_0);
                __m512i p2_r_yvec2_clamped = _mm512_max_epi16(_mm512_min_epi16(p2_r_yvec2_unclamped, u16_512_255), u16_512_0);
                __m512i p2_g_yvec1_clamped = _mm512_max_epi16(_mm512_min_epi16(p2_g_yvec1_unclamped, u16_512_255), u16_512_0);
                __m512i p2_g_yvec2_clamped = _mm512_max_epi16(_mm512_min_epi16(p2_g_yvec2_unclamped, u16_512_255), u16_512_0);
                __m512i p2_b_yvec1_clamped = _mm512_max_epi16(_mm512_min_epi16(p2_b_yvec1_unclamped, u16_512_255), u16_512_0);
                __m512i p2_b_yvec2_clamped = _mm512_max_epi16(_mm512_min_epi16(p2_b_yvec2_unclamped, u16_512_255), u16_512_0);
                __m512i p2_r_yvec1 = _mm512_and_si512(p2_r_yvec1_clamped, u16_512_0xff);
                __m512i p2_r_yvec2 = _mm512_and_si512(p2_r_yvec2_clamped, u16_512_0xff);
                __m512i p2_g_yvec1 = _mm512_and_si512(p2_g_yvec1_clamped, u16_512_0xff);
                __m512i p2_g_yvec2 = _mm512_and_si512(p2_g_yvec2_clamped, u16_512_0xff);
                __m512i p2_b_yvec1 = _mm512_and_si512(p2_b_yvec1_clamped, u16_512_0xff);
                __m512i p2_b_yvec2 = _mm512_and_si512(p2_b_yvec2_clamped, u16_512_0xff);

                // #define MIX(a, x1, x2) ((((a) * ((x1) - (x2))) >> 8) + x2)
                // image overlap
                /* __m512i r_prime_yvec1 =
                    _mm512_add_epi16(_mm512_srli_epi16(_mm512_mullo_epi16(alpha_vec, _mm512_sub_epi16(p1_r_yvec1, p2_r_yvec1)), 8), p2_r_yvec1);
                __m512i g_prime_yvec1 =
                    _mm512_add_epi16(_mm512_srli_epi16(_mm512_mullo_epi16(alpha_vec, _mm512_sub_epi16(p1_g_yvec1, p2_g_yvec1)), 8), p2_g_yvec1);
                __m512i b_prime_yvec1 =
                    _mm512_add_epi16(_mm512_srli_epi16(_mm512_mullo_epi16(alpha_vec, _mm512_sub_epi16(p1_b_yvec1, p2_b_yvec1)), 8), p2_b_yvec1);
                __m512i r_prime_yvec2 =
                    _mm512_add_epi16(_mm512_srli_epi16(_mm512_mullo_epi16(alpha_vec, _mm512_sub_epi16(p1_r_yvec2, p2_r_yvec2)), 8), p2_r_yvec2);
                __m512i g_prime_yvec2 =
                    _mm512_add_epi16(_mm512_srli_epi16(_mm512_mullo_epi16(alpha_vec, _mm512_sub_epi16(p1_g_yvec2, p2_g_yvec2)), 8), p2_g_yvec2);
                __m512i b_prime_yvec2 =
                    _mm512_add_epi16(_mm512_srli_epi16(_mm512_mullo_epi16(alpha_vec, _mm512_sub_epi16(p1_b_yvec2, p2_b_yvec2)), 8), p2_b_yvec2);
                */
                __m512i r_prime_yvec1 =
                    _mm512_srli_epi16(
                        _mm512_add_epi16(
                            _mm512_mullo_epi16(alpha_vec, p1_r_yvec1),
                            _mm512_mullo_epi16(_mm512_sub_epi16(u16_512_256, alpha_vec), p2_r_yvec1)),
                        8);
                __m512i g_prime_yvec1 =
                    _mm512_srli_epi16(
                        _mm512_add_epi16(
                            _mm512_mullo_epi16(alpha_vec, p1_g_yvec1),
                            _mm512_mullo_epi16(_mm512_sub_epi16(u16_512_256, alpha_vec), p2_g_yvec1)),
                        8);
                __m512i b_prime_yvec1 =
                    _mm512_srli_epi16(
                        _mm512_add_epi16(
                            _mm512_mullo_epi16(alpha_vec, p1_g_yvec1),
                            _mm512_mullo_epi16(_mm512_sub_epi16(u16_512_256, alpha_vec), p2_g_yvec1)),
                        8);

                __m512i r_prime_yvec2 =
                    _mm512_srli_epi16(
                        _mm512_add_epi16(
                            _mm512_mullo_epi16(alpha_vec, p1_r_yvec2),
                            _mm512_mullo_epi16(_mm512_sub_epi16(u16_512_256, alpha_vec), p2_r_yvec2)),
                        8);
                __m512i g_prime_yvec2 =
                    _mm512_srli_epi16(
                        _mm512_add_epi16(
                            _mm512_mullo_epi16(alpha_vec, p1_g_yvec2),
                            _mm512_mullo_epi16(_mm512_sub_epi16(u16_512_256, alpha_vec), p2_g_yvec2)),
                        8);
                __m512i b_prime_yvec2 =
                    _mm512_srli_epi16(
                        _mm512_add_epi16(
                            _mm512_mullo_epi16(alpha_vec, p1_g_yvec2),
                            _mm512_mullo_epi16(_mm512_sub_epi16(u16_512_256, alpha_vec), p2_g_yvec2)),
                        8);

                // ARGB -> YUV
                __m512i y_prime_yvec1 =
                    _mm512_add_epi16(
                        _mm512_srli_epi16(
                            _mm512_add_epi16(
                                _mm512_add_epi16(
                                    _mm512_add_epi16(
                                        _mm512_mullo_epi16(u16_512_66, r_prime_yvec1),
                                        _mm512_mullo_epi16(u16_512_129, g_prime_yvec1)),
                                    _mm512_mullo_epi16(u16_512_25, b_prime_yvec1)),
                                u16_512_128),
                            8),
                        u16_512_16);
                __m512i y_prime_yvec2 =
                    _mm512_add_epi16(
                        _mm512_srli_epi16(
                            _mm512_add_epi16(
                                _mm512_add_epi16(
                                    _mm512_add_epi16(
                                        _mm512_mullo_epi16(u16_512_66, r_prime_yvec2),
                                        _mm512_mullo_epi16(u16_512_129, g_prime_yvec2)),
                                    _mm512_mullo_epi16(u16_512_25, b_prime_yvec2)),
                                u16_512_128),
                            8),
                        u16_512_16);

                __m512i u_prime_yvec1 =
                    _mm512_add_epi16(
                        _mm512_srli_epi16(
                            _mm512_add_epi16(
                                _mm512_add_epi16(
                                    _mm512_sub_epi16(
                                        _mm512_mullo_epi16(u16_512_minus38, r_prime_yvec1),
                                        _mm512_mullo_epi16(u16_512_74, g_prime_yvec1)),
                                    _mm512_mullo_epi16(u16_512_112, b_prime_yvec1)),
                                u16_512_128),
                            8),
                        u16_512_128);

                __m512i v_prime_yvec1 =
                    _mm512_add_epi16(
                        _mm512_srli_epi16(
                            _mm512_add_epi16(
                                _mm512_sub_epi16(
                                    _mm512_sub_epi16(
                                        _mm512_mullo_epi16(u16_512_112, r_prime_yvec1),
                                        _mm512_mullo_epi16(u16_512_94, g_prime_yvec1)),
                                    _mm512_mullo_epi16(u16_512_18, b_prime_yvec1)),
                                u16_512_128),
                            8),
                        u16_512_128);

                __m256i y_prime_yvec1_packed = _mm512_cvtepi16_epi8(y_prime_yvec1);
                __m256i y_prime_yvec2_packed = _mm512_cvtepi16_epi8(y_prime_yvec2);
                __m256i u_prime_yvec1_packed_repeated = _mm512_cvtepi16_epi8(u_prime_yvec1);
                __m256i v_prime_yvec1_packed_repeated = _mm512_cvtepi16_epi8(v_prime_yvec1);
                __m256i u_prime_yvec1_shuffled = _mm256_shuffle_epi8(u_prime_yvec1_packed_repeated, shuffle_mask);
                __m256i v_prime_yvec1_shuffled = _mm256_shuffle_epi8(v_prime_yvec1_packed_repeated, shuffle_mask);
                __m256i u_prime_yvec1_permuted = _mm256_permute4x64_epi64(u_prime_yvec1_shuffled, imm8);
                __m256i v_prime_yvec1_permuted = _mm256_permute4x64_epi64(v_prime_yvec1_shuffled, imm8);
                __m128i u_prime_yvec1_packed = _mm256_castsi256_si128(u_prime_yvec1_permuted);
                __m128i v_prime_yvec1_packed = _mm256_castsi256_si128(v_prime_yvec1_permuted);

                // Store value
                _mm256_storeu_epi8((uint8_t *)y_result + image_idx * Y_SIZE + y_index_1, y_prime_yvec1_packed);
                _mm256_storeu_epi8((uint8_t *)y_result + image_idx * Y_SIZE + y_index_2, y_prime_yvec2_packed);
                _mm_storeu_epi8((uint8_t *)u_result + image_idx * U_SIZE + uv_index, u_prime_yvec1_packed);
                _mm_storeu_epi8((uint8_t *)v_result + image_idx * V_SIZE + uv_index, v_prime_yvec1_packed);
            }
        }
    }
}