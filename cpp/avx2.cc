#include "avx2.hh"

#include <immintrin.h>
#include "misc.hh"

#define VECTOR_SIZE 16 // An AVX2 vector register could hold 16 uint16_t

static __m256i u16_512_16 = _mm256_set1_epi16(16);
static __m256i u16_512_298 = _mm256_set1_epi16(298);
static __m256i u16_512_409 = _mm256_set1_epi16(409);
static __m256i u16_512_128 = _mm256_set1_epi16(128);
static __m256i u16_512_8 = _mm256_set1_epi16(8);
static __m256i u16_512_100 = _mm256_set1_epi16(100);
static __m256i u16_512_208 = _mm256_set1_epi16(208);
static __m256i u16_512_516 = _mm256_set1_epi16(516);
static __m256i u16_512_0xff = _mm256_set1_epi16(0xff);
static __m256i u16_512_255 = _mm256_set1_epi16(255);
static __m256i u16_512_0 = _mm256_set1_epi16(0);
static __m256i u16_512_66 = _mm256_set1_epi16(66);
static __m256i u16_512_129 = _mm256_set1_epi16(129);
static __m256i u16_512_25 = _mm256_set1_epi16(25);
static __m256i u16_512_minus38 = _mm256_set1_epi16(-38);
static __m256i u16_512_74 = _mm256_set1_epi16(74);
static __m256i u16_512_112 = _mm256_set1_epi16(112);
static __m256i u16_512_94 = _mm256_set1_epi16(94);
static __m256i u16_512_18 = _mm256_set1_epi16(18);
static __m256i u16_512_256 = _mm256_set1_epi16(256);
static __m256i u16_512_42 = _mm256_set1_epi16(42);
static __m256i u16_512_153 = _mm256_set1_epi16(153);
static __m256i u16_512_4 = _mm256_set1_epi16(4);
static __m256i u16_512_2 = _mm256_set1_epi16(2);
static __m256i u16_512_38 = _mm256_set1_epi16(38);
static __m256i u16_512_24224 = _mm256_set1_epi16(24224);
static __m256i u16_512_2016 = _mm256_set1_epi16(2016);
static __m256i u16_512_5152 = _mm256_set1_epi16(5152);
static __m256i u16_512_8192 = _mm256_set1_epi16(8192);
static __m256i u16_512_16384 = _mm256_set1_epi16(16384);
static __m256i u16_512_1920 = _mm256_set1_epi16(1920);
static __m256i u16_512_120 = _mm256_set1_epi16(120);
static __m256i u16_512_47 = _mm256_set1_epi16(47);

static __m256i load_permute_mask = _mm256_set_epi16(
    7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
static __m256i shuffle_mask = _mm256_set_epi8(
    31, 29, 27, 25, 23, 21, 19, 17, 30, 28, 26, 24, 22, 20, 18, 16,
    15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
const int imm8 = 0xD8;

void solve_avx2_part2(
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
        __m256i alpha_vec = _mm256_set1_epi16(alpha);

        for (int j = 0; j < HEIGHT; j += 2) // read 2 Y data lines a time
        {
            for (int i = 0; i < WIDTH; i += VECTOR_SIZE)
            {

                size_t y_index_1 = j * WIDTH + i;
                size_t y_index_2 = (j + 1) * WIDTH + i;
                size_t uv_index = size_t(j / 2) * size_t(WIDTH / 2) + size_t(i / 2);

                // Load data
                __m256i y_vec_2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(y_data + y_index_2)));
                __m256i y_vec_1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(y_data + y_index_1)));
                __m128i u_vec_16 = _mm_cvtepu8_epi16(_mm_loadu_si64((__m64 *)(u_data + uv_index)));
                __m128i v_vec_16 = _mm_cvtepu8_epi16(_mm_loadu_si64((__m64 *)(v_data + uv_index)));
                __m256i u_vec = _mm256_permutexvar_epi16(load_permute_mask, _mm256_castsi128_si256(u_vec_16));
                __m256i v_vec = _mm256_permutexvar_epi16(load_permute_mask, _mm256_castsi128_si256(v_vec_16));

                // YUV -> ARGB
                __m256i y_vec_1_mul_42 = _mm256_mullo_epi16(u16_512_42, y_vec_1); // 42 * (y1)
                __m256i y_vec_2_mul_42 = _mm256_mullo_epi16(u16_512_42, y_vec_2); // 42 * (y2)
                __m256i v_vec_mul_153 = _mm256_mullo_epi16(u16_512_153, v_vec);   // 153 * (v)
                __m256i u_vec_mul_100 = _mm256_mullo_epi16(u16_512_100, u_vec);   // 100 * (u)
                __m256i v_vec_mul_47 = _mm256_mullo_epi16(u16_512_47, v_vec);     // 47 * (v)
                __m256i u_vec_mul_4 = _mm256_mullo_epi16(u16_512_4, u_vec);       // 4 * (u)

                __m256i r_yvec1_unclamped =
                    _mm256_add_epi16(
                        _mm256_srai_epi16(
                            _mm256_add_epi16(
                                _mm256_sub_epi16(y_vec_1_mul_42, u16_512_24224), v_vec_mul_153),
                            8),
                        _mm256_add_epi16(_mm256_sub_epi16(y_vec_1, u16_512_128), v_vec));

                __m256i r_yvec2_unclamped =
                    _mm256_add_epi16(
                        _mm256_srai_epi16(
                            _mm256_add_epi16(
                                _mm256_sub_epi16(y_vec_2_mul_42, u16_512_24224), v_vec_mul_153),
                            8),
                        _mm256_add_epi16(_mm256_sub_epi16(y_vec_2, u16_512_128), v_vec));

                __m256i g_yvec1_unclamped =
                    _mm256_add_epi16(
                        _mm256_srai_epi16(
                            _mm256_add_epi16(
                                _mm256_sub_epi16(y_vec_1_mul_42, u_vec_mul_100),
                                _mm256_add_epi16(u16_512_2016, v_vec_mul_47)),
                            8),
                        _mm256_sub_epi16(_mm256_add_epi16(y_vec_1, u16_512_128), v_vec));

                __m256i g_yvec2_unclamped =
                    _mm256_add_epi16(
                        _mm256_srai_epi16(
                            _mm256_add_epi16(
                                _mm256_sub_epi16(y_vec_2_mul_42, u_vec_mul_100),
                                _mm256_add_epi16(u16_512_2016, v_vec_mul_47)),
                            8),
                        _mm256_sub_epi16(_mm256_add_epi16(y_vec_2, u16_512_128), v_vec));

                __m256i b_yvec1_unclamped =
                    _mm256_add_epi16(
                        _mm256_srai_epi16(
                            _mm256_sub_epi16(
                                _mm256_add_epi16(
                                    y_vec_1_mul_42,
                                    u_vec_mul_4),
                                u16_512_5152),
                            8),
                        _mm256_add_epi16(
                            _mm256_sub_epi16(y_vec_1, u16_512_256),
                            _mm256_mullo_epi16(u16_512_2, u_vec)));

                __m256i b_yvec2_unclamped =
                    _mm256_add_epi16(
                        _mm256_srai_epi16(
                            _mm256_sub_epi16(
                                _mm256_add_epi16(
                                    y_vec_2_mul_42,
                                    u_vec_mul_4),
                                u16_512_5152),
                            8),
                        _mm256_add_epi16(
                            _mm256_sub_epi16(y_vec_2, u16_512_256),
                            _mm256_mullo_epi16(u16_512_2, u_vec)));

                __m256i r_yvec1_clamped = _mm256_max_epi16(_mm256_min_epi16(r_yvec1_unclamped, u16_512_255), u16_512_0);
                __m256i r_yvec2_clamped = _mm256_max_epi16(_mm256_min_epi16(r_yvec2_unclamped, u16_512_255), u16_512_0);
                __m256i g_yvec1_clamped = _mm256_max_epi16(_mm256_min_epi16(g_yvec1_unclamped, u16_512_255), u16_512_0);
                __m256i g_yvec2_clamped = _mm256_max_epi16(_mm256_min_epi16(g_yvec2_unclamped, u16_512_255), u16_512_0);
                __m256i b_yvec1_clamped = _mm256_max_epi16(_mm256_min_epi16(b_yvec1_unclamped, u16_512_255), u16_512_0);
                __m256i b_yvec2_clamped = _mm256_max_epi16(_mm256_min_epi16(b_yvec2_unclamped, u16_512_255), u16_512_0);
                __m256i r_yvec1 = _mm256_and_si256(r_yvec1_clamped, u16_512_0xff);
                __m256i r_yvec2 = _mm256_and_si256(r_yvec2_clamped, u16_512_0xff);
                __m256i g_yvec1 = _mm256_and_si256(g_yvec1_clamped, u16_512_0xff);
                __m256i g_yvec2 = _mm256_and_si256(g_yvec2_clamped, u16_512_0xff);
                __m256i b_yvec1 = _mm256_and_si256(b_yvec1_clamped, u16_512_0xff);
                __m256i b_yvec2 = _mm256_and_si256(b_yvec2_clamped, u16_512_0xff);

                // Alpha mix
                __m256i r_prime_yvec1 = _mm256_srli_epi16(_mm256_mullo_epi16(alpha_vec, r_yvec1), 8);
                __m256i g_prime_yvec1 = _mm256_srli_epi16(_mm256_mullo_epi16(alpha_vec, g_yvec1), 8);
                __m256i b_prime_yvec1 = _mm256_srli_epi16(_mm256_mullo_epi16(alpha_vec, b_yvec1), 8);
                __m256i r_prime_yvec2 = _mm256_srli_epi16(_mm256_mullo_epi16(alpha_vec, r_yvec2), 8);
                __m256i g_prime_yvec2 = _mm256_srli_epi16(_mm256_mullo_epi16(alpha_vec, g_yvec2), 8);
                __m256i b_prime_yvec2 = _mm256_srli_epi16(_mm256_mullo_epi16(alpha_vec, b_yvec2), 8);

                // ARGB -> YUV
                __m256i y_prime_yvec1 =
                    _mm256_add_epi16(
                        _mm256_srai_epi16(
                            _mm256_add_epi16(
                                _mm256_add_epi16(
                                    _mm256_sub_epi16(_mm256_mullo_epi16(u16_512_66, r_prime_yvec1), u16_512_8192),
                                    _mm256_sub_epi16(_mm256_mullo_epi16(u16_512_129, g_prime_yvec1), u16_512_16384)),
                                _mm256_mullo_epi16(u16_512_25, b_prime_yvec1)),
                            8),
                        u16_512_112);

                __m256i y_prime_yvec2 =
                    _mm256_add_epi16(
                        _mm256_srai_epi16(
                            _mm256_add_epi16(
                                _mm256_add_epi16(
                                    _mm256_sub_epi16(_mm256_mullo_epi16(u16_512_66, r_prime_yvec2), u16_512_8192),
                                    _mm256_sub_epi16(_mm256_mullo_epi16(u16_512_129, g_prime_yvec2), u16_512_16384)),
                                _mm256_mullo_epi16(u16_512_25, b_prime_yvec2)),
                            8),
                        u16_512_112);

                __m256i u_prime_yvec1 =
                    _mm256_add_epi16(
                        _mm256_srai_epi16(
                            _mm256_add_epi16(
                                _mm256_sub_epi16(
                                    _mm256_mullo_epi16(u16_512_112, b_prime_yvec1),
                                    _mm256_mullo_epi16(u16_512_38, r_prime_yvec1)),
                                _mm256_sub_epi16(
                                    u16_512_128,
                                    _mm256_mullo_epi16(u16_512_74, g_prime_yvec1))),
                            8),
                        u16_512_128);

                __m256i v_prime_yvec1 =
                    _mm256_add_epi16(
                        _mm256_srai_epi16(
                            _mm256_add_epi16(
                                _mm256_sub_epi16(
                                    _mm256_mullo_epi16(u16_512_112, r_prime_yvec1),
                                    _mm256_mullo_epi16(u16_512_94, g_prime_yvec1)),
                                _mm256_sub_epi16(
                                    u16_512_128,
                                    _mm256_mullo_epi16(u16_512_18, b_prime_yvec1))),
                            8),
                        u16_512_128);

                __m128i y_prime_yvec1_packed = _mm256_extracti128_si256(_mm256_permute4x64_epi64(
                    _mm256_packus_epi16(y_prime_yvec1, _mm256_setzero_si256()), imm8), 0);
                __m128i y_prime_yvec2_packed = _mm256_extracti128_si256(_mm256_permute4x64_epi64(
                    _mm256_packus_epi16(y_prime_yvec2, _mm256_setzero_si256()), imm8), 0);

                // AABBCCDD XXYYZZWW
                __m256i u_prime_yvec1_packed_repeated = _mm256_packus_epi16(u_prime_yvec1, _mm256_setzero_si256());
                __m256i v_prime_yvec1_packed_repeated = _mm256_packus_epi16(v_prime_yvec1, _mm256_setzero_si256());
                // 00000000 AABBCCDD 00000000 XXYYZZWW
                __m256i u_prime_yvec1_permuted = _mm256_permute4x64_epi64(u_prime_yvec1_packed_repeated, imm8);
                __m256i v_prime_yvec1_permuted = _mm256_permute4x64_epi64(v_prime_yvec1_packed_repeated, imm8);
                // 00000000 00000000 AABBCCDD XXYYZZWW
                __m256i u_prime_yvec1_shuffled = _mm256_shuffle_epi8(u_prime_yvec1_permuted, shuffle_mask);
                __m256i v_prime_yvec1_shuffled = _mm256_shuffle_epi8(v_prime_yvec1_permuted, shuffle_mask);
                // 00000000 00000000 ABCDXYZW ABCDXYZW
                __int64_t u_prime_yvec1_packed = _mm256_extract_epi64(u_prime_yvec1_shuffled, 0);
                __int64_t v_prime_yvec1_packed = _mm256_extract_epi64(v_prime_yvec1_shuffled, 0);
                // ABCDXYZW

                // Store value
                _mm_storeu_epi8((uint8_t *)y_result + image_idx * Y_SIZE + y_index_1, y_prime_yvec1_packed);
                _mm_storeu_epi8((uint8_t *)y_result + image_idx * Y_SIZE + y_index_2, y_prime_yvec2_packed);
                *(int64_t *)((uint8_t *)u_result + image_idx * U_SIZE + uv_index) = u_prime_yvec1_packed;
                *(int64_t *)((uint8_t *)v_result + image_idx * V_SIZE + uv_index) = v_prime_yvec1_packed;
            }
        }
    }
}

#undef VECTOR_SIZE