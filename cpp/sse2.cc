#include "sse2.hh"

#include "misc.hh"
#include <emmintrin.h> // emmintrin is for SSE2
#include <cstdint>

#define VECTOR_SIZE 8

static __m128i u16_512_16 = _mm_set1_epi16(16);
static __m128i u16_512_298 = _mm_set1_epi16(298);
static __m128i u16_512_409 = _mm_set1_epi16(409);
static __m128i u16_512_128 = _mm_set1_epi16(128);
static __m128i u16_512_8 = _mm_set1_epi16(8);
static __m128i u16_512_100 = _mm_set1_epi16(100);
static __m128i u16_512_208 = _mm_set1_epi16(208);
static __m128i u16_512_516 = _mm_set1_epi16(516);
static __m128i u16_512_0xff = _mm_set1_epi16(0xff);
static __m128i u16_512_255 = _mm_set1_epi16(255);
static __m128i u16_512_0 = _mm_set1_epi16(0);
static __m128i u16_512_66 = _mm_set1_epi16(66);
static __m128i u16_512_129 = _mm_set1_epi16(129);
static __m128i u16_512_25 = _mm_set1_epi16(25);
static __m128i u16_512_minus38 = _mm_set1_epi16(-38);
static __m128i u16_512_74 = _mm_set1_epi16(74);
static __m128i u16_512_112 = _mm_set1_epi16(112);
static __m128i u16_512_94 = _mm_set1_epi16(94);
static __m128i u16_512_18 = _mm_set1_epi16(18);
static __m128i u16_512_256 = _mm_set1_epi16(256);
static __m128i u16_512_42 = _mm_set1_epi16(42);
static __m128i u16_512_153 = _mm_set1_epi16(153);
static __m128i u16_512_4 = _mm_set1_epi16(4);
static __m128i u16_512_2 = _mm_set1_epi16(2);
static __m128i u16_512_38 = _mm_set1_epi16(38);
static __m128i u16_512_24224 = _mm_set1_epi16(24224);
static __m128i u16_512_2016 = _mm_set1_epi16(2016);
static __m128i u16_512_5152 = _mm_set1_epi16(5152);
static __m128i u16_512_8192 = _mm_set1_epi16(8192);
static __m128i u16_512_16384 = _mm_set1_epi16(16384);
static __m128i u16_512_1920 = _mm_set1_epi16(1920);
static __m128i u16_512_120 = _mm_set1_epi16(120);
static __m128i u16_512_47 = _mm_set1_epi16(47);
const int imm8 = 0xD8;

void solve_sse2_part2(
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
        __m128i alpha_vec = _mm_set1_epi16(alpha);

        for (int j = 0; j < HEIGHT; j += 2) // read 2 Y data lines a time
        {
            for (int i = 0; i < WIDTH; i += VECTOR_SIZE)
            {
                size_t y_index_1 = j * WIDTH + i;
                size_t y_index_2 = (j + 1) * WIDTH + i;
                size_t uv_index = size_t(j / 2) * size_t(WIDTH / 2) + size_t(i / 2);

                // Load data
                __m128i y_vec_1 = _mm_unpacklo_epi8(_mm_set_epi64x(0, *(long long *)(y_data + y_index_1)), _mm_setzero_si128());
                __m128i y_vec_2 = _mm_unpacklo_epi8(_mm_set_epi64x(0, *(long long *)(y_data + y_index_2)), _mm_setzero_si128());
                __m128i u_loaded = _mm_set_epi32(0, 0, 0, *(int *)(u_data + uv_index));
                __m128i v_loaded = _mm_set_epi32(0, 0, 0, *(int *)(v_data + uv_index));
                __m128i u_load_repeated = _mm_unpacklo_epi8(u_loaded, u_loaded);
                __m128i v_load_repeated = _mm_unpacklo_epi8(v_loaded, v_loaded);
                __m128i u_vec = _mm_unpacklo_epi8(u_load_repeated, _mm_setzero_si128());
                __m128i v_vec = _mm_unpacklo_epi8(v_load_repeated, _mm_setzero_si128());

                // YUV -> ARGB
                __m128i y_vec_1_mul_42 = _mm_mullo_epi16(u16_512_42, y_vec_1); // 42 * (y1)
                __m128i y_vec_2_mul_42 = _mm_mullo_epi16(u16_512_42, y_vec_2); // 42 * (y2)
                __m128i v_vec_mul_153 = _mm_mullo_epi16(u16_512_153, v_vec);   // 153 * (v)
                __m128i u_vec_mul_100 = _mm_mullo_epi16(u16_512_100, u_vec);   // 100 * (u)
                __m128i v_vec_mul_47 = _mm_mullo_epi16(u16_512_47, v_vec);     // 47 * (v)
                __m128i u_vec_mul_4 = _mm_mullo_epi16(u16_512_4, u_vec);       // 4 * (u)

                __m128i r_yvec1_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(y_vec_1_mul_42, u16_512_24224), v_vec_mul_153),
                            8),
                        _mm_add_epi16(_mm_sub_epi16(y_vec_1, u16_512_128), v_vec));

                __m128i r_yvec2_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(y_vec_2_mul_42, u16_512_24224), v_vec_mul_153),
                            8),
                        _mm_add_epi16(_mm_sub_epi16(y_vec_2, u16_512_128), v_vec));

                __m128i g_yvec1_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(y_vec_1_mul_42, u_vec_mul_100),
                                _mm_add_epi16(u16_512_2016, v_vec_mul_47)),
                            8),
                        _mm_sub_epi16(_mm_add_epi16(y_vec_1, u16_512_128), v_vec));

                __m128i g_yvec2_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(y_vec_2_mul_42, u_vec_mul_100),
                                _mm_add_epi16(u16_512_2016, v_vec_mul_47)),
                            8),
                        _mm_sub_epi16(_mm_add_epi16(y_vec_2, u16_512_128), v_vec));

                __m128i b_yvec1_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_sub_epi16(
                                _mm_add_epi16(
                                    y_vec_1_mul_42,
                                    u_vec_mul_4),
                                u16_512_5152),
                            8),
                        _mm_add_epi16(
                            _mm_sub_epi16(y_vec_1, u16_512_256),
                            _mm_mullo_epi16(u16_512_2, u_vec)));

                __m128i b_yvec2_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_sub_epi16(
                                _mm_add_epi16(
                                    y_vec_2_mul_42,
                                    u_vec_mul_4),
                                u16_512_5152),
                            8),
                        _mm_add_epi16(
                            _mm_sub_epi16(y_vec_2, u16_512_256),
                            _mm_mullo_epi16(u16_512_2, u_vec)));

                __m128i r_yvec1_clamped = _mm_max_epi16(_mm_min_epi16(r_yvec1_unclamped, u16_512_255), u16_512_0);
                __m128i r_yvec2_clamped = _mm_max_epi16(_mm_min_epi16(r_yvec2_unclamped, u16_512_255), u16_512_0);
                __m128i g_yvec1_clamped = _mm_max_epi16(_mm_min_epi16(g_yvec1_unclamped, u16_512_255), u16_512_0);
                __m128i g_yvec2_clamped = _mm_max_epi16(_mm_min_epi16(g_yvec2_unclamped, u16_512_255), u16_512_0);
                __m128i b_yvec1_clamped = _mm_max_epi16(_mm_min_epi16(b_yvec1_unclamped, u16_512_255), u16_512_0);
                __m128i b_yvec2_clamped = _mm_max_epi16(_mm_min_epi16(b_yvec2_unclamped, u16_512_255), u16_512_0);
                __m128i r_yvec1 = _mm_and_si128(r_yvec1_clamped, u16_512_0xff);
                __m128i r_yvec2 = _mm_and_si128(r_yvec2_clamped, u16_512_0xff);
                __m128i g_yvec1 = _mm_and_si128(g_yvec1_clamped, u16_512_0xff);
                __m128i g_yvec2 = _mm_and_si128(g_yvec2_clamped, u16_512_0xff);
                __m128i b_yvec1 = _mm_and_si128(b_yvec1_clamped, u16_512_0xff);
                __m128i b_yvec2 = _mm_and_si128(b_yvec2_clamped, u16_512_0xff);

                // Alpha mix
                __m128i r_prime_yvec1 = _mm_srli_epi16(_mm_mullo_epi16(alpha_vec, r_yvec1), 8);
                __m128i g_prime_yvec1 = _mm_srli_epi16(_mm_mullo_epi16(alpha_vec, g_yvec1), 8);
                __m128i b_prime_yvec1 = _mm_srli_epi16(_mm_mullo_epi16(alpha_vec, b_yvec1), 8);
                __m128i r_prime_yvec2 = _mm_srli_epi16(_mm_mullo_epi16(alpha_vec, r_yvec2), 8);
                __m128i g_prime_yvec2 = _mm_srli_epi16(_mm_mullo_epi16(alpha_vec, g_yvec2), 8);
                __m128i b_prime_yvec2 = _mm_srli_epi16(_mm_mullo_epi16(alpha_vec, b_yvec2), 8);

                // ARGB -> YUV
                __m128i y_prime_yvec1 =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_add_epi16(
                                    _mm_sub_epi16(_mm_mullo_epi16(u16_512_66, r_prime_yvec1), u16_512_8192),
                                    _mm_sub_epi16(_mm_mullo_epi16(u16_512_129, g_prime_yvec1), u16_512_16384)),
                                _mm_mullo_epi16(u16_512_25, b_prime_yvec1)),
                            8),
                        u16_512_112);

                __m128i y_prime_yvec2 =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_add_epi16(
                                    _mm_sub_epi16(_mm_mullo_epi16(u16_512_66, r_prime_yvec2), u16_512_8192),
                                    _mm_sub_epi16(_mm_mullo_epi16(u16_512_129, g_prime_yvec2), u16_512_16384)),
                                _mm_mullo_epi16(u16_512_25, b_prime_yvec2)),
                            8),
                        u16_512_112);

                __m128i u_prime_yvec1 =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(
                                    _mm_mullo_epi16(u16_512_112, b_prime_yvec1),
                                    _mm_mullo_epi16(u16_512_38, r_prime_yvec1)),
                                _mm_sub_epi16(
                                    u16_512_128,
                                    _mm_mullo_epi16(u16_512_74, g_prime_yvec1))),
                            8),
                        u16_512_128);

                __m128i v_prime_yvec1 =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(
                                    _mm_mullo_epi16(u16_512_112, r_prime_yvec1),
                                    _mm_mullo_epi16(u16_512_94, g_prime_yvec1)),
                                _mm_sub_epi16(
                                    u16_512_128,
                                    _mm_mullo_epi16(u16_512_18, b_prime_yvec1))),
                            8),
                        u16_512_128);

                __int64_t y_prime_yvec1_packed = _mm_cvtsi128_si64(_mm_packus_epi16(y_prime_yvec1, _mm_setzero_si128()));
                __int64_t y_prime_yvec2_packed = _mm_cvtsi128_si64(_mm_packus_epi16(y_prime_yvec2, _mm_setzero_si128()));

                // XXYYZZWW
                __m128i u_prime_yvec1_shuffled = _mm_shufflehi_epi16(_mm_shufflelo_epi16(u_prime_yvec1, imm8), imm8);
                __m128i v_prime_yvec1_shuffled = _mm_shufflehi_epi16(_mm_shufflelo_epi16(v_prime_yvec1, imm8), imm8);
                // XYXYZWZW
                __m128i u_prime_yvec1_permuted = _mm_shuffle_epi32(u_prime_yvec1_shuffled, imm8);
                __m128i v_prime_yvec1_permuted = _mm_shuffle_epi32(v_prime_yvec1_shuffled, imm8);
                // XYZWXYZW
                __m128i u_prime_yvec1_repeated = _mm_packus_epi16(u_prime_yvec1_permuted, _mm_setzero_si128());
                __m128i v_prime_yvec1_repeated = _mm_packus_epi16(v_prime_yvec1_permuted, _mm_setzero_si128());

                int u_prime_yvec1_packed = _mm_cvtsi128_si32(u_prime_yvec1_repeated);
                int v_prime_yvec1_packed = _mm_cvtsi128_si32(v_prime_yvec1_repeated);

                // Store value
                *(__int64_t *)((uint8_t *)y_result + image_idx * Y_SIZE + y_index_1) = y_prime_yvec1_packed;
                *(__int64_t *)((uint8_t *)y_result + image_idx * Y_SIZE + y_index_2) = y_prime_yvec2_packed;
                *(int *)((uint8_t *)u_result + image_idx * U_SIZE + uv_index) = u_prime_yvec1_packed;
                *(int *)((uint8_t *)v_result + image_idx * V_SIZE + uv_index) = v_prime_yvec1_packed;
            }
        }
    }
}

void solve_sse2_part3(
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
        const __m128i alpha_vec = _mm_set1_epi16(alpha);
        const __m128i alpha_256_minus_vec = _mm_sub_epi16(u16_512_256, alpha_vec);
        for (int j = 0; j < HEIGHT; j += 2)
        {
            for (int i = 0; i < WIDTH; i += VECTOR_SIZE)
            {
                size_t y_index_1 = j * WIDTH + i;
                size_t y_index_2 = y_index_1 + WIDTH;
                size_t uv_index = size_t(j / 2) * size_t(WIDTH / 2) + size_t(i / 2);

                // Load data
                __m128i p1_y_vec_1 = _mm_unpacklo_epi8(_mm_set_epi64x(0, *(long long *)(p1_y_data + y_index_1)), _mm_setzero_si128());
                __m128i p1_y_vec_2 = _mm_unpacklo_epi8(_mm_set_epi64x(0, *(long long *)(p1_y_data + y_index_2)), _mm_setzero_si128());
                __m128i p1_u_loaded = _mm_set_epi32(0, 0, 0, *(int *)(p1_u_data + uv_index));
                __m128i p1_v_loaded = _mm_set_epi32(0, 0, 0, *(int *)(p1_v_data + uv_index));
                __m128i p1_u_load_repeated = _mm_unpacklo_epi8(p1_u_loaded, p1_u_loaded);
                __m128i p1_v_load_repeated = _mm_unpacklo_epi8(p1_v_loaded, p1_v_loaded);
                __m128i p1_u_vec = _mm_unpacklo_epi8(p1_u_load_repeated, _mm_setzero_si128());
                __m128i p1_v_vec = _mm_unpacklo_epi8(p1_v_load_repeated, _mm_setzero_si128());

                __m128i p2_y_vec_1 = _mm_unpacklo_epi8(_mm_set_epi64x(0, *(long long *)(p2_y_data + y_index_1)), _mm_setzero_si128());
                __m128i p2_y_vec_2 = _mm_unpacklo_epi8(_mm_set_epi64x(0, *(long long *)(p2_y_data + y_index_2)), _mm_setzero_si128());
                __m128i p2_u_loaded = _mm_set_epi32(0, 0, 0, *(int *)(p2_u_data + uv_index));
                __m128i p2_v_loaded = _mm_set_epi32(0, 0, 0, *(int *)(p2_v_data + uv_index));
                __m128i p2_u_load_repeated = _mm_unpacklo_epi8(p2_u_loaded, p2_u_loaded);
                __m128i p2_v_load_repeated = _mm_unpacklo_epi8(p2_v_loaded, p2_v_loaded);
                __m128i p2_u_vec = _mm_unpacklo_epi8(p2_u_load_repeated, _mm_setzero_si128());
                __m128i p2_v_vec = _mm_unpacklo_epi8(p2_v_load_repeated, _mm_setzero_si128());

                // YUV -> ARGB
                __m128i p1_y_vec_1_mul_42 = _mm_mullo_epi16(u16_512_42, p1_y_vec_1); // 42 * (y1)
                __m128i p1_y_vec_2_mul_42 = _mm_mullo_epi16(u16_512_42, p1_y_vec_2); // 42 * (y2)
                __m128i p1_v_vec_mul_153 = _mm_mullo_epi16(u16_512_153, p1_v_vec);   // 153 * (v)
                __m128i p1_u_vec_mul_100 = _mm_mullo_epi16(u16_512_100, p1_u_vec);   // 100 * (u)
                __m128i p1_v_vec_mul_47 = _mm_mullo_epi16(u16_512_47, p1_v_vec);     // 47 * (v)
                __m128i p1_u_vec_mul_4 = _mm_mullo_epi16(u16_512_4, p1_u_vec);       // 4 * (u)

                __m128i p1_r_yvec1_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(p1_y_vec_1_mul_42, u16_512_24224), p1_v_vec_mul_153),
                            8),
                        _mm_add_epi16(_mm_sub_epi16(p1_y_vec_1, u16_512_128), p1_v_vec));

                __m128i p1_r_yvec2_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(p1_y_vec_2_mul_42, u16_512_24224), p1_v_vec_mul_153),
                            8),
                        _mm_add_epi16(_mm_sub_epi16(p1_y_vec_2, u16_512_128), p1_v_vec));

                __m128i p1_g_yvec1_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(p1_y_vec_1_mul_42, p1_u_vec_mul_100),
                                _mm_add_epi16(u16_512_2016, p1_v_vec_mul_47)),
                            8),
                        _mm_sub_epi16(_mm_add_epi16(p1_y_vec_1, u16_512_128), p1_v_vec));

                __m128i p1_g_yvec2_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(p1_y_vec_2_mul_42, p1_u_vec_mul_100),
                                _mm_add_epi16(u16_512_2016, p1_v_vec_mul_47)),
                            8),
                        _mm_sub_epi16(_mm_add_epi16(p1_y_vec_2, u16_512_128), p1_v_vec));

                __m128i p1_b_yvec1_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_sub_epi16(
                                _mm_add_epi16(
                                    p1_y_vec_1_mul_42,
                                    p1_u_vec_mul_4),
                                u16_512_5152),
                            8),
                        _mm_add_epi16(
                            _mm_sub_epi16(p1_y_vec_1, u16_512_256),
                            _mm_mullo_epi16(u16_512_2, p1_u_vec)));

                __m128i p1_b_yvec2_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_sub_epi16(
                                _mm_add_epi16(
                                    p1_y_vec_2_mul_42,
                                    p1_u_vec_mul_4),
                                u16_512_5152),
                            8),
                        _mm_add_epi16(
                            _mm_sub_epi16(p1_y_vec_2, u16_512_256),
                            _mm_mullo_epi16(u16_512_2, p1_u_vec)));

                __m128i p1_r_yvec1_clamped = _mm_max_epi16(_mm_min_epi16(p1_r_yvec1_unclamped, u16_512_255), u16_512_0);
                __m128i p1_r_yvec2_clamped = _mm_max_epi16(_mm_min_epi16(p1_r_yvec2_unclamped, u16_512_255), u16_512_0);
                __m128i p1_g_yvec1_clamped = _mm_max_epi16(_mm_min_epi16(p1_g_yvec1_unclamped, u16_512_255), u16_512_0);
                __m128i p1_g_yvec2_clamped = _mm_max_epi16(_mm_min_epi16(p1_g_yvec2_unclamped, u16_512_255), u16_512_0);
                __m128i p1_b_yvec1_clamped = _mm_max_epi16(_mm_min_epi16(p1_b_yvec1_unclamped, u16_512_255), u16_512_0);
                __m128i p1_b_yvec2_clamped = _mm_max_epi16(_mm_min_epi16(p1_b_yvec2_unclamped, u16_512_255), u16_512_0);
                __m128i p1_r_yvec1 = _mm_and_si128(p1_r_yvec1_clamped, u16_512_0xff);
                __m128i p1_r_yvec2 = _mm_and_si128(p1_r_yvec2_clamped, u16_512_0xff);
                __m128i p1_g_yvec1 = _mm_and_si128(p1_g_yvec1_clamped, u16_512_0xff);
                __m128i p1_g_yvec2 = _mm_and_si128(p1_g_yvec2_clamped, u16_512_0xff);
                __m128i p1_b_yvec1 = _mm_and_si128(p1_b_yvec1_clamped, u16_512_0xff);
                __m128i p1_b_yvec2 = _mm_and_si128(p1_b_yvec2_clamped, u16_512_0xff);

                // YUV -> ARGB
                __m128i p2_y_vec_2_mul_42 = _mm_mullo_epi16(u16_512_42, p2_y_vec_2); // 42 * (y2)
                __m128i p2_y_vec_1_mul_42 = _mm_mullo_epi16(u16_512_42, p2_y_vec_1); // 42 * (y1)
                __m128i p2_v_vec_mul_153 = _mm_mullo_epi16(u16_512_153, p2_v_vec);   // 153 * (v)
                __m128i p2_u_vec_mul_100 = _mm_mullo_epi16(u16_512_100, p2_u_vec);   // 100 * (u)
                __m128i p2_v_vec_mul_47 = _mm_mullo_epi16(u16_512_47, p2_v_vec);     // 47 * (v)
                __m128i p2_u_vec_mul_4 = _mm_mullo_epi16(u16_512_4, p2_u_vec);       // 4 * (u)

                __m128i p2_r_yvec1_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(p2_y_vec_1_mul_42, u16_512_24224), p2_v_vec_mul_153),
                            8),
                        _mm_add_epi16(_mm_sub_epi16(p2_y_vec_1, u16_512_128), p2_v_vec));

                __m128i p2_r_yvec2_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(p2_y_vec_2_mul_42, u16_512_24224), p2_v_vec_mul_153),
                            8),
                        _mm_add_epi16(_mm_sub_epi16(p2_y_vec_2, u16_512_128), p2_v_vec));

                __m128i p2_g_yvec1_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(p2_y_vec_1_mul_42, p2_u_vec_mul_100),
                                _mm_add_epi16(u16_512_2016, p2_v_vec_mul_47)),
                            8),
                        _mm_sub_epi16(_mm_add_epi16(p2_y_vec_1, u16_512_128), p2_v_vec));

                __m128i p2_g_yvec2_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(p2_y_vec_2_mul_42, p2_u_vec_mul_100),
                                _mm_add_epi16(u16_512_2016, p2_v_vec_mul_47)),
                            8),
                        _mm_sub_epi16(_mm_add_epi16(p2_y_vec_2, u16_512_128), p2_v_vec));

                __m128i p2_b_yvec1_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_sub_epi16(
                                _mm_add_epi16(
                                    p2_y_vec_1_mul_42,
                                    p2_u_vec_mul_4),
                                u16_512_5152),
                            8),
                        _mm_add_epi16(
                            _mm_sub_epi16(p2_y_vec_1, u16_512_256),
                            _mm_mullo_epi16(u16_512_2, p2_u_vec)));

                __m128i p2_b_yvec2_unclamped =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_sub_epi16(
                                _mm_add_epi16(
                                    p2_y_vec_2_mul_42,
                                    p2_u_vec_mul_4),
                                u16_512_5152),
                            8),
                        _mm_add_epi16(
                            _mm_sub_epi16(p2_y_vec_2, u16_512_256),
                            _mm_mullo_epi16(u16_512_2, p2_u_vec)));

                __m128i p2_r_yvec1_clamped = _mm_max_epi16(_mm_min_epi16(p2_r_yvec1_unclamped, u16_512_255), u16_512_0);
                __m128i p2_r_yvec2_clamped = _mm_max_epi16(_mm_min_epi16(p2_r_yvec2_unclamped, u16_512_255), u16_512_0);
                __m128i p2_g_yvec1_clamped = _mm_max_epi16(_mm_min_epi16(p2_g_yvec1_unclamped, u16_512_255), u16_512_0);
                __m128i p2_g_yvec2_clamped = _mm_max_epi16(_mm_min_epi16(p2_g_yvec2_unclamped, u16_512_255), u16_512_0);
                __m128i p2_b_yvec1_clamped = _mm_max_epi16(_mm_min_epi16(p2_b_yvec1_unclamped, u16_512_255), u16_512_0);
                __m128i p2_b_yvec2_clamped = _mm_max_epi16(_mm_min_epi16(p2_b_yvec2_unclamped, u16_512_255), u16_512_0);
                __m128i p2_r_yvec1 = _mm_and_si128(p2_r_yvec1_clamped, u16_512_0xff);
                __m128i p2_r_yvec2 = _mm_and_si128(p2_r_yvec2_clamped, u16_512_0xff);
                __m128i p2_g_yvec1 = _mm_and_si128(p2_g_yvec1_clamped, u16_512_0xff);
                __m128i p2_g_yvec2 = _mm_and_si128(p2_g_yvec2_clamped, u16_512_0xff);
                __m128i p2_b_yvec1 = _mm_and_si128(p2_b_yvec1_clamped, u16_512_0xff);
                __m128i p2_b_yvec2 = _mm_and_si128(p2_b_yvec2_clamped, u16_512_0xff);

                // #define MIX(a, x1, x2) ((((a) * ((x1) - (x2))) >> 8) + x2)
                // image overlap
                // #define MIX(a, x1, x2) (((((a) * (x1)) + ((256 - (a)) * (x2))) + 128) >> 8)
                // Max = 256 * 255 + 128 = 65408 < 65535
                __m128i r_prime_yvec1 =
                    _mm_srli_epi16(
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mullo_epi16(alpha_vec, p1_r_yvec1),
                                _mm_mullo_epi16(
                                    alpha_256_minus_vec,
                                    p2_r_yvec1)),
                            u16_512_128),
                        8);
                __m128i g_prime_yvec1 =
                    _mm_srli_epi16(
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mullo_epi16(alpha_vec, p1_g_yvec1),
                                _mm_mullo_epi16(
                                    alpha_256_minus_vec,
                                    p2_g_yvec1)),
                            u16_512_128),
                        8);
                __m128i b_prime_yvec1 =
                    _mm_srli_epi16(
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mullo_epi16(alpha_vec, p1_b_yvec1),
                                _mm_mullo_epi16(
                                    alpha_256_minus_vec,
                                    p2_b_yvec1)),
                            u16_512_128),
                        8);

                __m128i r_prime_yvec2 =
                    _mm_srli_epi16(
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mullo_epi16(alpha_vec, p1_r_yvec2),
                                _mm_mullo_epi16(
                                    alpha_256_minus_vec,
                                    p2_r_yvec2)),
                            u16_512_128),
                        8);
                __m128i g_prime_yvec2 =
                    _mm_srli_epi16(
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mullo_epi16(alpha_vec, p1_g_yvec2),
                                _mm_mullo_epi16(
                                    alpha_256_minus_vec,
                                    p2_g_yvec2)),
                            u16_512_128),
                        8);
                __m128i b_prime_yvec2 =
                    _mm_srli_epi16(
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mullo_epi16(alpha_vec, p1_b_yvec2),
                                _mm_mullo_epi16(
                                    alpha_256_minus_vec,
                                    p2_b_yvec2)),
                            u16_512_128),
                        8);

                // ARGB -> YUV
                __m128i y_prime_yvec1 =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_add_epi16(
                                    _mm_sub_epi16(_mm_mullo_epi16(u16_512_66, r_prime_yvec1), u16_512_8192),
                                    _mm_sub_epi16(_mm_mullo_epi16(u16_512_129, g_prime_yvec1), u16_512_16384)),
                                _mm_mullo_epi16(u16_512_25, b_prime_yvec1)),
                            8),
                        u16_512_112);

                __m128i y_prime_yvec2 =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_add_epi16(
                                    _mm_sub_epi16(_mm_mullo_epi16(u16_512_66, r_prime_yvec2), u16_512_8192),
                                    _mm_sub_epi16(_mm_mullo_epi16(u16_512_129, g_prime_yvec2), u16_512_16384)),
                                _mm_mullo_epi16(u16_512_25, b_prime_yvec2)),
                            8),
                        u16_512_112);

                __m128i u_prime_yvec1 =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(
                                    _mm_mullo_epi16(u16_512_112, b_prime_yvec1),
                                    _mm_mullo_epi16(u16_512_38, r_prime_yvec1)),
                                _mm_sub_epi16(
                                    u16_512_128,
                                    _mm_mullo_epi16(u16_512_74, g_prime_yvec1))),
                            8),
                        u16_512_128);

                __m128i v_prime_yvec1 =
                    _mm_add_epi16(
                        _mm_srai_epi16(
                            _mm_add_epi16(
                                _mm_sub_epi16(
                                    _mm_mullo_epi16(u16_512_112, r_prime_yvec1),
                                    _mm_mullo_epi16(u16_512_94, g_prime_yvec1)),
                                _mm_sub_epi16(
                                    u16_512_128,
                                    _mm_mullo_epi16(u16_512_18, b_prime_yvec1))),
                            8),
                        u16_512_128);

                __int64_t y_prime_yvec1_packed = _mm_cvtsi128_si64(_mm_packus_epi16(y_prime_yvec1, _mm_setzero_si128()));
                __int64_t y_prime_yvec2_packed = _mm_cvtsi128_si64(_mm_packus_epi16(y_prime_yvec2, _mm_setzero_si128()));

                // XXYYZZWW
                __m128i u_prime_yvec1_shuffled = _mm_shufflehi_epi16(_mm_shufflelo_epi16(u_prime_yvec1, imm8), imm8);
                __m128i v_prime_yvec1_shuffled = _mm_shufflehi_epi16(_mm_shufflelo_epi16(v_prime_yvec1, imm8), imm8);
                // XYXYZWZW
                __m128i u_prime_yvec1_permuted = _mm_shuffle_epi32(u_prime_yvec1_shuffled, imm8);
                __m128i v_prime_yvec1_permuted = _mm_shuffle_epi32(v_prime_yvec1_shuffled, imm8);
                // XYZWXYZW
                __m128i u_prime_yvec1_repeated = _mm_packus_epi16(u_prime_yvec1_permuted, _mm_setzero_si128());
                __m128i v_prime_yvec1_repeated = _mm_packus_epi16(v_prime_yvec1_permuted, _mm_setzero_si128());

                int u_prime_yvec1_packed = _mm_cvtsi128_si32(u_prime_yvec1_repeated);
                int v_prime_yvec1_packed = _mm_cvtsi128_si32(v_prime_yvec1_repeated);

                // Store value
                *(__int64_t *)((uint8_t *)y_result + image_idx * Y_SIZE + y_index_1) = y_prime_yvec1_packed;
                *(__int64_t *)((uint8_t *)y_result + image_idx * Y_SIZE + y_index_2) = y_prime_yvec2_packed;
                *(int *)((uint8_t *)u_result + image_idx * U_SIZE + uv_index) = u_prime_yvec1_packed;
                *(int *)((uint8_t *)v_result + image_idx * V_SIZE + uv_index) = v_prime_yvec1_packed;
            }
        }
    }
}

#undef VECTOR_SIZE