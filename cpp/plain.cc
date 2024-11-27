#include <cmath>
#include <cstdio>

#include "misc.hh"

void solve_plain(
    const unsigned char *y_data,
    const unsigned char *u_data,
    const unsigned char *v_data,
    unsigned char **y_result,
    unsigned char **u_result,
    unsigned char **v_result)
{
    for (int image_idx = 0; image_idx < 84; image_idx++)
    {
        unsigned char alpha = 1 + image_idx * 3;
        float alpha_percentage = float(alpha) / 256;
        for (int j = 0; j < HEIGHT; j++)
        {
            for (int i = 0; i < WIDTH; i++)
            {
                size_t y_index = j * WIDTH + i;
                size_t uv_index = size_t(j / 2) * size_t(WIDTH / 2) + size_t(i / 2); // Cannot be changed

                unsigned char y = y_data[y_index];
                unsigned char u = u_data[uv_index];
                unsigned char v = v_data[uv_index];

                // YUV to ARGB
                float r = y + 1.140 * v;
                float g = y - 0.394 * u - 0.581 * v;
                float b = y + 2.032 * u;

                // Alpha mix
                float r2 = r * alpha_percentage;
                float g2 = g * alpha_percentage;
                float b2 = b * alpha_percentage;

                // RGB to YUV
                float y2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2;
                float u2 = -0.147 * r2 - 0.289 * g2 + 0.436 * b2;
                float v2 = 0.615 * r2 - 0.515 * g2 - 0.100 * b2;

                *(unsigned char *)((char *)y_result + image_idx * Y_SIZE + y_index) = (unsigned char)y2;
                *(unsigned char *)((char *)u_result + image_idx * U_SIZE + uv_index) = (unsigned char)u2;
                *(unsigned char *)((char *)v_result + image_idx * V_SIZE + uv_index) = (unsigned char)v2;
            }
        }
    }
}

void solve_plain_int(
    const unsigned char *y_data,
    const unsigned char *u_data,
    const unsigned char *v_data,
    unsigned char **y_result,
    unsigned char **u_result,
    unsigned char **v_result)
{
    for (int image_idx = 0; image_idx < 84; image_idx++)
    {
        unsigned char alpha = 1 + image_idx * 3;
        for (int j = 0; j < HEIGHT; j++)
        {
            for (int i = 0; i < WIDTH; i++)
            {
                size_t y_index = j * WIDTH + i;
                size_t uv_index = size_t(j / 2) * size_t(WIDTH / 2) + size_t(i / 2); // Cannot be changed

                unsigned char y = y_data[y_index];
                unsigned char u = u_data[uv_index];
                unsigned char v = v_data[uv_index];

                // YUV to ARGB
                unsigned char r = y + 1.140 * v;
                unsigned char g = y - 0.394 * u - 0.581 * v;
                unsigned char b = y + 2.032 * u;

                // Alpha mix
                unsigned char r2 = size_t(alpha) * r >> 8;
                unsigned char g2 = size_t(alpha) * g >> 8;
                unsigned char b2 = size_t(alpha) * b >> 8;

                unsigned char y2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2;
                unsigned char u2 = -0.147 * r2 - 0.289 * g2 + 0.436 * b2;
                unsigned char v2 = 0.615 * r2 - 0.515 * g2 - 0.100 * b2;

                *(unsigned char *)((char *)y_result + image_idx * Y_SIZE + y_index) = y2;
                *(unsigned char *)((char *)u_result + image_idx * U_SIZE + uv_index) = u2;
                *(unsigned char *)((char *)v_result + image_idx * V_SIZE + uv_index) = v2;
            }
        }
    }
}
