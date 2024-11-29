#include <iostream>
#include <fstream>
#include <filesystem>
#include <malloc.h>
#include <cstdint>
#include <cstdio>

#define WIDTH 1920
#define HEIGHT 1080
#define Y_SIZE WIDTH *HEIGHT
#define U_SIZE (Y_SIZE / 4)
#define V_SIZE (Y_SIZE / 4)

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    fs::path file_path = argv[1];
    uint8_t *y_data = (uint8_t *)malloc(Y_SIZE);
    uint8_t *u_data = (uint8_t *)malloc(U_SIZE);
    uint8_t *v_data = (uint8_t *)malloc(V_SIZE);

    std::ifstream file(file_path, std::ios::binary);
    FILE *output_file = fopen(argv[2], "w");

    file.read((char *)y_data, Y_SIZE);
    file.read((char *)u_data, U_SIZE);
    file.read((char *)v_data, V_SIZE);

    for (int i = 0; i < HEIGHT; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            uint8_t b = *((uint8_t *)y_data + i * WIDTH + j);
            fprintf(output_file, "%02X ", b);
        }
        fprintf(output_file, "\n");
    }

    fprintf(output_file, "\n");

    for (int i = 0; i < WIDTH; i++)
    {
        uint8_t b = *((uint8_t *)u_data + i);
        fprintf(output_file, "%02X ", b);
    }

    fprintf(output_file, "\n");

    for (int i = 0; i < WIDTH; i++)
    {
        uint8_t b = *((uint8_t *)v_data + i);
        fprintf(output_file, "%02X ", b);
    }

    return EXIT_SUCCESS;
}