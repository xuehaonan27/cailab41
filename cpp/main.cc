#include <iostream>
#include <filesystem>
#include <fstream>
#include <ctime>
#include <malloc.h>

#include "plain.hh"
#include "mmx.hh"
#include "sse2.hh"
#include "avx.hh"

#include "misc.hh"
namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <yuv_path> <save_dir>" << std::endl;
        return EXIT_FAILURE;
    }
    fs::path yuv_path = argv[1];
    fs::path save_dir = argv[2];

    std::cout << "YUV file: " << yuv_path << std::endl;
    std::cout << "Save to directory: " << save_dir << std::endl;

    if (!fs::exists(save_dir))
    {
        fs::create_directories(save_dir);
    }

    std::ifstream yuv_file(yuv_path, std::ios::binary);
    if (!yuv_file)
    {
        std::cerr << "Cannot open " << yuv_path << std::endl;
        return EXIT_FAILURE;
    }

    /* Picture read */

    unsigned char *y_data = (unsigned char *)malloc(Y_SIZE);
    unsigned char *u_data = (unsigned char *)malloc(U_SIZE);
    unsigned char *v_data = (unsigned char *)malloc(V_SIZE);

    // Read fields
    yuv_file.read(reinterpret_cast<char *>(y_data), Y_SIZE);
    yuv_file.read(reinterpret_cast<char *>(u_data), U_SIZE);
    yuv_file.read(reinterpret_cast<char *>(v_data), V_SIZE);
    yuv_file.close();

    unsigned char **y_result = (unsigned char **)malloc(84 * Y_SIZE);
    unsigned char **u_result = (unsigned char **)malloc(84 * U_SIZE);
    unsigned char **v_result = (unsigned char **)malloc(84 * V_SIZE);

    clock_t start_time = clock();
    solve_plain(y_data, u_data, v_data, y_result, u_result, v_result);
    clock_t end_time = clock();

    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Elapsed Time: " << elapsed_time << " seconds" << std::endl;

    for (int i = 0; i < 84; i++)
    {
        fs::path output_path = save_dir / (std::to_string(i) + ".yuv");
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>((char *)y_result + i * Y_SIZE), Y_SIZE);
        output_file.write(reinterpret_cast<char *>((char *)u_result + i * U_SIZE), U_SIZE);
        output_file.write(reinterpret_cast<char *>((char *)v_result + i * V_SIZE), V_SIZE);
        output_file.close();
    }

    std::cout << "Files saved" << std::endl;

    free(y_data);
    free(u_data);
    free(v_data);
    free(y_result);
    free(u_result);
    free(v_result);

    return EXIT_SUCCESS;
}
