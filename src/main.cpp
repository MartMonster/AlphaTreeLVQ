#include <PNGcodec.hpp>
#include <defines.hpp>

#include <AlphaTree.hpp>
#include <AlphaTreeConfig.hpp>
#include <RandGenImage.hpp>

// args: Filename, nchannels, numthreads, testimgsize, algorithmcode, bitdepth, tseflag
int main(int argc, char **argv) {
    // srand(time(NULL));

    const auto configFileName = argc < 2 ? "config.txt" : std::string(argv[1]);
    alphatreeConfig.initialize(configFileName);
    auto config = alphatreeConfig.load(argc, argv);

    if (config.has_value() == false) {
        std::cerr << "Unable to open configuration file." << std::endl;
        return -1;
    }

    AlphaTreeConfig::AlphaTreeParameters params = config.value();

    const std::string filePath = params.imageFileName;

    auto [image, w, h, ch] = PNGCodec::imread(params.UseRandomlyGeneratedImages ? "RAND" : filePath);

    const bool reduceImageBitdepth = true;

    const auto &width = params.UseRandomlyGeneratedImages ? params.randomGenImageWidth : w;
    const auto &height = params.UseRandomlyGeneratedImages ? params.randomGenImageHeight : h;
    const auto &bitdepth = params.UseRandomlyGeneratedImages ? params.bitdepth : 16 * ch;
    const auto &nch = params.UseRandomlyGeneratedImages ? params.nchannels : ch;
    const auto &dMetric = params.dissimilarityMetric;
    const auto &conn = params.connectivity;
    const auto &algCode = params.alphaTreeAlgorithmCode;
    const auto &nthr = params.numthreads;
    const auto &nitr = params.numitr;
    const auto &tse = params.tse;
    const auto &alphaFilter = params.alphaFilter;
    const auto &iparam1 = params.iparam1;
    const auto &fparam1 = params.fparam1;
    const auto &fparam2 = params.fparam2;

    if (!params.UseRandomlyGeneratedImages)
        std::cout << "Image file name: " << params.imageFileName << std::endl;
    std::cout << "=======================================================================" << std::endl;
    std::cout << "========== imgsize = " << height << " x " << width << " (" << bitdepth << " bits, " << nch << " ch, "
              << params.connectivity << "N) ================" << std::endl;
    std::cout << "=======================================================================" << std::endl;
    std::cout << "-----------------------------------------------------------------------------------" << std::endl;
    std::cout << algCode << " Running " << alphatreeConfig.getAlphaTreeAlgorithmName(algCode) << " (" << nthr
              << " threads)" << std::endl;
    std::cout << "-----------------------------------------------------------------------------------" << std::endl;
    std::vector<double> runtimes;

    for (int itr = 0; itr < nitr; itr++) {
        double tStart = 0, tEnd = INFINITY;
        if (params.UseRandomlyGeneratedImages == true) {
            if (bitdepth > 32 && bitdepth <= 64) {
                uint64_t *image = (uint64_t *)Malloc(width * height * sizeof(uint64_t));
                RandGenImage::randomize64(image, width, height, bitdepth);
                AlphaTree<uint64_t> tree;
                tStart = get_wall_time();
                tree.BuildAlphaTree(image, height, width, nch, dMetric, conn, algCode, nthr, tse, fparam1, fparam2,
                                    iparam1);
                tEnd = get_wall_time();
                Free(image);
            } else if (bitdepth > 16) {
                uint32_t *image = (uint32_t *)Malloc(width * height * sizeof(uint32_t));
                RandGenImage::randomize32(image, width, height, bitdepth);
                AlphaTree<uint32_t> tree;
                tStart = get_wall_time();
                tree.BuildAlphaTree(image, height, width, nch, dMetric, conn, algCode, nthr, tse, fparam1, fparam2,
                                    iparam1);
                tEnd = get_wall_time();
                Free(image);
            } else if (bitdepth > 8) {
                uint16_t *image = (uint16_t *)Malloc(width * height * nch * sizeof(uint16_t));
                RandGenImage::randomize16(image, width, height, bitdepth, nch);
                AlphaTree<uint16_t> tree;
                tStart = get_wall_time();
                tree.BuildAlphaTree(image, height, width, nch, dMetric, conn, algCode, nthr, tse, fparam1, fparam2,
                                    iparam1);
                tEnd = get_wall_time();
                Free(image);
            } else if (bitdepth > 0) {
                uint8_t *image = (uint8_t *)Malloc(width * height * nch * sizeof(uint8_t));
                RandGenImage::randomize8(image, width, height, bitdepth, nch);
                AlphaTree<uint8_t> tree;
                tStart = get_wall_time();
                tree.BuildAlphaTree(image, height, width, nch, dMetric, conn, algCode, nthr, tse, fparam1, fparam2,
                                    iparam1);
                tEnd = get_wall_time();
                Free(image);
            } else {
                std::cerr << "Invalid bit-depth. " << std::endl;
                return -1;
            }

        } else {

            if (reduceImageBitdepth) {
                const size_t shamt = 16 - params.bitdepth;
                uint16_t pMin = 65535;
                uint16_t pMax = 0;

                for (auto &pixel : image) {
                    pixel = pixel >> shamt;
                    // std::cerr << pixel << std::endl;
                    pMax = std::max(pMax, pixel);
                    pMin = std::min(pMin, pixel);
                }

                std::cout << "reduceImageBitdepth - BitDepth = params.bitdepth = " << params.bitdepth
                          << " / DR = " << pMin << " - " << pMax << std::endl;
            }

            uint16_t maxVal = *std::max_element(image.begin(), image.end());
            if ((int)maxVal > (int)255) {
                AlphaTree<uint16_t> tree;
                tStart = get_wall_time();
                tree.BuildAlphaTree(image.data(), height, width, nch, dMetric, conn, algCode, nthr, tse, fparam1,
                                    fparam2, iparam1);

                const bool rgbFilter = true;
                if (rgbFilter) {
                    tree.AlphaFilter(image.data(), alphaFilter);
                    // tree.AreaFilter(image.data(), 1);
                    std::ostringstream filename;
                    filename << "output/output.png";
                    std::cerr << "trying to write image to " << filename.str() << std::endl;
                    PNGCodec::imwrite(image, w, h, ch, filename.str());
                    std::cerr << "wrote image to " << filename.str() << std::endl;
                }

                tEnd = get_wall_time();
            } else {
                std::vector<uint8_t> image8(image.size());
                for (size_t i = 0; i < image.size(); i++)
                    image8[i] = (uint8_t)image[i];
                AlphaTree<uint8_t> tree;
                tStart = get_wall_time();
                tree.BuildAlphaTree(image8.data(), height, width, nch, dMetric, conn, algCode, nthr, tse, fparam1,
                                    fparam2, iparam1);
                for (size_t i = 0; i < image.size() && itr > 0; i++) {
                    if (image8[i] > 0)
                        std::cout << i << ": " << (uint16_t)image8[i] << std::endl;
                }
                tEnd = get_wall_time();

                // if (!itr) {
                //     tree.AlphaFilter(image8.data(), 170);
                //     PNGCodec::imwrite(image, w, h, ch, "out005.png");
                // }
            }
        }

        auto runtime = tEnd - tStart;
        std::cout << "-------------------Run " << itr + 1 << "/" << nitr << ": " << runtime << "------------------"
                  << std::endl;
        runtimes.push_back(runtime);
    }

    if (runtimes.empty() == false) {
        double minRuntime = *std::min_element(runtimes.begin(), runtimes.end());
        double imgsize = (double)(width * height);

        std::cout << "================== Summary ==================" << std::endl;
        std::cout << "Processing speed: " << (imgsize / minRuntime) * 1e-6 << "Mpix/s / Memory use "
                  << (double)max_memuse / imgsize << "B/pix" << std::endl;
        std::cout << "=============================================" << std::endl;
    }

    return 0;
}
