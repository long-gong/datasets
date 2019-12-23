#include "BinaryFile.h"

#include <iostream>
#include <string>

int main() {
    std::string fn = "datasets/audio/audio.data";
    BinaryFile bf(fn, BinaryFile::Mode::Read);

    std::cout << bf.size() << std::endl;
}