#include "matrix.h"
#include "layervector.h"
#include <iostream>

int main() {

    Matrix myMatrix(32, 3001, 1);

    LayerVector inputLayer(3000, 1);
    LayerVector nextLayer(32, 0);

    nextLayer.computeValues(myMatrix, inputLayer);

    std::cout << nextLayer.getValue(0);

}