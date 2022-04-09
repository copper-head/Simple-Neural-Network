
#include <vector>
#include "matrix.h"
#include "layervector.h"

class NeuralNet {

    public:

        // ------ CONSTRUCTOR ------ //
        NeuralNet(int layerSizes[], const int layerSizesLen);

        // ------ PUBLIC METHODS ------ //
        void inputData(int data[]);


    private:

        int inputLength;    // Length of input layer (layer in index 0 of layers)
        int layerCount;     // Number of layers in the network

        std::vector<LayerVector> layers;
        std::vector<Matrix> weightMatices;

        // ------ PRIVATE METHODS ------ //

        void propogate();

};