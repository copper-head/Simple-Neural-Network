#include "neuralnet.h"

NeuralNet::NeuralNet(int layerSizes[], const int layerSizesLen) {

    this->inputLength = layerSizes[0];
    this->layerCount = layerSizesLen;

    // Create Layers
    for (int i=0;i<layerSizesLen;i++) {
        this->layers.push_back(LayerVector(layerSizes[i], 0));
    }

    // Create Weight matrices based on layers
    for (int i=1;i<layerSizesLen;i++) {
        this->weightMatices.push_back(Matrix(layerSizes[i], layerSizes[i-1] + 1,0));
    }
}

void NeuralNet::inputData(int data[]) {
    for (int i=0; i<this->inputLength; i++) {
        this->layers[0].setValue(i, data[i]);
    }
}

void NeuralNet::propogate() {
    for (int i=1;i<this->layerCount;i++) {
        this->layers[i].computeValues(this->weightMatices[i-1], this->layers[i-1]);
    }
}