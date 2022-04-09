#include "layervector.h"

LayerVector::LayerVector(const int size, const float defaultValue) {

    this->length = size;

    // Create a vector of the specified size.
    for (int i = 0; i < size; i++) {
        this->layerVector.push_back(defaultValue);
    }
}

float LayerVector::getValue(const int index) const {
    return this->layerVector[index];
}

void LayerVector::setValue(const int index, const float value) {
    this->layerVector[index] = value;
}

void LayerVector::clear() {
    std::fill(this->layerVector.begin(), this->layerVector.end(), 0);
}

void LayerVector::computeValues(Matrix &weights, LayerVector &prevLayer) {

    // Check to make sure Matrix product between weights and prevLayer is valid
    if (weights.numCols() - 1 != prevLayer.length) {
        return;
    }

    // Check to make sure that the product will have the same dimensions as this Vector
    if (weights.numRows() != this->length) {
        return;
    }
    
    for (int i = 0; i < this->length; i++) {
        this->layerVector[i] = 0;

        for (int j = 0; j < weights.numCols() - 1; j++) {
            this->layerVector[i] += weights.getValue(i, j) * prevLayer.getValue(j);
        }
        
        // last value in the weights matrix is the bias.
        this->layerVector[i] += weights.getValue(i, weights.numCols()-1);

        // Finally squish the value down to (0,1) with fast sigmoid function 
        this->layerVector[i] = 0.5 * (this->layerVector[i] / (1.0 + abs(this->layerVector[i]))) + 0.5;
    }

}