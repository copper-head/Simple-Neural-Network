#include "neuralnet.h"
#include <stdlib.h>

NeuralNet::NeuralNet(int layerSizes[], const int layerSizesLen) {

    this->inputLength = layerSizes[0];
    this->layerCount = layerSizesLen;
    this->currentTrainingExampleCount = 0;

    // Create Layers
    for (int i=0;i<layerSizesLen;i++) {
        this->layers.push_back(LayerVector(layerSizes[i], 0));
        this->deltaLayers.push_back(LayerVector(layerSizes[i], 0));
    }

    // Create Weight matrices based on layers
    for (int i=1;i<layerSizesLen;i++) {
        this->weightMatices.push_back(Matrix(layerSizes[i], layerSizes[i-1] + 1, 0));
        this->deltaWeightMatrices.push_back(Matrix(layerSizes[i], layerSizes[i-1] + 1, 0));
    }

    // Create output vector that will be used for computing cost function
    this->desiredOutput = LayerVector(this->layers[layerCount-1].length(), 0);

}

void NeuralNet::inputData(float data[]) {
    for (int i=0; i<this->inputLength; i++) {
        this->layers[0].setValue(i, data[i]);
    }
}

void NeuralNet::propogate() {
    for (int i=1;i<this->layerCount;i++) {
        this->layers[i].computeValues(this->weightMatices[i-1], this->layers[i-1]);
    }
}

void NeuralNet::propogateDeltas() {
    for (int i=1;i<this->layerCount;i++) {
        this->deltaLayers[i].computeDeltaValues(this->weightMatices[i-1], this->layers[i-1]);
    }
}

void NeuralNet::printOutput() {
    for (int i=0; i < this->layers[this->layerCount-1].length(); i++) {
        std::cout << this->layers[this->layerCount-1].getValue(i) << "  ";
    }
}

void NeuralNet::printLayer(int layerIndex) {
    for (int i=0; i < this->layers[layerIndex].length(); i++) {
        std::cout << this->layers[layerIndex].getValue(i) << "  ";
    }
}

void NeuralNet::randomizeWeights() {
    for (int i=0; i<this->layerCount-1;i++) {
        for (int m=0; m<this->weightMatices[i].numRows();m++) {
            for (int n=0; n<this->weightMatices[i].numCols();n++) {
                this->weightMatices[i].setValue(m, n, float(rand() % 2000 - 1000) / 1000.0);
            }
        }
    }
}

void NeuralNet::setTrainingMode(bool mode) {
    this->trainingMode = mode;
}

void NeuralNet::trainExample(float inputData[], float outputData[]) {

    if (!trainingMode) {
        std::cout << "Cannot train network on example -- training mode is not enabled." << std::endl;
        return;
    }

    this->currentTrainingExampleCount += 1;

    this->inputData(inputData);
    this->propogate();
    this->propogateDeltas();
    this->setDesiredOutput(outputData);

    float activationCost;
    float zFactor;
    float weightChange;
    int currenti;

    for (int i = 0; i < this->deltaWeightMatrices.size(); i++) {
        for (int m = 0; m < this->deltaWeightMatrices[i].numRows(); m++) {
            currenti = i + 1;
            zFactor = this->deltaLayers[i+1].getValue(m);
            activationCost = this->computeCostChangeOfActivation(currenti, m);
            for (int n = 0; n < this->deltaWeightMatrices[i].numCols(); n++) {
                weightChange = activationCost * zFactor * this->layers[i].getValue(n);
                this->deltaWeightMatrices[i].addToValue(m, n, weightChange);
            }
        }
    }

}

void NeuralNet::setDesiredOutput(float outputData[]) {

    for (int i = 0; i < this->desiredOutput.length(); i++) {
        this->desiredOutput.setValue(i, outputData[i]);
    }

}

float NeuralNet::computeCostChangeOfActivation(int &layer, int &index) {

    // Base Case -- If the output layer is reached, compute cost and return.
    if (layer == this->layerCount - 1) {
        return 2 * (this->layers[layer].getValue(index) - this->desiredOutput.getValue(index));
    }

    // Standard Case
    float deltaCost;
    int nextLayer = layer + 1;
    int currentIndex = index;
    for (int i = 0; i < this->layers[nextLayer].length(); i++) {
        deltaCost += this->weightMatices[layer].getValue(i, index) * this->deltaLayers[layer].getValue(i) * this->computeCostChangeOfActivation(nextLayer, i);
    }

    return deltaCost;

}

void NeuralNet::printWeightChange() {

    for (int i = 0; i < this->deltaWeightMatrices.size(); i++) {
        for (int m = 0; m < this->deltaWeightMatrices[i].numRows(); m++) {
            for (int n = 0; n < this->deltaWeightMatrices[i].numCols(); n++) {
                std::cout << deltaWeightMatrices[i].getValue(m, n) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void NeuralNet::averageOutWeightChanges() {

    for (int i = 0; i < this->deltaWeightMatrices.size(); i++) {
        for (int m = 0; m < this->deltaWeightMatrices[i].numRows(); m++) {
            for (int n = 0; n < this->deltaWeightMatrices[i].numCols(); n++) {
                this->deltaWeightMatrices[i].setValue(m, n, this->deltaWeightMatrices[i].getValue(m, n) / this->currentTrainingExampleCount);
            }
        }
    }
}