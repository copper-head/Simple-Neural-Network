#include "neuralnet.h"
#include <stdlib.h>

NeuralNet::NeuralNet(int layerSizes[], const int layerSizesLen) {

    this->inputLength = layerSizes[0];
    this->layerCount = layerSizesLen;
    this->currentTrainingExampleCount = 0;
    this->averageCost = 0;

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

void NeuralNet::inputData(std::vector<float> data) {
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

float NeuralNet::trainExample(float inputData[], float outputData[]) {

    if (!trainingMode) {
        std::cout << "Cannot train network on example -- training mode is not enabled." << std::endl;
        return 0;
    }

    int exampleCost = 0;

    this->currentTrainingExampleCount += 1;

    this->inputData(inputData);
    this->propogate();
    this->propogateDeltas();
    this->setDesiredOutput(outputData);

    // Compute the cost of the example
    for (int i = 0; i < this->layers[layerCount-1].length(); i++) {
        exampleCost += (this->layers[layerCount-1].getValue(i) - outputData[i]) * (this->layers[layerCount-1].getValue(i) - outputData[i]);
    }

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

    return exampleCost;

}

float NeuralNet::trainExample(std::vector<float> inputData, std::vector<float> outputData) {

    if (!trainingMode) {
        std::cout << "Cannot train network on example -- training mode is not enabled." << std::endl;
        return 0;
    }

    float exampleCost = 0;

    this->currentTrainingExampleCount += 1;

    this->inputData(inputData);
    this->propogate();
    this->propogateDeltas();
    this->setDesiredOutput(outputData);

    // Compute the cost of the example
    for (int i = 0; i < this->layers[layerCount-1].length(); i++) {
        exampleCost += (this->layers[layerCount-1].getValue(i) - outputData[i]) * (this->layers[layerCount-1].getValue(i) - outputData[i]);
    }

    float activationCost;
    float zFactor;
    float weightChange;
    int currenti;

    for (int i = 0; i < this->deltaWeightMatrices.size(); i++) {

        // Compute Weight Gradient Components
        for (int m = 0; m < this->deltaWeightMatrices[i].numRows(); m++) {
            currenti = i + 1;
            zFactor = this->deltaLayers[i+1].getValue(m);
            activationCost = this->computeCostChangeOfActivation(currenti, m);
            for (int n = 0; n < this->deltaWeightMatrices[i].numCols()-1; n++) {
                weightChange = activationCost * zFactor * this->layers[i].getValue(n);
                this->deltaWeightMatrices[i].addToValue(m, n, weightChange);
            }

            // Compute Bias Gradient Components
            weightChange = activationCost * zFactor;
            this->deltaWeightMatrices[i].addToValue(m, this->deltaWeightMatrices[i].numCols()-1, weightChange);
        }
    }

    return exampleCost;

}

void NeuralNet::setDesiredOutput(float outputData[]) {

    for (int i = 0; i < this->desiredOutput.length(); i++) {
        this->desiredOutput.setValue(i, outputData[i]);
    }

}

void NeuralNet::setDesiredOutput(std::vector<float> outputData) {

    for (int i = 0; i < this->desiredOutput.length(); i++) {
        this->desiredOutput.setValue(i, outputData[i]);
    }

}

float NeuralNet::computeCostChangeOfActivation(const int layer, const int index) {

    // Base Case -- If the output layer is reached, compute cost and return.
    if (layer == this->layerCount - 1) {
        return 2 * (this->layers[layer].getValue(index) - this->desiredOutput.getValue(index));
    }

    // Standard Case
    float deltaCost = 0;
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

void NeuralNet::printWeights() {

    for (int i = 0; i < this->weightMatices.size(); i++) {
        for (int m = 0; m < this->weightMatices[i].numRows(); m++) {
            for (int n = 0; n < this->weightMatices[i].numCols(); n++) {
                std::cout << weightMatices[i].getValue(m, n) << " ";
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


void NeuralNet::trainModel(std::vector<std::vector<float>> inputData, std::vector<std::vector<float>> outputData, const int numBatches, const int iterations) {

    if (inputData.size() != outputData.size()) {
        std::cout << "Error training network from data -- inputData and outputData parameters are not of the same length." << std::endl;
    }

    float cost = 0;
    int currentBatchNum = 0;
    int elementsPerBatch = inputData.size() / numBatches;

    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < inputData.size(); j++) {       

            cost += this->trainExample(inputData[j], outputData[j]);

            /*
            currentBatchNum++;
            if (currentBatchNum > elementsPerBatch) {
                this->averageOutWeightChanges();
                currentBatchNum = 0;
                */
            //}
        }

        this->averageOutWeightChanges();
        this->applyDeltaWeights();
        this->currentTrainingExampleCount = 0;
        this->averageCost = cost;
        //std::cout << "Average Count: " << this->currentTrainingExampleCount << std::endl;
        std::cout << "Network Cost: " << this->averageCost << std::endl;  
    }

}

void NeuralNet::applyDeltaWeights() {

    // Iterates through deltaWeightMatrices, which is the gradient of the cost function.
    for (int i = 0; i < this->deltaWeightMatrices.size(); i++) {
        for (int m = 0; m < this->deltaWeightMatrices[i].numRows(); m++) {
            for (int n = 0; n < this->deltaWeightMatrices[i].numCols(); n++) {
                this->weightMatices[i].addToValue(m, n, -this->deltaWeightMatrices[i].getValue(m, n));
                this->deltaWeightMatrices[i].setValue(m, n, 0);
            }
        }
    }
}