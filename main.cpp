#include "neuralnet.h"
#include "testdata.h"
#include <iostream>
#include <stdlib.h>

int main() {

    srand(123);

    int layerSizes[] = {8, 2};

    NeuralNet testNetwork(layerSizes, 2);
    testNetwork.randomizeWeights();

    const int dataSize = 1;
    std::vector<std::vector<float>> inputTrain;
    std::vector<std::vector<float>> outputTrain;

    const int testSize = 1;
    std::vector<std::vector<float>> inputTest;
    std::vector<std::vector<float>> outputTest; 

    generateData(inputTrain, outputTrain, dataSize);
    generateData(inputTest, outputTest, testSize);

    /*
    for (int i = 0; i < dataSize; i++) {
        inputTrain.push_back(inputTrain[i]);
        outputTrain.push_back(outputTrain[i]);
    }
    */

    testNetwork.printWeights();
    testNetwork.setTrainingMode(true);
    testNetwork.trainModel(inputTrain, outputTrain, 1, 100);

    //testNetwork.printWeightChange();
    testNetwork.printWeights();

    std::cout << std::endl << testNetwork.currentTrainingExampleCount;

}