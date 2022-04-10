#include "neuralnet.h"
#include "testdata.h"
#include <iostream>
#include <stdlib.h>

int main() {

    srand(123);

    int layerSizes[] = {8, 6, 4, 2};

    NeuralNet testNetwork(layerSizes, 4);
    testNetwork.randomizeWeights();

    const int dataSize = 1000;
    float inputData[dataSize][8];
    float outputData[dataSize][2];

    generateData(inputData, outputData, dataSize);

    testNetwork.printWeightChange();
    testNetwork.setTrainingMode(true);

    for (int i = 0; i < dataSize; i ++)
        testNetwork.trainExample(inputData[i], outputData[i]);

    testNetwork.averageOutWeightChanges();
    
    testNetwork.printWeightChange();
    std::cout << std::endl << testNetwork.currentTrainingExampleCount;

}