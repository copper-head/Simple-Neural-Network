#include "testdata.h"
#include <iostream>

void generateData(float inputData[][8], float outputData[][2], const int dataSize) {

    int sumFirst;
    int sumSecond;

    for (int i = 0; i < dataSize; i++) {

        for (int j = 0; j < 8; j++) {
            inputData[i][j] = float(rand() % 1000) / 1000;
        }

        sumFirst = 0;
        for (int j = 0; j < 4; j++) {
            sumFirst += inputData[i][j];
        }

        sumSecond = 0;
        for (int j = 4; j < 8; j++) {
            sumSecond += inputData[i][j];
        }

        if (sumFirst >= sumSecond) {
            outputData[i][0] = 1;
            outputData[i][1] = 0;
        } else {
            outputData[i][0] = 0;
            outputData[i][1] = 1;
        }

    }

}

void generateData(std::vector<std::vector<float>> &inputData, std::vector<std::vector<float>> &outputData, const int dataSize) {
    int sumFirst;
    int sumSecond;

    //std::cout << "works";

    std::vector<float> tempInput;
    std::vector<float> tempOutput;


    for (int j = 0; j < 8; j++) {
        tempInput.push_back(0);
    }

    for (int j = 0; j < 2; j++) {
        tempOutput.push_back(0);
    }

    for (int i = 0; i < dataSize; i++) {

        for (int j = 0; j < 8; j++) {
            tempInput[i] = float(rand() % 1000) / 1000;
        }

        inputData.push_back(tempInput);
    }


    for (int i = 0; i < dataSize; i++) {

        sumFirst = 0;
        for (int j = 0; j < 4; j++) {
            sumFirst += inputData[i][j];
        }

        sumSecond = 0;
        for (int j = 4; j < 8; j++) {
            sumSecond += inputData[i][j];
        }

        if (sumFirst >= sumSecond) {
            tempOutput[0] = 1;
            tempOutput[1] = 0;
        } else {
            tempOutput[0] = 0;
            tempOutput[1] = 1;
        }
        
        outputData.push_back(tempOutput);
    }

    return;
}