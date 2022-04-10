#include "testdata.h"

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