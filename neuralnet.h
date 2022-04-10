
#include <vector>
#include <iostream>
#include "matrix.h"
#include "layervector.h"

class NeuralNet {

    public:

        // ------ CONSTRUCTOR ------ //
        NeuralNet(int layerSizes[], const int layerSizesLen);

        // ------ PUBLIC METHODS ------ //
        void inputData(float data[]);
        void propogate();

        // For printing things
        void printOutput();
        void printLayer(int layerIndex);
        void printWeightChange();

        void setTrainingMode(bool mode);
        void randomizeWeights();

        void propogateDeltas();
        void trainExample(float inputData[], float outputData[]);
        void setDesiredOutput(float outputData[]);

        // This function computes the cost change with respect to activation,
        // which is needed in order to compute the cost gradient with respect to weights.
        float computeCostChangeOfActivation(int &layer, int &index);
        void averageOutWeightChanges();
        int currentTrainingExampleCount;


    private:

        bool trainingMode = false;

        int inputLength;    // Length of input layer (layer in index 0 of layers)
        int layerCount;     // Number of layers in the network


        std::vector<LayerVector> layers;
        std::vector<Matrix> weightMatices;

        // ------ Vectors and stuff needed for training ----- //
        std::vector<Matrix> deltaWeightMatrices;
        std::vector<LayerVector> deltaLayers;
        LayerVector desiredOutput = LayerVector(0, 0);

        // ------ PRIVATE METHODS ------ //

};