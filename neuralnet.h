
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
        void inputData(std::vector<float> data);
        void propogate();

        // For printing things
        void printOutput();
        void printLayer(int layerIndex);
        void printWeightChange();
        void printWeights();

        void setTrainingMode(bool mode);
        void randomizeWeights();

        void propogateDeltas();
        float trainExample(float inputData[], float outputData[]);
        float trainExample(std::vector<float> inputData, std::vector<float> outputData);
        void setDesiredOutput(float outputData[]);
        void setDesiredOutput(std::vector<float> outputData);

        // This function computes the cost change with respect to activation,
        // which is needed in order to compute the cost gradient with respect to weights.
        // CRITICALLY IMPORTANT RECURSIVE ALGORITHM
        float computeCostChangeOfActivation(const int layer, const int index);


        void averageOutWeightChanges();

        // changes weights based on current delta weight aka the gradient.
        void applyDeltaWeights();

        // Train Model from data
        void trainModel(std::vector<std::vector<float>> inputData, std::vector<std::vector<float>> outputData, const int numBatches, const int iterations);



        int currentTrainingExampleCount;


    private:

        bool trainingMode = false;

        int inputLength;    // Length of input layer (layer in index 0 of layers)
        int layerCount;     // Number of layers in the network

        float averageCost;


        std::vector<LayerVector> layers;
        std::vector<Matrix> weightMatices;

        // ------ Vectors and stuff needed for training ----- //
        std::vector<Matrix> deltaWeightMatrices;
        std::vector<LayerVector> deltaLayers;
        LayerVector desiredOutput = LayerVector(0, 0);

        // ------ PRIVATE METHODS ------ //

};