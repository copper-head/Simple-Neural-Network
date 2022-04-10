#ifndef VECTORLAYER_H
#define VECTORLAYER_H

#include <vector>
#include <iostream>
#include <cmath>
#include "matrix.h"

class LayerVector {

    public:

        // PUBLIC ATTRIBUTES //


        LayerVector(const int size, const float defaultValue);

        //
        //
        //
        float getValue(const int index) const;

        //
        //
        //
        void setValue(const int index, const float value);

        //
        //
        //
        void addToValue(const int index, const float value);

        //
        //
        //
        void clear();

        //
        //
        //
        void computeValues(Matrix &weights, LayerVector &prevLayer);


        void computeDeltaValues(Matrix &weights, LayerVector &prevLayer);

        int length() const;

        std::vector<float> layerVector;

    private:

        int vectorLength;

};


#endif