#ifndef VECTORLAYER_H
#define VECTORLAYER_H

#include <vector>
#include <iostream>
#include <cmath>
#include "matrix.h"

class LayerVector {

    public:

        // PUBLIC ATTRIBUTES //
        int length;


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
        void clear();

        //
        //
        //
        void computeValues(Matrix &weights, LayerVector &prevLayer);


    private:

        std::vector<float> layerVector;

};


#endif