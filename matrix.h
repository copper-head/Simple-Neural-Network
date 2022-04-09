// Filename: matrix.cpp
//
// Contains declerations for matrix

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

class Matrix {

    /*
        Description: This class is used to store weights for the neural network.
    */

    public:



        // ------ CONSTRUCTOR ----- //
        Matrix(const int num_rows, const int num_columns, const int default_value);

        // ------ PUBLIC METHODS ------ //
        float getValue(const int row, const int col);
        void setValue(const int row, const int col, const float value);

        int numCols() const;
        int numRows() const;

    private:
        
        // ------ PRIVATE ATTRIBUTES ------ //
        std::vector<std::vector<float>> matrix;

        // Matrix Dimensions
        int NUM_COLS;
        int NUM_ROWS;

};


#endif