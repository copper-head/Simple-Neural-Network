#include "matrix.h"

Matrix::Matrix(const int num_rows, const int num_columns, const float default_value) {

    this->NUM_ROWS = num_rows;
    this->NUM_COLS = num_columns;

    for (int i = 0; i < num_rows; i++) {

        std::vector<float> tempVector;
        for (int j = 0; j < num_columns; j++) {
            tempVector.push_back(default_value);
        }
        this->matrix.push_back(tempVector);
    }
}

float Matrix::getValue(const int row, const int col) {
    return this->matrix[row][col];
}

void Matrix::setValue(const int row, const int col, const float value) {
    this->matrix[row][col] = value;
}

int Matrix::numCols() const {
    return this->NUM_COLS;
}

int Matrix::numRows() const {
    return this->NUM_ROWS;
}

void Matrix::addToValue(const int row, const int col, const float value) {
    this->matrix[row][col] += value;
}