#pragma once

// STL
#include <cstddef>
#include <vector>
#include <iostream>

// CUPYBARA

// TODO: this only works for integer numbers currently

class scan {
private:
    // ARGUMENTS //

    // pointer to the data on the HOST
    int* data;

    // length of the input array which we are attempting to scan
    std::size_t len;

    // whether the scan is inclusive or exclusive
    bool is_inclusive;

    // ARGUMENTS //

    // GPU DATA //

    int* device_input;

    int* device_output;

    // GPU DATA //

    // HOST DATA //

    std::vector<int> output;

    // HOST DATA //


public:

    // initializes the scan object and enqueues all of the operations to perform the scan
    scan(int* _data, std::size_t _len, bool _is_inclusive = false);

    void print();

    ~scan() = default;
};
