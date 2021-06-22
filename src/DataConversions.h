#ifndef LAB0_DATACONVERSIONS_H
#define LAB0_DATACONVERSIONS_H

#include <vector>
#include "Math/float3.h"

template <typename T>
VecArray convertToVecArray(const std::vector<T>& v) {
    VecArray res;
    for (unsigned int i=0; i<v.size(); i+=3) {
        res.emplace_back(vec(v[i], v[i+1], v[i+2]));
    }
    return res;
}

template <typename T>
VecArray convertToVecArray(const T* data, unsigned int length) {
    VecArray res;
    for (unsigned int i=0; i<length; i+=3) {
        res.emplace_back(vec(*(data+i), *(data+i+1), *(data+i+2)));
    }
    return res;
}

#endif //LAB0_DATACONVERSIONS_H