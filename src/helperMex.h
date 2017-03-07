// =============================================================================
// == helperMEX.h
// == --------------------------------------------------------------------------
// == Functions to help in MEX interfaces
// == --------------------------------------------------------------------------
// == Copyright 2011. MIT. All Rights Reserved.
// == Written by Jason Chang 06-13-2011
// =============================================================================

#ifndef HELPERMEX
#define HELPERMEX
enum inputType {SCALAR, VECTOR, MATRIX, MATRIX3, NSCALAR, CELL, STRUCT};

#include <math.h>
#include <opencv2/opencv.hpp>
//#define DEBUG // define by compiler
// compile with "mex -O -DDEBUG ..." for debugging"
// compile with "mex -O ..." for debugging"

#include "array.h"

#ifdef DEBUG_ARR
#define arr(type)    array<type>

template <typename T>
array<T> allocate_memory(unsigned int length)
{
    return array<T>(length);
}
template <typename T>
array<T> allocate_memory(unsigned int length, T value)
{
    return array<T>(length, value);
}
template <typename T>
array<T> assign_memory(const mxArray *pm)
{
    return array<T>(mxGetNumberOfElements(pm), (T*)mxGetData(pm));
}
template <typename T>
void deallocate_memory(array<T> &ptr)
{
    mxFree(ptr.getData());
}
template <typename T>
array<T> getArrayInput(const mxArray *prhs[], int index)
{
    return array<T>(mxGetNumberOfElements(prhs[index]), (T *) mxGetData(prhs[index]));
}
template <typename T>
array<T> getArrayInput(const mxArray* pm)
{
    if (pm==NULL)
        mexErrMsgTxt("Getting array input of NULL!");
    return array<T>(mxGetNumberOfElements(pm), (T *) mxGetPr(pm));
}
#else
#define arr(type)    type*

template <typename T>
T* allocate_memory(unsigned int length)
{
    return (T*)malloc(length * sizeof(T));
}
template <typename T>
T* allocate_memory(unsigned int length, T value)
{
    T* ptr = (T*)malloc(length * sizeof(T));
    for (int i=length-1; i>=0; i--)
        ptr[i] = value;
    return ptr;
}
/*
template <typename T>
T* assign_memory(const mxArray *pm)
{
    return (T*)mxGetData(pm);
}
*/
template <typename T>
void deallocate_memory(T* ptr)
{
    free(ptr);
}

template<typename T>
array<T> getArrayInput(int len, void *ptr) {
    return array<T>(len, (T *) ptr);
}

template<typename T>
array<T> getArrayInput(cv::Mat &m) {
    return array<T>(m.rows*m.cols, (T*) m.data);
}

template<typename T>
array<T> getArrayInput(cv::Vec<T, 3> &v) {
    return array<T>(3, (T *)v.val);
}

template<typename T>
array<T> getArrayInput(cv::Vec<T, 2> &v) {
    return array<T>(2, (T *)v.val);
}


/*
template <typename T>
T* getArrayInput(const mxArray *prhs[], int index)
{
    return (T *) mxGetData(prhs[index]);
}
template <typename T>
T* getArrayInput(const mxArray* pm)
{
    if (pm==NULL)
        mexErrMsgTxt("Getting array input of NULL!");
    return (T *) mxGetPr(pm);
}
*/
#endif



template <typename T>
T min(T v1, T v2)
{
    return (v1<v2 ? v1 : v2);
}

template <typename T>
T max(T v1, T v2)
{
    return (v1>v2 ? v1 : v2);
}

/*
template <typename T>
T getInput(const mxArray *prhs[], int index)
{
    return *( (T *) mxGetData(prhs[index]));
}
template <typename T>
T getInput(const mxArray *pm)
{
    if (pm==NULL)
        mexErrMsgTxt("Getting input of NULL!");
    return *( (T *) mxGetData(pm));
}
*/

inline void dbl_memset(double* ptr, double val, int N)
{
    for (int i=N-1; i>=0; i--)
        ptr[i] = val;
}

/*inline double mx_rand(void)
{
    mxArray *plhs[1];
    if(mexCallMATLAB(1,plhs,0,NULL,"rand")) {
        mexErrMsgTxt("mexCallMATLAB(rand) failed");
    }
    return mxGetPr(plhs[0])[0];
}*/

/*
// --------------------------------------------------------------------------
// -- checkInput
// --   Checks the input to be the correct type and/or dimensions.  If
// -- incorrect, this will throw an error.
// --
// --   parameters:
// --     - prhs : the parameters passed into the MEX function
// --     - index : the index of the parameter to check
// --     - flag : a character indicating what the parameter must be
// --         's' : scalar
// --         'v' : vector
// --         'm' : matrix
// --         'n' : not scalar (vector or matrix)
// --         'c' : cell
// --------------------------------------------------------------------------
inline void checkInput(const mxArray *prhs[], int index, inputType type)
{
    // flag is either 's'-scalar, 'v'-vector, 'm'-matrix
    int mrows = mxGetM(prhs[index]);
    int ncols = mxGetN(prhs[index]);
    int ndims = mxGetNumberOfDimensions(prhs[index]);

    switch (type)
    {
    case SCALAR:
        if (!(mrows==1 && ncols==1))
        {
            std::cerr << " ->Error with input " << index << "\n";
            mexErrMsgTxt("    Should be a scalar");
        }
        break;
    case VECTOR:
        if ((mrows==1 && ncols==1) || (mrows!=1 && ncols!=1))
        {
            std::cerr << " ->Error with input " << index << "\n";
            mexErrMsgTxt("    Should be a vector");
        }
        break;
    case MATRIX:
        if (mrows==1 || ncols==1)
        {
            std::cerr << " ->Error with input " << index << "\n";
            mexErrMsgTxt("    Should be a matrix");
        }
        break;
    case MATRIX3:
        if (ndims!=3)
        {
            std::cerr << " ->Error with input " << index << "\n";
            mexErrMsgTxt("    Should be a 3D array");
        }
        break;
    case NSCALAR:
        if (mrows==1 && ncols==1)
        {
            std::cerr << " ->Error with input " << index << "\n";
            mexErrMsgTxt("    Should not be a scalar");
        }
        break;
    case CELL:
        if (!mxIsCell(prhs[index]))
        {
            std::cerr << " ->Error with input " << index << "\n";
            mexErrMsgTxt("    Should be a cell array");
        }
        break;
    case STRUCT:
        if (!mxIsStruct(prhs[index]))
        {
            std::cerr << " ->Error with input " << index << "\n";
            mexErrMsgTxt("    Should be a structure");
        }
    }
}
inline void checkInput(const mxArray *prhs[], int index, inputType type, const char* classname)
{
    checkInput(prhs, index, type);
    if (!mxIsClass(prhs[index], classname))
    {
        std::cerr << " ->Error with input " << index << "\n";
        mexErrMsgTxt("    Class error");
    }
}

inline void checkInput(const mxArray *prhs, inputType type)
{
    // flag is either 's'-scalar, 'v'-vector, 'm'-matrix
    int mrows = mxGetM(prhs);
    int ncols = mxGetN(prhs);
    int ndims = mxGetNumberOfDimensions(prhs);

    switch (type)
    {
    case SCALAR:
        if (!(mrows==1 && ncols==1))
        {
            std::cerr << " ->Error with input\n";
            mexErrMsgTxt("    Should be a scalar");
        }
        break;
    case VECTOR:
        if ((mrows==1 && ncols==1) || (mrows!=1 && ncols!=1))
        {
            std::cerr << " ->Error with input\n";
            mexErrMsgTxt("    Should be a vector");
        }
        break;
    case MATRIX:
        if (mrows==1 || ncols==1)
        {
            std::cerr << " ->Error with input\n";
            mexErrMsgTxt("    Should be a matrix");
        }
        break;
    case MATRIX3:
        if (ndims!=3)
        {
            std::cerr << " ->Error with input\n";
            mexErrMsgTxt("    Should be a 3D array");
        }
        break;
    case NSCALAR:
        if (mrows==1 && ncols==1)
        {
            std::cerr << " ->Error with input\n";
            mexErrMsgTxt("    Should not be a scalar");
        }
        break;
    case CELL:
        if (!mxIsCell(prhs))
        {
            std::cerr << " ->Error with input\n";
            mexErrMsgTxt("    Should be a cell array");
        }
        break;
    case STRUCT:
        if (!mxIsStruct(prhs))
        {
            std::cerr << " ->Error with input\n";
            mexErrMsgTxt("    Should be a structure");
        }
    }
}
inline void checkInput(const mxArray *prhs, inputType type, const char* classname)
{
    checkInput(prhs, type);
    if (!mxIsClass(prhs, classname))
    {
        std::cerr << " ->Error with input\n";
        mexErrMsgTxt("    Class error");
    }
}
*/

/*
// --------------------------------------------------------------------------
// -- checkInput
// --   Checks the input to be the correct type and/or dimensions.  If
// -- incorrect, this will throw an error.
// --
// --   parameters:
// --     - prhs : the parameters passed into the MEX function
// --     - index : the index of the parameter to check
// --     - flag : a character indicating what the parameter must be
// --         's' : scalar
// --         'v' : vector
// --         'm' : matrix
// --         'n' : not scalar (vector or matrix)
// --         'c' : cell
// --------------------------------------------------------------------------
inline void checkInput(const mxArray *prhs[], int index, char flag)
{
    // flag is either 's'-scalar, 'v'-vector, 'm'-matrix
    int mrows = mxGetM(prhs[index]);
    int ncols = mxGetN(prhs[index]);
    int ndims = mxGetNumberOfDimensions(prhs[index]);

    switch (flag)
    {
    case 's':
        if (!(mrows==1 && ncols==1))
        {
            puts(" ->Error with input!");
            puts("   Input number:");
            char str[3];
            sprintf(str, "%d", index);
            puts(str);
            mexErrMsgTxt("    Should be a scalar");
        }
        break;
    case 'v':
        if ((mrows==1 && ncols==1) || (mrows!=1 && ncols!=1))
        {
            puts(" ->Error with input!");
            puts("   Input number:");
            char str[3];
            sprintf(str, "%d", index);
            puts(str);
            mexErrMsgTxt("    Should be a vector");
        }
        break;
    case 'm':
        if (mrows==1 || ncols==1)
        {
            puts(" ->Error with input!");
            puts("   Input number:");
            char str[3];
            sprintf(str, "%d", index);
            puts(str);
            mexErrMsgTxt("    Should be a matrix");
        }
        break;
    case 'n':
        if (mrows==1 && ncols==1)
        {
            puts(" ->Error with input!");
            puts("   Input number:");
            char str[3];
            sprintf(str, "%d", index);
            puts(str);
            mexErrMsgTxt("    Should not be a scalar");
        }
        break;
    case 'c':
        if (!mxIsCell(prhs[index]))
        {
            puts(" ->Error with input!");
            puts("   Input number:");
            char str[3];
            sprintf(str, "%d", index);
            puts(str);
            mexErrMsgTxt("    Should be a cell array");
        }
        break;
    case '3':
        if (ndims!=3)
        {
            puts(" ->Error with input!");
            puts("   Input number:");
            char str[3];
            sprintf(str, "%d", index);
            puts(str);
            mexErrMsgTxt("    Should be a 3D array");
        }
        break;
    }
}

// --------------------------------------------------------------------------
// -- findCoordinates
// --   converts 1D coordinates to 2D coordinates
// --
// --   parameters:
// --     - (X,Y) : the dimensions
// --     - index : the 1D coordinate
// --
// --   return parameters:
// --     - (x,y) : the 2D coordinates
// --------------------------------------------------------------------------
*/
inline void findCoordinates(int X, int Y, int index, int &x, int &y)
{
    // index first does x's then does y's
    x = index%Y;
    y = index/Y;
}

// --------------------------------------------------------------------------
// -- findCoordinates
// --   converts 2D coordinates to 1D coordinate
// --
// --   parameters:
// --     - (X,Y) : the dimensions
// --     - (x,y) : the 2D coordinates
// --
// --   return value:
// --     - index : the 1D coordinates
// --------------------------------------------------------------------------
inline int findIndex(int X, int Y, int x, int y)
{
    return y*X + x;
}
// --------------------------------------------------------------------------
// -- findCoordinates
// --   converts 3D coordinates to 1D coordinate
// --
// --   parameters:
// --     - (X,Y,Z) : the dimensions
// --     - (x,y,z) : the 3D coordinates
// --
// --   return value:
// --     - index : the 1D coordinates
// --------------------------------------------------------------------------
inline int findIndex(int X, int Y, int Z, int x, int y, int z)
{
    return z*X*Y + y*X + x;
}


#endif
