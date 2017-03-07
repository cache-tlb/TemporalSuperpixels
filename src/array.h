// =============================================================================
// == array.h
// == --------------------------------------------------------------------------
// == An array class to help with MEX debugging that can be optimized to only
// == be a pointer.  See helperMEX.h
// == --------------------------------------------------------------------------
// == Written by Jason Chang 06-20-2013
// =============================================================================


#ifndef ARRAY
#define ARRAY

#include "assert.h"
#include "string.h"
#include <cstdlib>
#include <stdlib.h>
#include <iostream>

template <typename T>
class array
{
private:
   int length;
   T* data;

public:
   // --------------------------------------------------------------------------
   // -- array
   // --   constructor; initializes the array to nothing
   // --------------------------------------------------------------------------
   array();

   array(int theLength, T* theData);

   // --------------------------------------------------------------------------
   // -- array
   // --   constructor; initializes the array to have length theLength
   // --------------------------------------------------------------------------
   array(int theLength);

   // --------------------------------------------------------------------------
   // -- array
   // --   constructor; initializes the array to have specific length and values
   // --------------------------------------------------------------------------
   array(int theLength, T theValue);

   // --------------------------------------------------------------------------
   // -- getLength
   // --   returns the length of the array
   // --------------------------------------------------------------------------
   int getLength() const;

   // --------------------------------------------------------------------------
   // -- getData
   // --   returns the pointer to the data array
   // --------------------------------------------------------------------------
   T* getData() const;

   // --------------------------------------------------------------------------
   // -- setData
   // --   sets the data and length to inputs
   // --------------------------------------------------------------------------
   void setData(int theLength, T* theData);

   // --------------------------------------------------------------------------
   // -- operator[]
   // --   returns the value at nIndex
   // --------------------------------------------------------------------------
   T& operator[](const int nIndex) const;

   // --------------------------------------------------------------------------
   // -- (T*) this
   // --   overloaded type cast to array pointer
   // --------------------------------------------------------------------------
   operator T*();

   // --------------------------------------------------------------------------
   // -- operator+
   // --   increments the pointer and returns a new
   // --------------------------------------------------------------------------
   array<T> operator+(int inc);
//   template <typename T2>
//   friend const array<T2> operator+(const array<T2> &theArray, int inc) const;

};




// --------------------------------------------------------------------------
// -- array
// --   constructor; initializes the array to nothing
// --------------------------------------------------------------------------
template <typename T>
array<T>::array() :
   length(0), data(NULL)
{
}

template <typename T>
array<T>::array(int theLength, T* theData) :
   length(theLength), data(theData)
{
}

// --------------------------------------------------------------------------
// -- array
// --   constructor; initializes the array to have length theLength
// --------------------------------------------------------------------------
template <typename T>
array<T>::array(int theLength) :
   length(theLength)
{
   data = (T*)malloc(theLength, sizeof(T));
}

// --------------------------------------------------------------------------
// -- array
// --   constructor; initializes the array to have specific length and values
// --------------------------------------------------------------------------
template <typename T>
array<T>::array(int theLength, T theValue) :
   length(theLength)
{
   data = (T*)malloc(theLength, sizeof(T));
   for (int i=0; i<theLength; i++)
      data[i] = theValue;
}

// --------------------------------------------------------------------------
// -- getLength
// --   returns the length of the array
// --------------------------------------------------------------------------
template <typename T>
int array<T>::getLength() const
{
   return length;
}

// --------------------------------------------------------------------------
// -- getData
// --   returns the pointer to the data array
// --------------------------------------------------------------------------
template <typename T>
T* array<T>::getData() const
{
   return data;
}

// --------------------------------------------------------------------------
// -- setData
// --   sets the data and length to inputs
// --------------------------------------------------------------------------
template <typename T>
void array<T>::setData(int theLength, T* theData)
{
   if (length>0 && data!=NULL)
      free(data);
   length = theLength;
   data = theData;
}

// --------------------------------------------------------------------------
// -- operator[]
// --   returns the value at nIndex
// --------------------------------------------------------------------------
template <typename T>
T& array<T>::operator[](const int nIndex) const
{
#ifdef DEBUG_ARR
   if (nIndex<0 || nIndex>=length)
   {
      //print_stack_trace();
      std::cerr<< "Array index (" << nIndex << " / " << length << ") out of range!\n";
      printf("Array indexing error\n");
   }
#endif
   return data[nIndex];
}

// --------------------------------------------------------------------------
// -- (T*) this
// --   overloaded type cast to array pointer
// --------------------------------------------------------------------------
template <typename T>
array<T>::operator T*()
{
   return data;
}

// --------------------------------------------------------------------------
// -- operator+
// --   increments the pointer and returns a new
// --------------------------------------------------------------------------
template <typename T>
array<T> array<T>::operator+(int inc)
{
#ifdef DEBUG_ARR
   if (inc<0 || inc>=length)
      printf("Array indexing error with pointer addition\n");
#endif
   return array<T>(length-inc, data+inc);
}
/*template <typename T2>
const array<T2> operator+(const array<T2> &theArray, int inc) const
{
   if (inc<0 || inc>=theArray.length)
      std::cerr<<"Array indexing addition out of range!\n";

   return array<T2>(theArray.length-inc, theArray.data+inc, false);
}*/


#endif
