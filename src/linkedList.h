// =============================================================================
// == linkedList.h
// == --------------------------------------------------------------------------
// == A Linked List Node class
// == --------------------------------------------------------------------------
// == Copyright 2011. MIT. All Rights Reserved.
// == Written by Jason Chang 01-14-2008
// =============================================================================
//#ifndef LINKEDLIST
//#define LINKEDLIST
//#include "linkedList.cpp"

#include "assert.h"
#include "string.h"
#include <cstdlib>
#include <stdlib.h>
#include <iostream>

template <typename T>
class linkedListNode
{
private:
    T data;
    linkedListNode* prev;
    linkedListNode* next;
    template <typename TT> friend class linkedList;

public:
    // --------------------------------------------------------------------------
    // -- linkedListNode
    // --   constructor; initializes the linked list node to nothing
    // --------------------------------------------------------------------------
    linkedListNode();

    // --------------------------------------------------------------------------
    // -- linkedListNode
    // --   constructor; initializes the linked list node to contain the data
    // --
    // --   parameters:
    // --     - new_data : the data to put in the new node
    // --------------------------------------------------------------------------
    linkedListNode(T new_data);

    // --------------------------------------------------------------------------
    // -- getData
    // --   retrieves the data at the node
    // --------------------------------------------------------------------------
    T getData() const;

    // --------------------------------------------------------------------------
    // -- setData
    // --   set the data at the node
    // --------------------------------------------------------------------------
    void setData(T newData);
    // --------------------------------------------------------------------------
    // -- getPrev
    // --   returns the pointer to the previous node
    // --------------------------------------------------------------------------
    linkedListNode<T>* getPrev() const;
    // --------------------------------------------------------------------------
    // -- getNext
    // --   returns the pointer to the previous node
    // --------------------------------------------------------------------------
    linkedListNode<T>* getNext() const;

    void applyFunction(T (*func)(T));
    void applyFunction(T (*func)(T,T), T data2);

    void printForward();
};


// =============================================================================
// == linkedList.h
// == --------------------------------------------------------------------------
// == A Linked List class
// == --------------------------------------------------------------------------
// == Written by Jason Chang 01-14-2008
// =============================================================================


template <typename T>
class linkedList
{
private:
    int length;
    linkedListNode<T>* first;
    linkedListNode<T>* last;

public:
    // --------------------------------------------------------------------------
    // -- linkedList
    // --   constructor; initializes the linked list to nothing
    // --------------------------------------------------------------------------
    linkedList();
    linkedList(const linkedList& that);
    // --------------------------------------------------------------------------
    // -- linkedList
    // --   destructor
    // --------------------------------------------------------------------------
    ~linkedList();

    // --------------------------------------------------------------------------
    // -- getFirst
    // --   returns the pointer to the first
    // --------------------------------------------------------------------------
    linkedListNode<T>* getFirst() const;

    // --------------------------------------------------------------------------
    // -- getLength
    // --   returns the length of the linked list
    // --------------------------------------------------------------------------
    int getLength() const;

    // --------------------------------------------------------------------------
    // -- getArray
    // --   returns a pointer to an array of the data that should be deallocated
    // -- by yourself
    // --------------------------------------------------------------------------
    T* getArray();

    bool isempty();

    // --------------------------------------------------------------------------
    // -- addNode
    // --   adds a node to the beginning of the list
    // --
    // --   parameters:
    // --     - new_data : the data of the new node
    // --
    // --   return_value: a pointer to the newly added leaf
    // --------------------------------------------------------------------------
    linkedListNode<T>* addNode(T new_data);

    // --------------------------------------------------------------------------
    // -- addNodeEnd
    // --   adds a node to the end of the list
    // --
    // --   parameters:
    // --     - new_data : the data of the new node
    // --
    // --   return_value: a pointer to the newly added leaf
    // --------------------------------------------------------------------------
    linkedListNode<T>* addNodeEnd(T new_data);

    // --------------------------------------------------------------------------
    // -- deleteNode
    // --   deletes a node
    // --
    // --   parameters:
    // --     - theNode : a pointer to the node to remove
    // --------------------------------------------------------------------------
    void deleteNode(linkedListNode<T>* theNode);

    T popFirst();
    T popLast();

    void popFirst(int numToPop, linkedList<T> &poppedList);

    // --------------------------------------------------------------------------
    // -- dataAt
    // --   Returns the data at index
    // --------------------------------------------------------------------------
    T dataAt(int index);

    // --------------------------------------------------------------------------
    // -- indexOf
    // --   Returns the first index of data
    // --------------------------------------------------------------------------
    int indexOf(T theData);

    // --------------------------------------------------------------------------
    // -- clear
    // --   deletes the entire linked list
    // --------------------------------------------------------------------------
    void clear();

    // --------------------------------------------------------------------------
    // -- print
    // --   prints the list.  used for debugging
    // --------------------------------------------------------------------------
    void print();

    void append(linkedList<T> &b);

    // --------------------------------------------------------------------------
    // -- sort
    // --   sorts the list
    // --------------------------------------------------------------------------
    void sort();
    void merge_with(linkedList<T> &b);
    void merge_sort(linkedListNode<T>* &a);
    linkedListNode<T>* sorted_merge(linkedListNode<T>* a, linkedListNode<T>* b);
    void split(linkedListNode<T>* a, linkedListNode<T>* &b);
};

//#endif
