/*
/       ACS2 in C++
/	------------------------------------
/       choice of without/with GA, subsumption, PEEs
/
/     (c) by Martin V. Butz
/     University of Wuerzburg / University of Illinois at Urbana/Champaign
/     butz@illigal.ge.uiuc.edu
/     Last modified: 02-23-2001
/
/     list structure for effect part header
*/


#ifndef _ProbCharPosList_h_
#define _ProbCharPosList_h_

#include<iostream>

#include"ProbabilityEnhancedAttribute.h"

// Node in ProbCharPosList.
// (position, ProbabilityEnhancedAttribute) tuple, along with a pointer to the next node.
class ProbCharPosItem {
    friend class ProbCharPosList;

public:
    int getPos() { return p; }

    ProbabilityEnhancedAttribute *getItem() { return item; }

private:
    ProbCharPosItem(ProbabilityEnhancedAttribute *pcl, int pos) {
        p = pos;
        item = pcl;
        next = 0;
    }

    ~ProbCharPosItem() {
        delete item;
        delete next;
    }

    int p;
    ProbabilityEnhancedAttribute *item;

    ProbCharPosItem *next;
};

// Full Enhanced Effect representation.
// List of (position, ProbabilityEnhancedAttribute) tuples, with a stored iterator.
class ProbCharPosList {
    friend class Effect;

public:
    void reset() { act = first; }

    ProbCharPosItem *getNextItem();

private:
    ProbCharPosList() {
        first = 0;
        act = 0;
        size = 0;
    }

    ProbCharPosList(char ch, int pos);

    ProbCharPosList(ProbCharPosList *oldList);

    ~ProbCharPosList() { delete first; }

    int insert(char chr, int pos);

    int insertAt(char chr, int epos);

    int remove(int pos);

    int removeAt(int nr);

    int getSize() { return size; }

    ProbCharPosItem *getItem(int pos);

    void remove(ProbCharPosItem *cpip, ProbCharPosItem *cpipl);

    ProbCharPosItem *first;
    ProbCharPosItem *act;
    int size;
};

#endif
