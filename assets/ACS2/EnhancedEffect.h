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

// Node in EnhancedEffect.
// (position, ProbabilityEnhancedAttribute) tuple, along with a pointer to the next node.
class AttrWithPos {
    friend class EnhancedEffect;

public:
    int getPos() { return p; }

    ProbabilityEnhancedAttribute *getItem() { return item; }

private:
    AttrWithPos(ProbabilityEnhancedAttribute *pcl, int pos) {
        p = pos;
        item = pcl;
        next = 0;
    }

    ~AttrWithPos() {
        delete item;
        delete next;
    }

    int p;
    ProbabilityEnhancedAttribute *item;

    AttrWithPos *next;
};

// Full Enhanced Effect representation.
// List of (position, ProbabilityEnhancedAttribute) tuples, with a stored iterator.
class EnhancedEffect {
    friend class Effect;

public:
    void reset() { act = first; }

    AttrWithPos *getNextItem();

private:
    EnhancedEffect() {
        first = 0;
        act = 0;
        size = 0;
    }

    EnhancedEffect(char ch, int pos);

    EnhancedEffect(EnhancedEffect *oldList);

    ~EnhancedEffect() { delete first; }

    int insert(char chr, int pos);

    int insertAt(char chr, int epos);

    int remove(int pos);

    int removeAt(int nr);

    int getSize() { return size; }

    AttrWithPos *getItem(int pos);

    void remove(AttrWithPos *cpip, AttrWithPos *cpipl);

    AttrWithPos *first;
    AttrWithPos *act;
    int size;
};

#endif
