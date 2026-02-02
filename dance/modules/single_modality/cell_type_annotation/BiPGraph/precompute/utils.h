#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <queue>
#include <unordered_map>
#include <math.h>
#include <cmath>
#include <limits.h>
#include <unistd.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <thread>
#include <string>
#include <unistd.h>
#include <sys/time.h>
#include "Propagation.h"
#include "cnpy.h"

#define SetBit(A, k)     ( A[(k/32)] |= (1 << (k%32)) )
#define ClearBit(A, k)   ( A[(k/32)] &= ~(1 << (k%32)) )
#define TestBit(A, k)    ( A[(k/32)] & (1 << (k%32)) )

#define DSetBit(A, k, j, n)     ( A[(k*(n/32)+(j/32))] |= (1 << (((k%32)*(n%32))%32+j%32)%32) )
#define DClearBit(A, k, j, n)   ( A[(k*(n/32)+(j/32))] &= ~(1 << (((k%32)*(n%32))%32+j%32)%32) )
#define DTestBit(A, k, j, n)    ( A[(k*(n/32)+(j/32))] & (1 << (((k%32)*(n%32))%32+j%32)%32) )

using namespace std;

class Node_Set {
public:
  int vert;
  int bit_vert;
  int *HashKey;
  int *HashValue;
  int KeyNumber;


  Node_Set(int n) {
    vert = n;
    bit_vert = n / 32 + 1;
    HashKey = new int[vert];
    HashValue = new int[bit_vert];
    for (int i = 0; i < vert; i++) {
      HashKey[i] = 0;
    }
    for (int i = 0; i < bit_vert; i++) {
      HashValue[i] = 0;
    }
    KeyNumber = 0;
  }


  void Push(int node) {

    if (!TestBit(HashValue, node)) {
      HashKey[KeyNumber] = node;
      KeyNumber++;
    }
    SetBit(HashValue, node);
  }


  int Pop() {
    if (KeyNumber == 0) {
      return -1;
    } else {
      int k = HashKey[KeyNumber - 1];
      ClearBit(HashValue, k);
      KeyNumber--;
      return k;
    }
  }

  void Clean() {
    for (int i = 0; i < KeyNumber; i++) {
      ClearBit(HashValue, HashKey[i]);
      HashKey[i] = 0;
    }
    KeyNumber = 0;
  }

  ~Node_Set() {
    delete[] HashKey;
    delete[] HashValue;
  }
};


#endif