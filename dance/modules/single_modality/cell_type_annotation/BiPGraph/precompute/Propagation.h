#ifndef PROPAGATION_H
#define PROPAGATION_H
#include <iostream>
#include <vector>
#include <string>
using namespace std;

vector<vector<double> > ppr(string data,double alpha,double rmax,double rrr);
vector<vector<double> > transition(string data,double rmax, int rwnum ,double rrr);





#endif
