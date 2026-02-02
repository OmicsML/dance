%module  BiP
%{
#define SWIG_WITH_INIT
#include "precompute/Propagation.h"
%}
%include "precompute/Propagation.h"
%include stl.i
%include "std_string.i"
%include "std_vector.i"
// %include "std_pair.i"

vector<vector<double> > ppr(std::string data,double alpha,double rmax,double rrr);
vector<vector<double> > transition(std::string data,double rmax, int rwnum ,double rrr);
namespace std{
    %template(doublevector) vector<double>;
    %template(doublemat) vector<vector<double> >;

}

