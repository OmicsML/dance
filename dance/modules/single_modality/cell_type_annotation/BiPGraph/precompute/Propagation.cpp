#include "base.h"

using namespace std;



vector<vector<double> > ppr(string data,double alpha,double rmax,double rrr){
    int NUMTHREAD = 40;
    Base g(data,NUMTHREAD,alpha,rmax,rrr,0);
    g.ppr_push();
    return g.positiveFeature;
}

vector<vector<double> > transition(string data,double rmax,int rwnum,double rrr){
    int NUMTHREAD = 40;
    Base g(data,NUMTHREAD,0,rmax,rrr,rwnum);
    g.rw_push();
    return g.positiveFeature;
}


int main(int argc, char* argv[]){
    return 0;
}
