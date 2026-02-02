#ifndef GRAPH_H
#define GRAPH_H

#include "utils.h"

using namespace std;

class Graph
{
public:
    int n;
    long long  m;
    vector< vector<int> > G;
    vector<int> Degree;
    int dimension;
    int NUMTHREAD;
    double alpha;
    double rmax;
    double rrr;
    int walk_num;
    vector<int> random_w;
    vector<vector<double>> negativeFeature;
    vector<vector<double>> positiveFeature;
    vector<double> negativeRowSum;
    vector<double> positiveRowSum;
    vector<int> rwIndex;
    vector<vector<pair<int, double>>> ppr1;
    vector<vector<pair<int, double>>> ppr2;
    vector<vector<pair<int, double>>> ppr3;
    vector<vector<pair<int, double>>> ppr4;
    string dataset;
    uint32_t seed = time(0);

    Graph(string dataStr,int num,double decay,double err,double rrz,int rwnum )
    {
        dataset = dataStr;
        NUMTHREAD = num;
        alpha = decay;
        rmax = err;
        rrr = rrz;
        walk_num = rwnum;
        if(dataset =="friendster"){
            LoadSplitAdjs();
            LoadSplitFeatures();
            LoadRWindex_friendster();

        }else{
            LoadGraph();
            LoadFeatures();
        }
        
    }

    void LoadGraph()
    {
    	n = 0;
        m = 0;
        string dataPath = "data/"+dataset+".txt";
    	ifstream infile(dataPath);
        assert(infile.is_open());
    	infile >> n;
        G = vector< vector<int> >(n);
        Degree = vector<int>(n,0);
        int fromNode, toNode;
        while(infile >> fromNode >> toNode){
            G[toNode].push_back(fromNode);
            Degree[fromNode]++;
            m++;
        }
        infile.close();
    }

    void LoadFeatures(){
        cnpy::NpyArray arr_mv1 = cnpy::npy_load("data/"+dataset+"_feat.npy");
        auto mv1 = arr_mv1.data<float>();
        int nrows = arr_mv1.shape [0];
        int ncols = arr_mv1.shape [1];
        dimension = ncols;
        if(NUMTHREAD > dimension){
            NUMTHREAD = dimension;
        }
        random_w = vector<int>(dimension);
        negativeRowSum = vector<double>(dimension,0);
        positiveRowSum = vector<double>(dimension,0);
        for(int i = 0 ; i < dimension ; i++ ){
            random_w[i] = i;
        }
        random_shuffle(random_w.begin(),random_w.end());
        negativeFeature = vector<vector<double>>(ncols,vector<double>(nrows));
        positiveFeature = vector<vector<double>>(ncols,vector<double>(nrows));        
        for(int row = 0; row <nrows; row ++){
            for(int col = 0; col <ncols; col ++){
                auto val = mv1[row*ncols+col];
                if(Degree[row]>0){
                    val = val/pow(Degree[row],rrr);
                }
                
                if(val>0){
                    positiveFeature[col][row]=val;
                    positiveRowSum[col]+=val;
                }else{
                    negativeFeature[col][row]=-val;
                    negativeRowSum[col]+=-val;
                }
            }
        }
        for(int i = 0 ; i < ncols ; i++ ){
            if(positiveRowSum[i]==0){
                positiveRowSum[i] = 1;
            }
            if(negativeRowSum[i]==0){
                negativeRowSum[i] = 1;
            }
        }
    }


    void LoadRWindex_friendster(){
        cnpy::NpyArray arr_mv1 = cnpy::npy_load("data/friendster_rw_index.npy");
        auto mv1 = arr_mv1.data<long>();
        int rw_len = arr_mv1.shape [0];
        cout<<"rw-index: "<<rw_len<<endl;
        rwIndex = vector<int>(rw_len);

        for(int row = 0; row <rw_len; row ++){
            rwIndex[row] = (int)mv1[row];
        }
        random_shuffle(rwIndex.begin(),rwIndex.end());
    }

    void LoadSplitFeatures(){
        cnpy::NpyArray arr_mv1 = cnpy::npy_load("data/"+dataset+"_feat1.npy");
        cnpy::NpyArray arr_mv2 = cnpy::npy_load("data/"+dataset+"_feat2.npy");
        cnpy::NpyArray arr_mv3 = cnpy::npy_load("data/"+dataset+"_feat3.npy");
        cnpy::NpyArray arr_mv4 = cnpy::npy_load("data/"+dataset+"_feat4.npy");
        cnpy::NpyArray arr_mv5 = cnpy::npy_load("data/"+dataset+"_feat5.npy");
        cnpy::NpyArray arr_mv6 = cnpy::npy_load("data/"+dataset+"_feat6.npy");
        cnpy::NpyArray arr_mv7 = cnpy::npy_load("data/"+dataset+"_feat7.npy");
        cnpy::NpyArray arr_mv8 = cnpy::npy_load("data/"+dataset+"_feat8.npy");
        cnpy::NpyArray arr_mv9 = cnpy::npy_load("data/"+dataset+"_feat9.npy");
        cnpy::NpyArray arr_mv10 = cnpy::npy_load("data/"+dataset+"_feat10.npy");
        auto mv1 = arr_mv1.data<float>();
        auto mv2 = arr_mv2.data<float>();
        auto mv3 = arr_mv3.data<float>();
        auto mv4 = arr_mv4.data<float>();
        auto mv5 = arr_mv5.data<float>();
        auto mv6 = arr_mv6.data<float>();
        auto mv7 = arr_mv7.data<float>();
        auto mv8 = arr_mv8.data<float>();
        auto mv9 = arr_mv9.data<float>();
        auto mv10 = arr_mv10.data<float>();
        vector<int> row_vec(10);
        row_vec[0] = arr_mv1.shape[0];
        row_vec[1] = arr_mv2.shape[0];
        row_vec[2] = arr_mv3.shape[0];
        row_vec[3] = arr_mv4.shape[0];
        row_vec[4] = arr_mv5.shape[0];
        row_vec[5] = arr_mv6.shape[0];
        row_vec[6] = arr_mv7.shape[0];
        row_vec[7] = arr_mv8.shape[0];
        row_vec[8] = arr_mv9.shape[0];
        row_vec[9] = arr_mv10.shape[0];
        int nrows = 0;
        for(auto r: row_vec ){
            nrows+=r;
        }
        int ncols = arr_mv1.shape [1];
        dimension = ncols;
        if(NUMTHREAD > dimension){
            NUMTHREAD = dimension;
        }
        random_w = vector<int>(dimension);
        negativeRowSum = vector<double>(dimension,0);
        positiveRowSum = vector<double>(dimension,0);
        for(int i = 0 ; i < dimension ; i++ ){
            random_w[i] = i;
        }
        random_shuffle(random_w.begin(),random_w.end());
        negativeFeature = vector<vector<double>>(ncols,vector<double>(nrows));
        positiveFeature = vector<vector<double>>(ncols,vector<double>(nrows));


        int finished = 0;
        for(int row = 0; row <row_vec[0]; row ++){
            for(int col = 0; col <ncols; col ++){
                auto val = mv1[row*ncols+col];
                if(val>0){
                    positiveRowSum[col]+=val;
                }else{
                    negativeRowSum[col]+=val;
                }
                val = val/pow(Degree[row+finished],rrr);
                negativeFeature[col][row+finished]=val;
            }
        }
        finished+=row_vec[0];
        for(int row = 0; row <row_vec[1]; row ++){
            for(int col = 0; col <ncols; col ++){
                auto val = mv2[row*ncols+col];
                if(val>0){
                    positiveRowSum[col]+=val;
                }else{
                    negativeRowSum[col]+=val;
                }
                val = val/pow(Degree[row+finished],rrr);
                negativeFeature[col][row+finished]=val;
            }
        }
        finished+=row_vec[1];
        for(int row = 0; row <row_vec[2]; row ++){
            for(int col = 0; col <ncols; col ++){
                auto val = mv3[row*ncols+col];
                if(val>0){
                    positiveRowSum[col]+=val;
                }else{
                    negativeRowSum[col]+=val;
                }
                val = val/pow(Degree[row+finished],rrr);
                negativeFeature[col][row+finished]=val;
            }
        }
        finished+=row_vec[2];
        for(int row = 0; row <row_vec[3]; row ++){
            for(int col = 0; col <ncols; col ++){
                auto val = mv4[row*ncols+col];
                if(val>0){
                    positiveRowSum[col]+=val;
                }else{
                    negativeRowSum[col]+=val;
                }
                val = val/pow(Degree[row+finished],rrr);
                negativeFeature[col][row+finished]=val;
            }
        }
        finished+=row_vec[3];
        for(int row = 0; row <row_vec[4]; row ++){
            for(int col = 0; col <ncols; col ++){
                auto val = mv5[row*ncols+col];
                if(val>0){
                    positiveRowSum[col]+=val;
                }else{
                    negativeRowSum[col]+=val;
                }
                val = val/pow(Degree[row+finished],rrr);
                negativeFeature[col][row+finished]=val;
            }
        }
        finished+=row_vec[4];
        for(int row = 0; row <row_vec[5]; row ++){
            for(int col = 0; col <ncols; col ++){
                auto val = mv6[row*ncols+col];
                if(val>0){
                    positiveRowSum[col]+=val;
                }else{
                    negativeRowSum[col]+=val;
                }
                val = val/pow(Degree[row+finished],rrr);
                negativeFeature[col][row+finished]=val;
            }
        }
        finished+=row_vec[5];
        for(int row = 0; row <row_vec[6]; row ++){
            for(int col = 0; col <ncols; col ++){
                auto val = mv7[row*ncols+col];
                if(val>0){
                    positiveRowSum[col]+=val;
                }else{
                    negativeRowSum[col]+=val;
                }
                val = val/pow(Degree[row+finished],rrr);
                negativeFeature[col][row+finished]=val;
            }
        }
        finished+=row_vec[6];
        for(int row = 0; row <row_vec[7]; row ++){
            for(int col = 0; col <ncols; col ++){
                auto val = mv8[row*ncols+col];
                if(val>0){
                    positiveRowSum[col]+=val;
                }else{
                    negativeRowSum[col]+=val;
                }
                val = val/pow(Degree[row+finished],rrr);
                negativeFeature[col][row+finished]=val;
            }
        }
        finished+=row_vec[7];
        for(int row = 0; row <row_vec[8]; row ++){
            for(int col = 0; col <ncols; col ++){
                auto val = mv9[row*ncols+col];
                if(val>0){
                    positiveRowSum[col]+=val;
                }else{
                    negativeRowSum[col]+=val;
                }
                val = val/pow(Degree[row+finished],rrr);
                negativeFeature[col][row+finished]=val;
            }
        }
        finished+=row_vec[8];
        for(int row = 0; row <row_vec[9]; row ++){
            for(int col = 0; col <ncols; col ++){
                auto val = mv10[row*ncols+col];
                if(val>0){
                    positiveRowSum[col]+=val;
                }else{
                    negativeRowSum[col]+=val;
                }
                val = val/pow(Degree[row+finished],rrr);
                negativeFeature[col][row+finished]=val;
            }
        }
    }




    void LoadSplitAdjs(){
  
        cnpy::NpyArray arr_mv1 = cnpy::npy_load("data/"+dataset+"1.npy");
        cnpy::NpyArray arr_mv2 = cnpy::npy_load("data/"+dataset+"2.npy");
        cnpy::NpyArray arr_mv3 = cnpy::npy_load("data/"+dataset+"3.npy");
        cnpy::NpyArray arr_mv4 = cnpy::npy_load("data/"+dataset+"4.npy");
        cnpy::NpyArray arr_mv5 = cnpy::npy_load("data/"+dataset+"5.npy");
        cnpy::NpyArray arr_mv6 = cnpy::npy_load("data/"+dataset+"6.npy");
        cnpy::NpyArray arr_mv7 = cnpy::npy_load("data/"+dataset+"7.npy");
        cnpy::NpyArray arr_mv8 = cnpy::npy_load("data/"+dataset+"8.npy");
        cnpy::NpyArray arr_mv9 = cnpy::npy_load("data/"+dataset+"9.npy");
        cnpy::NpyArray arr_mv10 = cnpy::npy_load("data/"+dataset+"10.npy");
        auto mv1 = arr_mv1.data<long>();
        auto mv2 = arr_mv2.data<long>();
        auto mv3 = arr_mv3.data<long>();
        auto mv4 = arr_mv4.data<long>();
        auto mv5 = arr_mv5.data<long>();
        auto mv6 = arr_mv6.data<long>();
        auto mv7 = arr_mv7.data<long>();
        auto mv8 = arr_mv8.data<long>();
        auto mv9 = arr_mv9.data<long>();
        auto mv10 = arr_mv10.data<long>();
        vector<int> row_vec(10);
        row_vec[0] = arr_mv1.shape[0];
        row_vec[1] = arr_mv2.shape[0];
        row_vec[2] = arr_mv3.shape[0];
        row_vec[3] = arr_mv4.shape[0];
        row_vec[4] = arr_mv5.shape[0];
        row_vec[5] = arr_mv6.shape[0];
        row_vec[6] = arr_mv7.shape[0];
        row_vec[7] = arr_mv8.shape[0];
        row_vec[8] = arr_mv9.shape[0];
        row_vec[9] = arr_mv10.shape[0];
        n = mv1[0];
        int nrows = 0;
        for(auto r: row_vec ){
            nrows+=r;
        }
        m = mv1[1];
        assert(m==nrows-1);
        m*=2;
        G = vector< vector<int> >(n);
        Degree = vector<int>(n,0);
        ppr1 = vector<vector<pair<int, double>>>(n);
        ppr2 = vector<vector<pair<int, double>>>(n);
        ppr3 = vector<vector<pair<int, double>>>(n);
        ppr4 = vector<vector<pair<int, double>>>(n);
        int ncols = arr_mv1.shape [1];
        for(int row = 1; row <row_vec[0]; row ++){
            int u = mv1[row*ncols];
            int v = mv1[row*ncols+1];
            G[u].push_back(v);
            G[v].push_back(u);
            Degree[u]++;
            Degree[v]++;
        }
        for(int row = 0; row <row_vec[1]; row ++){
            int u = mv2[row*ncols];
            int v = mv2[row*ncols+1];
            G[u].push_back(v);
            G[v].push_back(u);
            Degree[u]++;
            Degree[v]++;
        }
        for(int row = 0; row <row_vec[2]; row ++){
            int u = mv3[row*ncols];
            int v = mv3[row*ncols+1];
            G[u].push_back(v);
            G[v].push_back(u);
            Degree[u]++;
            Degree[v]++;
        }
        for(int row = 0; row <row_vec[3]; row ++){
            int u = mv4[row*ncols];
            int v = mv4[row*ncols+1];
            G[u].push_back(v);
            G[v].push_back(u);
            Degree[u]++;
            Degree[v]++;
        }
        for(int row = 0; row <row_vec[4]; row ++){
            int u = mv5[row*ncols];
            int v = mv5[row*ncols+1];
            G[u].push_back(v);
            G[v].push_back(u);
            Degree[u]++;
            Degree[v]++;
        }
        for(int row = 0; row <row_vec[5]; row ++){
            int u = mv6[row*ncols];
            int v = mv6[row*ncols+1];
            G[u].push_back(v);
            G[v].push_back(u);
            Degree[u]++;
            Degree[v]++;
        }
        for(int row = 0; row <row_vec[6]; row ++){
            int u = mv7[row*ncols];
            int v = mv7[row*ncols+1];
            G[u].push_back(v);
            G[v].push_back(u);
            Degree[u]++;
            Degree[v]++;
        }
        for(int row = 0; row <row_vec[7]; row ++){
            int u = mv8[row*ncols];
            int v = mv8[row*ncols+1];
            G[u].push_back(v);
            G[v].push_back(u);
            Degree[u]++;
            Degree[v]++;
        }
        for(int row = 0; row <row_vec[8]; row ++){
            int u = mv9[row*ncols];
            int v = mv9[row*ncols+1];
            G[u].push_back(v);
            G[v].push_back(u);
            Degree[u]++;
            Degree[v]++;
        }
        for(int row = 0; row <row_vec[9]; row ++){
            int u = mv10[row*ncols];
            int v = mv10[row*ncols+1];
            G[u].push_back(v);
            G[v].push_back(u);
            Degree[u]++;
            Degree[v]++;
        }

        //self loop
        for(int i = 0 ; i < n ; i ++ ){
            G[i].push_back(i);
            Degree[i]++;
            m++;
        }
    }

    double rand_0_1(){
        return rand_r(&seed)%RAND_MAX/(double)RAND_MAX;
    }
    int rand_max(){
        return rand_r(&seed);
    }

};

#endif
