#ifndef BASE_H
#define BASE_H

#include "graph.h"


using namespace std;

class Base:public Graph{

    public:

    Base(string dataStr,int num,double decay,double err,double rrz,int rwnum):Graph(dataStr,num,decay,err,rrz,rwnum){

    }

    void BackwardPush_ppr(int start, int end){
        double* reserve = new double[n];
        Node_Set* candidate_set1 = new Node_Set(n);
        Node_Set* candidate_set2 = new Node_Set(n);

        for(int it = start ; it < end ; it++ ){
            int w = random_w[it];
            double rowsum_pos = positiveRowSum[w];
            double rowsum_neg = negativeRowSum[w];

            for(int ik = 0 ; ik < n ; ik++ ){
                if( positiveFeature[w][ik] > rmax*rowsum_pos ){
                    candidate_set1->Push(ik);
                }
                if( negativeFeature[w][ik] > rmax*rowsum_neg ){
                    candidate_set2->Push(ik);
                }
                reserve[ik] = 0;
            }

            while(candidate_set1->KeyNumber !=0){
                int oldNode = candidate_set1->Pop();
                double oldresidue = positiveFeature[w][oldNode];
                double rpush = (1-alpha)*oldresidue;
                reserve[oldNode]+=alpha*oldresidue;
                positiveFeature[w][oldNode] = 0;
                for( auto newNode : G[oldNode] ){
                    positiveFeature[w][newNode]+=rpush/Degree[newNode];
                    if(positiveFeature[w][newNode] > rmax*rowsum_pos){
                        candidate_set1->Push(newNode);
                    }
                }
            }
            while(candidate_set2->KeyNumber !=0){
                int oldNode = candidate_set2->Pop();
                double oldresidue = negativeFeature[w][oldNode];
                double rpush = (1-alpha)*oldresidue;
                reserve[oldNode]-=alpha*oldresidue;
                negativeFeature[w][oldNode] = 0;
                for( auto newNode : G[oldNode] ){
                    negativeFeature[w][newNode]+=rpush/Degree[newNode];
                    if(negativeFeature[w][newNode] > rmax*rowsum_neg){
                        candidate_set2->Push(newNode);
                    }
                }
            }
            for (int k = 0; k < n; k++){
                double tmp = alpha*(positiveFeature[w][k]-negativeFeature[w][k])+reserve[k];
                if(Degree[k]>0){
                    tmp*=pow(Degree[k],rrr);
                }
                positiveFeature[w][k] = tmp;
            }
            vector<double>().swap(negativeFeature[w]);
            candidate_set1->Clean();
            candidate_set2->Clean();
        }
        delete[] reserve;
    }

    void ppr_push(){
        struct timeval t_start,t_end; 
        double timeCost;
        gettimeofday(&t_start, NULL); 
        vector<thread> threads;
        int ti;
        int start;
        int end = 0;
        for( ti=1 ; ti <= dimension%NUMTHREAD ; ti++ ){
            start = end;
            end+=ceil((double)dimension/NUMTHREAD);
            threads.push_back(thread(&Base::BackwardPush_ppr,this, start, end));
        }
        for( ; ti<=NUMTHREAD ; ti++ ){
            start = end;
            end+=dimension/NUMTHREAD;
            threads.push_back(thread(&Base::BackwardPush_ppr,this, start, end));
        }
        for (int t = 0; t < NUMTHREAD ; t++){
            threads[t].join();
        }
        vector<vector<double>>().swap(negativeFeature);
        vector<thread>().swap(threads);
        gettimeofday(&t_end, NULL); 
        timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
        cout<<dataset<<" pre-computation cost: "<<timeCost<<" s"<<endl;
    }

    void random_walk(int start, int end){
        Node_Set* touch_set = new Node_Set(n);
        vector<vector<double>> reserve(4,vector<double>(n,0));
        for(int it = start ; it < end ; it++ ){
            int root = rwIndex[it];
            for(int i = 0; i < walk_num; i++){
                int curNode = root;
                for(int j = 0 ; j < 4 ; j++ ){
                    int outIndex = rand_max()%Degree[curNode];
                    curNode = G[curNode][outIndex];
                    reserve[j][curNode] += 1.0/walk_num;
                    touch_set->Push(curNode);
                }
            }

            for (int k = 0; k < touch_set->KeyNumber; k++){
                int vv = touch_set->HashKey[k];
                if( reserve[0][vv] > 0 ){
                    ppr1[root].push_back(pair<int,double>(vv,reserve[0][vv]));
                    reserve[0][vv] = 0;
                }
                if( reserve[1][vv] > 0 ){
                    ppr2[root].push_back(pair<int,double>(vv,reserve[1][vv]));
                    reserve[1][vv] = 0;
                }
                if( reserve[2][vv] > 0 ){
                    ppr3[root].push_back(pair<int,double>(vv,reserve[2][vv]));
                    reserve[2][vv] = 0;
                }
                if( reserve[3][vv] > 0 ){
                    ppr4[root].push_back(pair<int,double>(vv,reserve[3][vv]));
                    reserve[3][vv] = 0;
                }
            }
            touch_set->Clean();
        }
    }

    void BackwardPush(int start, int end){
        double* residue = new double[n];
        for( int i = 0 ; i < n ; i++ ){
            residue[i] = 0;
        }
        for(int it = start ; it < end ; it++ ){
            int w = random_w[it];
            double rowsum_pos = positiveRowSum[w];
            double rowsum_neg = negativeRowSum[w];
            ////////// L=1 //////////
            for(int ik = 0 ; ik < n ; ik++ ){
                double old = negativeFeature[w][ik];
                if(old > rmax*rowsum_pos || old < rmax*rowsum_neg/50){
                    for(auto newNode : G[ik]){
                        residue[newNode]+=old/Degree[newNode];
                    }
                    negativeFeature[w][ik] = 0;
                }
                // else{
                //     residue[ik]+=old/Degree[ik];
                // }
            }
            for(auto rwi : rwIndex){
                for(auto j : ppr4[rwi] ){
                    int v = j.first;
                    double pi_r = j.second;
                    positiveFeature[w][rwi]+=pi_r*negativeFeature[w][v];
                }
            }
            for(int ik = 0 ; ik < n ; ik++ ){
                negativeFeature[w][ik] = 0;
            }
            ////////// L=2 //////////
            for(int ik = 0 ; ik < n ; ik++ ){
                double old = residue[ik];
                if(old > rmax*rowsum_pos || old < rmax*rowsum_neg/50){
                    for(auto newNode : G[ik]){
                        negativeFeature[w][newNode]+=old/Degree[newNode];
                    }
                    residue[ik] = 0;
                }
                // else{
                //     negativeFeature[w][ik]+=old/Degree[ik];
                // }
            }
            for(auto rwi : rwIndex){
                for(auto j : ppr3[rwi] ){
                    int v = j.first;
                    double pi_r = j.second;
                    positiveFeature[w][rwi]+=pi_r*residue[v];
                }
            }
            for(int ik = 0 ; ik < n ; ik++ ){
                residue[ik] = 0;
            }
            ////////// L=3 //////////
            for(int ik = 0 ; ik < n ; ik++ ){
                double old = negativeFeature[w][ik];
                if(old > rmax*rowsum_pos || old < rmax*rowsum_neg/50){
                    for(auto newNode : G[ik]){
                        residue[newNode]+=old/Degree[newNode];
                    }
                    negativeFeature[w][ik] = 0;
                }
            }
            for(auto rwi : rwIndex){
                for(auto j : ppr2[rwi] ){
                    int v = j.first;
                    double pi_r = j.second;
                    positiveFeature[w][rwi]+=pi_r*negativeFeature[w][v];
                }
            }
            // for(int ik = 0 ; ik < n ; ik++ ){
            //     negativeFeature[w][ik] = 0;
            // }
            ////////// L=4 //////////
            for(int ik = 0 ; ik < n ; ik++ ){
                double old = residue[ik];
                if(old > rmax*rowsum_pos || old < rmax*rowsum_neg/50){
                    for(auto newNode : G[ik]){
                        positiveFeature[w][newNode]+=old/Degree[newNode];
                    }
                    residue[ik] = 0;
                }
            }
            for(auto rwi : rwIndex){
                for(auto j : ppr1[rwi] ){
                    int v = j.first;
                    double pi_r = j.second;
                    positiveFeature[w][rwi]+=pi_r*residue[v];
                }
            }
            for(int ik = 0 ; ik < n ; ik++ ){
                positiveFeature[w][ik]*=pow(Degree[ik],rrr);
                residue[ik] = 0;
            }

            vector<double>().swap(negativeFeature[w]);
        }
        delete[] residue;
    }

    void rw_push(){
        
        struct timeval t_start,t_end; 
        double timeCost;
        gettimeofday(&t_start, NULL); 
        //MC
        int root_num = rwIndex.size();
        vector<thread> threads;
        int ti;
        int start;
        int end = 0;
        for( ti=1 ; ti <= root_num%NUMTHREAD ; ti++ ){
            start = end;
            end+=ceil((double)root_num/NUMTHREAD);
            threads.push_back(thread(&Base::random_walk,this, start, end));
        }
        for( ; ti<=NUMTHREAD ; ti++ ){
            start = end;
            end+=root_num/NUMTHREAD;
            threads.push_back(thread(&Base::random_walk,this, start, end));
        }
        for (int t = 0; t < NUMTHREAD ; t++){
            threads[t].join();
        }
        vector<thread>().swap(threads);

        //PUSH
        end = 0;
        for( ti=1 ; ti <= dimension%NUMTHREAD ; ti++ ){
            start = end;
            end+=ceil((double)dimension/NUMTHREAD);
            threads.push_back(thread(&Base::BackwardPush,this, start, end));
        }
        for( ; ti<=NUMTHREAD ; ti++ ){
            start = end;
            end+=dimension/NUMTHREAD;
            threads.push_back(thread(&Base::BackwardPush,this, start, end));
        }
        for (int t = 0; t < NUMTHREAD ; t++){
            threads[t].join();
        }
        vector<vector<double>>().swap(negativeFeature);
        vector<thread>().swap(threads);
        gettimeofday(&t_end, NULL); 
        timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
        cout<<dataset<<" pre-computation cost: "<<timeCost<<" s"<<endl;

    }


};


#endif
