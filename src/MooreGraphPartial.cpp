#include "MooreGraphPartial.hpp"
#include <tuple>

using std::tuple;

template<int MT>
MooreGraphPartial::MooreGraphPartial(){
  adjacency_list = vector<vector<int> >(MT * MT + 1,
					vector<int>());
  //We know this much at least about the graph
  //0 goes to 1,2,3...
  for (int i = 0; i < MT; i++) {
    adjacency_list[0].push_back(i + 1);
    adjacency_list[i + 1].push_back(0);
  }

  int prefix = MT + 1;
  for (int i = 0; i < MT; i++) { 
    for (int j = 0; j < MT - 1; j++) { 
      adjacency_list[1 + i].push_back(prefix + i * (MT - 1) + j);
      adjacency_list[prefix + i * (MT - 1) + j].push_back(1 + i);
    }
  }
    
  int prefix = MT + 1;
  for (int i = 0; i < MT - 1; i++) { 
    for (int j = 0; j < MT - 1; j++) { 
      adjacency_list[prefix + i].push_back(prefix + (j + 1) * (MT - 1) + i);
      adjacency_list[prefix + (j + 1) * (MT - 1) + i].push_back(prefix + i);
    }
  }
}

template<int MT>
void MoorGraphPartial<MT>::pushEdge(tuple<int, int> &edge){
  adjacency_list[edge<0>.get()].push_back(edge<1>.get());
  adjacency_list[edge<1>.get()].push_back(edge<0>.get());
}

template<int MT>
void MoorGraphPartial<MT>::popEdge(tuple<int, int> &edge){
  //order 1
  adjacency_list[edge<0>.get()].pop_back();
  adjacency_list[edge<1>.get()].pop_back();
}

template<int MT>
int MoorGraphPartial<MT>::degree(int vertex){
  return adjacency_list[vertex].size();
}

template<int MT>
void 
MooreGraphPartial::belowDegree(vector<tuple<int,int> > &verts){
  for (int i = 0; i < MT * MT + 1; i++){
    int degree = adjacency_list[i].size();
    if (degree < MT)
      verts.push_back(tuple<int, int>(vert_size, i));
  }
}

template<int MT>
void MoorGraphPartial<MT>::removeEdge(tuple<int, int> &edge){
  //This is order n
  std::remove(adjacency_list[edge<0>.get()].begin(),
	      adjacency_list[edge<0>.get()].end(),
	      edge<1>.get());
  std::remove(adjacency_list[edge<1>.get()].begin(),
	      adjacency_list[edge<1>.get()].end(),
	      edge<0>.get());
}

template<int MT>
int MoorGraphPartial<MT>::conflicts(int vertex){
  vector<int> lookup_table(MT * MT + 1, 0);
  for (int i = 0; i < MT; i++){
    int adj_vert = adjacency_list[vertex][i];
    lookup_table[adj_vert] += 1;
    for (int j = 0; j < MT; j++){
      lookup_table[adjacency_list[adj_vert][j]] += 1;
    }
  }
  int conflicts = -MT;
  for (int i = 0; i < MT; i++){
    if (lookup_table[i] > 1)
      conflicts += lookup_table[i] - 1;
  }
  return conflicts;
}
