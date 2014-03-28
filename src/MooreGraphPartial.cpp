#include <tuple>
#include "MooreGraphPartial.hpp"

using std::tuple;

template<int MT>
void MoorGraphPartial<MT>::pushEdge(tuple<int, int> edge){
  adjacency_list[edge<0>.get()].push_back(edge<1>.get());
  adjacency_list[edge<1>.get()].push_back(edge<0>.get());
}

template<int MT>
void MoorGraphPartial<MT>::popEdge(tuple<int, int> edge){
  adjacency_list[edge<0>.get()].pop_back();
  adjacency_list[edge<1>.get()].pop_back();
}


template<int MT>
void MoorGraphPartial<MT>::removeEdge(tuple<int, int> edge){
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

template<int MT>
int MooreGraphPartial<MT>::mostConstrained(){
  
}
