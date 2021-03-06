#include <random>
#include <cmath>
#include "MooreGraph.h"

template <int MT>
MooreGraph::MoreGraph(vector<vector<int> > vect){
  graph_data = vect;
}

template <int MT>
MooreGraph::MooreGraph(){
  //Starting at vertex 0 has a link to vertices 1 to 57
  for (int i = 0; i < MT; i++) {
    graph_data[0].push_back(i + 1);
    graph_data[i].push_back(0);
  }

  for (int i = 0; i < MT; i++) { 
    for (int j = 0; j < MT - 1; j++) { 
      //Vertex 1 to 57 have edges to vertex 58 to 58 + 57^2
      graph_data[i + 1].push_back(MT + 1 + i * (MT - 1) + j);
      //Vertex 58 to 58 + 57^2 have edges with 1 to 57
      graph_data[MT + 1 + i * (MT - 1) + j].push_back(i + 1);
    }
  }
  
  //Generating random k-regular graphs is very hard to do 
  //so I generate a graph with approximately the structure
  //I think it should have at the beginning.

  //For groups at the base of which there are 57
  int init_index = MT + 1;
  for (int i = 0; i < MT; i++) { 
    int group_index = init_index + (MT - 1) * i;
    //For vertices in each group which are of size 56
    for (int j = 0; j < MT - 1; j++) { 
      int vertex_index = group_index + j;
      //For edge connections in each vertex
      for (int k = 1 + i; k < MT; k++){
	int edge_index = init_index + k * (MT - 1) + j;
	//Each each of the 56 edges connect to one vertex in 
	//each group
	graph_data[vertex_index].push_back(edge_index);
      }
    }
  }
}

template <int MT>
MooreGraph::~MooreGraph(){
}

template <int MT>
int MooreGraph::groupNumber(int vertex_number){
  if (vertex_number < (MT + 1))
    return -1;
  else
    return (vertex_number - 1 - MT) % (MT - 1);
}

template<int MT>
MooreGraph& MooreGraph::operator=(MooreGraph &rhs){
  if (&rhs == this)
    return *this;
  else {
    for (int i = 0; i < graph_data.size(); i++){
      std::copy(graph_data[i].begin(), 
		graph_data[i].end(),
		rhs[i].graph_data.begin());
    }
  }
  return rhs:
}

template <int MT>
void MooreGraph::apply_delta(tuple<int, int, int, int> &delta){
  //Copy to new graph
  //Randomization scheme for edges
  //There is a bijective mapping of vertices in group i 
  //to vertices in group j. 
  //We pick two vertices u and v in a group i and then pick a group
  //j. Edges (u,x) and (v,y) will exist where x and y are in group j.
  //Delete edges (u,x) and (v,y). Then add edges (u,y) and (v,x).

  //First pick group i
  std::default_random_engine generator;
  int group_i = std::uniform_int_distribution<int>(0, MT - 1)(generator);
  
  //Now pick two vertices in group i
  int vertex_1 = std::uniform_int_distribution<int>(0, MT - 2)(generator);
  int vertex_2 = std::uniform_int_distribution<int>(0, MT - 3)(generator);
  
  //Now Pick group j
  int group_j = std::uniform_int_distribution<int>(0, MT - 2)(generator);
  
  if (vertex_1 <= vertex_2)
    vertex_2 += 1;

}

template<int MT>
int heuristicScore(){
  vector<int> lookup_table(MT * MT + 1, 0);

  //We start off with a negative value since we overcount
  //by one on each vertex.
  int heuristic_value = -(MT * MT + 1);
  //Go through all vertices
  for (int i = 0; i < MT * MT + 1; i++){
    //Go through all vertices that are adjacent to this vertex
    for (int j = 0; j < MT; j++){
      int adj_vert = graph_data[i][j];
      lookup_table[adj_vert] += 1;
      for (int k = 0; k < MT; k++){
	lookup_table[graph_data[adj_vert][k]] += 1;
      }
    }
    for (int j = 0; j < MT * MT + 1; j++){
      heuristic_value += pow(lookup_table[j] - 1, 2);
    }
  }
  return heuristic_value;
}

