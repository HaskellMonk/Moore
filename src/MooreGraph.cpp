#include "MooreGraph.h"
#include <random>

template <MooreType MT>
MooreGraph::MooreGraph(){
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0,MT - 1);
  std::vector vec(100, 0);
  for (int i = 0; i < 100; i++){
    vec[i] = distribution(generator);
  }
}

template <MooreType MT>
MooreGraph::~MooreGraph(){
}

template <MooreType MT>
std::vector<MooreGraph> bestChild(){
}

template <MooreType MT>
std::vector<MooreGraph> bestChildren(int n){

}

template <MooreType MT>
MooreGraph bestOfSample(int n){

}
