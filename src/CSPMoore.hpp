#include <queue>
#include <vector>
#include "MooreGraphPartial.hpp"

using std::vector;
using std::priority_queue;

template<int MT>
int constraintSolver(){
  MooreGraphPartial part;
  //Each vertex has so many possibilities
  typedef tuple<int, int> P;
  priority_queue<P> minValues([](P &a1, P &a2){
      return a1<0>.get() < a2<0>.get();
    });
  
  //The list of variables that we have to play with
  vector<tuple<int,int> > vect;
  part.belowDegree(vect);
  for (auto it = vect.begin(); it != vect.end(); it++){
    minValues.push(*it);
  }
  
}
