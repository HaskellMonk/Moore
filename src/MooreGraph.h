#include <vector>
using std::vector;

enum MooreType {
  MOORE3 = 3,
  MOORE7 = 7,
  MOORE57 = 57
};

template<MooreType MT>
class MooreGraph {
  //Empty adjacency list
  vector<vector<int> > graph_data(MT * MT + 1, vector<int>());
  ~MooreGraph();
 public:
  MooreGraph();
  
  MooreGraph bestChild();
  MooreGraph bestChildren(int n);
  MooreGraph bestOfSample(int n);
  int heuristic_score();
}; 

