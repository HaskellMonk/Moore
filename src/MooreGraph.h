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
  int groupNumber(int vertex_number);
  MooreGraph(vector<vector<int> > graph);
 public:
  MooreGraph();

  MooreGraph randomChild();
  int heuristic_score();
  
}; 

