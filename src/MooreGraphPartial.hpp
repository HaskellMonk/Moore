#include <tuple>
#include <vector>
using std::vector;

template<int MT>
class MooreGraphPartial {
  vector<vector<int> > adjacency_list;
public:
  void addEdge(tuple<int, int> edge);
  
}
