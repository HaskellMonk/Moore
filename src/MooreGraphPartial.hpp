#include <tuple>
#include <vector>


using std::vector;
using std::tuple;

template<int MT>
class MooreGraphPartial {
  vector<vector<int> > adjacency_list;
  
public:
  MooreGraphPartial();
  void addEdge(tuple<int, int> &edge);
  int conflicts(int vertex);
  int degree(int vertex);
  void removeEdge(tuple<int, int> &edge);
  void popEdge(tuple<int, int> &edge);
  void pushEdge(tuple<int, int> &edge);
  void belowDegree(vector<tuple<int,int> > &verts);
}
