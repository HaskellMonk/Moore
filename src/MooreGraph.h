#include <vector>
enum MooreType {
  MOORE3 = 3,
  MOORE7 = 7,
  MOORE57 = 57
};

template<MooreType MT>
class MooreGraph {
  std::vector<int> graph_data;
  void* device_data;
  
  ~MooreGraph();
 public:
  MooreGraph();
  MooreGraph bestChild();
  MooreGraph bestChildren(int n);
  MooreGraph bestOfSample(int n);
}; 
