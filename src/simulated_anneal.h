template <typename T>
class Optimizer {
 public:
  //Information the program needs
  T needed_args;

  //Cost of the state
  double cost(T &state);
  
  //Make a new uninitialized state.
  T initial_state();
  T operator=(T &lhs, T &rhs);
  T move(T &in, T &out);
  double temperature(int, int, double);
};

template <typename T>
T simmulated_anneal(Optimizer<T> optim, int max_iterations);

double temperature_cauchy(double i, double t0);
double temperature_boltzmann(double i, double t0);
