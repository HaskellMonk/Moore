#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "simulated_anneal.h"

#define INITIAL_SAMPLE_SIZE 1000
#define INITIAL_ACCEPT .9
#define ACCEPT_THRESHOLD 1e-7
#define SIMULATED_ANNEAL_P 1.0

double x_hat(double T, double* emax, double* emin, int sample_size) {
  //Calculate the next predicted probability of acceptance based on the current probability
  int i;
  double sum_num = 0;
  double sum_denom = 0;
  for (i = 0; i < sample_size; i++){
    sum_num += exp(-emax[i] / T);
    sum_denom += exp(-emin[i] / T);
  }  
  return sum_num / sum_denom;
}

template <typename T>
double temperature_initial(Optimizer<T> optim){
  //This is an iterative method from Computing the Initial Temperature of Simulated
  //Annealing by Walid Ben-Ameur look at the bottom of page 4
  int i;
  double s0Costs[INITIAL_SAMPLE_SIZE];
  double s1Costs[INITIAL_SAMPLE_SIZE];

  T state2 = optim.initial_state();
  for (i = 0; i < INITIAL_SAMPLE_SIZE;){
    //Get the costs associated with random moves
    T state1 = optim.initial_state(optim);
    s0Costs[i] = optim->cost(optim, state1);
    optim.move(state1, state2);
    s1Costs[i] = optim.cost(state2);
    //Only moves with higher costs
    if (s0Costs[i] < s1Costs[i])
      i++;
  }
  //Need to calculate a good starting point for our method
  double delta_sum = 0.0;
  for (i = 0; i < INITIAL_SAMPLE_SIZE; i++){
    delta_sum += s1Costs[i] - s0Costs[i];
  }
  double x_Hat;
  double t_nplus1;
  double t_n;
  //Do this iteration once just to set some values
  double t_nminus1 = -delta_sum / (INITIAL_SAMPLE_SIZE * log(INITIAL_ACCEPT));

  double p = SIMULATED_ANNEAL_P;
  x_Hat = x_hat(t_nminus1, s0Costs, s1Costs, INITIAL_SAMPLE_SIZE);
  t_n = -t_nminus1 * pow(log(x_Hat) / log(INITIAL_ACCEPT), 1 / p);
  
  while (1) {
    x_Hat = x_hat(t_n, s0Costs, s1Costs, INITIAL_SAMPLE_SIZE);
    if (abs(x_Hat - INITIAL_ACCEPT) <= ACCEPT_THRESHOLD)
      return t_n;
    t_nplus1 = -t_n * pow(log(x_Hat) / log(INITIAL_ACCEPT), 1. / p);

    if ((t_nplus1 - t_n) * (t_n - t_nminus1) < 0.0) {
      //If there is oscillation in the temperature then we need a larger value for p
      p = 2 * p; 
    }
    //begin next iteration
    t_nminus1 = t_n;
    t_n = t_nplus1;
  }
}

double acceptance_probability(double old_e, double new_e, double t){
  /*
    Calculate the acceptance probability corresponding to the Metropolis-Hastings algorithm
  */
  if (new_e < old_e)
    return 1;
  else
    return exp(-(new_e - old_e) / t);
}

double temperature_cauchy(double i, double t0){
  double tr = t0 /(1.0 + i);
  return tr;
}


double temperature_boltzmann(double i, double t0){
  return t0 / log(1 + i);
}

template <typename T>
T* simmulated_anneal(Optimizer<T> &optim, int max_iterations){
  int i;
  //initialize the random number generator
  srand48(time(NULL));
  srand(time(NULL));
  
  //Allocate space for some temporary states
  T current_state = optim.initial_state();
  T best_state = optim.initial_state();
  best_state = current_state;
  T new_state = optim.initial_state();
  //Evaluate the initial energy level
  double current_energy = optim.cost(current_state);
  double best_energy = current_energy;
  double t0 = temperature_initial(optim);

  double t_i;
  double new_energy;
  //Iterate for a maximum number of iterations
  for (i = 0; i < max_iterations; i++){
    t_i = optim.temperature(i, max_iterations, t0);
    current_energy = optim.cost(current_state);
    optim.move(current_state, new_state);
    new_energy = optim.cost(new_state);
    if (acceptance_probability(current_energy, new_energy, t_i) > 
	drand48()) {
      optim.copy_state(new_state, current_state);
      current_energy = new_energy;
    }
    if (current_energy < best_energy){
      optim.copy_state(new_state, best_state);
      best_energy = new_energy;
    }
  }
  return best_state;
}

