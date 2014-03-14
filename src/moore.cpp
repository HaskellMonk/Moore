#include <vector>
#include <iostream>


int number_of_vertices(int degree){
  return degree * degree + 1;
}

int array_size(int degree){
  return (degree * degree + 1) * degree;
}

int main(int argc, char* argv[]){
  
  const int[] moore_graphs = {3, 7, 57};
  int degree;
  if ((argc == 2) && 
      (atoi(argv[1]) < 4) && 
      (atoi(argv[1]) > -1)) {
    degree = argv[atoi(argv[1])];
  }
  else if (argc == 1) {
    degree = 7;
  } else {
    std::cout << "Please enter 0, 1, or 2 for the moore graphs 3, 7, " <<
      " and 57." << std::endl;
    return 0;
  }
  
  vector<int> initial_graph(array_size(degree), 0);

  
  void* devArray;
  // cudaMalloc the memory to hold the graph representation
  cudaMalloc(&devArray, array_size(degree) * sizeof(int));

  // zero out the device array with cudaMemset
  //cudaMemset(devArray, 0, array_size(degree) * sizeof(int));

  //
  cudaMemcpy(host_array, device_array, num_bytes, 
	     cudaMemcpyDeviceToHost);

  // print out the result element by element
  for(int i = 0; i < num_elements; ++i)
    printf("%d ", host_array[i]);

  // use free to deallocate the host array
  free(host_array);

  // use cudaFree to deallocate the device array
  cudaFree(device_array);

  return 0;
}
