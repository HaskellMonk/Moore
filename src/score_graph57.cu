#define K 57
#define N K * K + 1

__global__ void vertex_heuristic(int* graph_tbl, 
				 int* heuristic_value){
  //We parallelize over the vertices
  int i;

  int vertex_number = blockIdx.x * blockDim.x + threadIdx.x;

  //zero local lookup table
  int lookup[N];
  for (i = 0; i < N; i++)
    lookup[i] = 0;

  //add up each ajacency
  for (i = 0; i < K; i++){
    int adjacent = graph_tbl[vertex_number * K + i];
    lookup[adjacent]++;

    for (j = 0; j < K; j++){
      lookup[graph_tbl[adjacent * K + j]]++;
    }
  }
  
  //Sumarize the lookup table
  int h_value = 0;
  for (i = 0; i < N; i++){
    h_value += abs(lookup[i] - 1);
  }
  //Subtract out our value.
  h_value--;
  heuristic_value[vertex_number] = h_value;
  
}

			  


