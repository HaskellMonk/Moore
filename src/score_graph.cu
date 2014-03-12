#define K 57
#define N K * K + 1

__global__ void heuristic_grade(int* graph_tbl, 
				int* heuristic_value, 
				int n){
  int i;

  //zero local lookup table
  __local__ int lookup[N];
  for (i = 0; i < N; i++)
    lookup[i] = 0;

  //add up each ajacency
  for (i = 0; i < K - 1; i++){
    int adjacent = graph_tbl[blockId.x * (K - 1) + i];
    lookup[adjacent]++;

    for (j = 0; j < K - 1; j++){
      lookup[graph_tbl[adjacent * (K - 1) + j]]++;
    }
  }
  
  //Sumarize the lookup table
  int h_value = 0;
  for (i = 0; i < N; i++){
    h_value += abs(lookup[i] - 1);
  }
  //Subtract out our value.
  h_value--;
  heuristic_value[blockId.x] = h_value;
}

__global__ void 


