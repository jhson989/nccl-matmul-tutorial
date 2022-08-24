#include <vector>
#include <algorithm>
#include <cstdio>
#include <nccl.h>

void sync_all();
void test_result(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C, const int N);
inline void cudaAssert(cudaError_t code, const char *file, int line);
inline void ncclAssert(ncclResult_t code, const char *file, int line);
#define ncclErrChk(ans) { ncclAssert((ans), __FILE__, __LINE__); }
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }


/************************************************************************
  * Kernels
  ************************************************************************/

__global__ void d_matmul(float* A, float* B, float* C, const int HEIGHT, const int WIDTH) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y<HEIGHT && x<WIDTH) {

        float sum = 0.0;
        for (int k=0; k<WIDTH; k++) {
            sum += A[y*WIDTH + k] * B[k*WIDTH + x];
        }
        C[y*WIDTH + x] = sum;
    }
    

}



/************************************************************************
  * Non-local Variable
  ************************************************************************/

int N = 8*1024;
int num_devices = 1;
int* ids_devices;
ncclComm_t* comms;
cudaStream_t* streams;



/************************************************************************
  * Main
  ************************************************************************/

int main (void) {
    
    /************************************************************************
      * Test Configuration
      ************************************************************************/
    
    printf("\n");
    printf("============================================================\n");
    printf("NCCL Tutorial\n");
    printf("- matmul A * B = C\n");
    printf("============================================================\n");

    printf("-- number of devices : ");
    scanf("%d", &num_devices);

    comms = new ncclComm_t[num_devices];
    ids_devices = new int[num_devices];
    streams = new cudaStream_t[num_devices];
    for (int i=0; i<num_devices; i++) {
        ids_devices[i] = i;
    }


    /************************************************************************
      * Test data generation
      ************************************************************************/
    printf("-- Test data gnerating ....\n");
    for (int i=0; i<num_devices; i++) {
        printf("    Device %d\n", i);
        printf("    A[%d ~ %d]\n", N*N/num_devices*(i), N*N/num_devices*(i+1));
        printf("    B[%d ~ %d]\n", 0, N*N);
        printf("    C[%d ~ %d]\n", N*N/num_devices*(i), N*N/num_devices*(i+1));
    }

    std::vector<float> A(N*N);
    std::generate(A.begin(), A.end(), [](){return (rand()%10-5);});
    std::vector<float> B(N*N);
    std::generate(B.begin(), B.end(), [](){return (rand()%10-5);});
    std::vector<float> C(N*N, 0);

    cudaErrChk( cudaHostRegister(A.data(), sizeof(float)*N*N, 0) );
    cudaErrChk( cudaHostRegister(B.data(), sizeof(float)*N*N, 0) );
    cudaErrChk( cudaHostRegister(C.data(), sizeof(float)*N*N, 0) );

    float **d_A, **d_B, **d_C;
    d_A = new float*[num_devices];
    d_B = new float*[num_devices];
    d_C = new float*[num_devices];

    for (int i=0; i<num_devices; i++) {
        cudaErrChk( cudaSetDevice(i) );
        cudaErrChk( cudaStreamCreate(&streams[i]) );
        cudaErrChk( cudaMalloc((void**)&d_A[i], sizeof(float)*N*N/num_devices) );
        cudaErrChk( cudaMalloc((void**)&d_B[i], sizeof(float)*N*N) );
        cudaErrChk( cudaMalloc((void**)&d_C[i], sizeof(float)*N*N/num_devices) );
        cudaErrChk( cudaMemset(d_C[i], 0, sizeof(float)*N*N/num_devices) );
    }

    // Init communications
    ncclErrChk( ncclCommInitAll(comms, num_devices, ids_devices) );


    /************************************************************************
      * Run matmul
      ************************************************************************/
    printf("-- Run matmul kernel ....\n");
    
    // A : Scatter
    ncclErrChk( ncclGroupStart() );
    for (int i=0; i<num_devices; i++) {
        ncclErrChk( ncclSend(A.data()+N*N/num_devices*(i), N*N/num_devices, ncclFloat32, i, comms[i], streams[i]) );
        ncclErrChk( ncclRecv(d_A[i], N*N/num_devices, ncclFloat32, i, comms[i], streams[i]) );
    }
    ncclErrChk( ncclGroupEnd() );

    // B : Broadcast
    ncclErrChk( ncclGroupStart() );
    for (int i=0; i<num_devices; i++) {
        ncclErrChk( ncclBroadcast(B.data(), d_B[i], N*N, ncclFloat32, 0, comms[i], streams[i]) );
    }
    ncclErrChk( ncclGroupEnd() );
    sync_all();

    // C = A * B
    const dim3 dim_threads(16, 16);
    const dim3 dim_blocks(((N)+dim_threads.x-1)/dim_threads.x, ((N/num_devices)+dim_threads.y-1)/dim_threads.y);
    for (int i=0; i<num_devices; i++) {
        cudaErrChk( cudaSetDevice(i) );
        d_matmul<<<dim_blocks, dim_threads, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], N/num_devices, N);
    }
    sync_all();
 
    // C : Gather
    ncclErrChk( ncclGroupStart() );
    for (int i=0; i<num_devices; i++) {
        ncclErrChk( ncclSend(d_C[i], N*N/num_devices, ncclFloat32, i, comms[i], streams[i]) );
        ncclErrChk( ncclRecv(C.data()+N*N/num_devices*(i), N*N/num_devices, ncclFloat32, i, comms[i], streams[i]) );
    }
    ncclErrChk( ncclGroupEnd() );
    sync_all();


    /************************************************************************
      * Test result
      ************************************************************************/
    printf("-- Test the result ....\n");
    test_result(A, B, C, N);


    /************************************************************************
      * Finalize
      ************************************************************************/

    // Free all device memorys
    for (int i=0; i<num_devices; i++) {
        cudaErrChk( cudaSetDevice(i) );
        cudaErrChk( cudaFree(d_A[i]) );
        cudaErrChk( cudaFree(d_B[i]) );
        cudaErrChk( cudaFree(d_C[i]) );
    }

    // Destory communication objects
    for (int i=0; i<num_devices; i++) {
        ncclErrChk( ncclCommDestroy(comms[i]) );
    }

    delete [] comms;
    delete [] ids_devices;
    delete [] streams;

    delete [] d_A;
    delete [] d_B;
    delete [] d_C;


}





/************************************************************************
  * Debug code
  ************************************************************************/

void sync_all() {
    for (int i=0; i<num_devices; i++) {
        cudaErrChk( cudaSetDevice(i) );
        cudaErrChk( cudaStreamSynchronize(streams[i]) );
    }
}

void test_result(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C, const int N) {

    for (int y=0; y<N; y++) {
        for (int x=0; x<N; x++) {
            float sum = 0.0;
            for (int k=0; k<N; k++) {
                sum += A[y*N+k]*B[k*N+x];
            }
            if (sum != C[y*N+x]) {
                printf("  -- Test failed!! C[%d,%d] = %.1f != real(%.1f)\n", y, x, C[y*N+x], sum); 
                return;
            }
        }
    }
    printf("  -- Test passed!!\n"); 


}

inline void cudaAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

inline void ncclAssert(ncclResult_t code, const char *file, int line) {
   if (code != ncclSuccess) {
      fprintf(stderr,"NCCL assert: %s %s %d\n", ncclGetErrorString(code), file, line);
      exit(code);
   }
}



