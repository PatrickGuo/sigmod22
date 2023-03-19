#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

// For filters: pad are all 0, kernel_w = width, num_filters should be filled.
template <typename T>
std::pair<int, int> im2col(const T *img, T *col, int width, int height, int channels,
    int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h, int num_filters) {

  // Data col matrix dimension: channels_col * (height_col * width_col)
  // Conv filter col matrix dimension: numFilters * channels_col
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1; // for conv filters, h_col=1
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1; // for conv filters, w_col=1
  int channels_col = channels * kernel_h * kernel_w;

  // Filter follows NCHW format.
  for (int f = 0; f < num_filters; ++f) {
    for (int c = 0; c < channels_col; ++c) {
      int w_offset = c % kernel_w;
      int h_offset = (c / kernel_w) % kernel_h;
      int c_im = c / (kernel_h * kernel_w);
  
      for (int h = 0; h < height_col; ++h) {
        for (int w = 0; w < width_col; ++w) {
          int h_pad = h*stride_h - pad_h + h_offset;
          int w_pad = w*stride_w - pad_w + w_offset;
          if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
            col[f*channels_col + (c*height_col+h) * width_col + w] =
              img[f*channels_col + (c_im * height + h_pad) * width + w_pad];
          } else {
            col[f*channels_col + (c*height_col+h) * width_col + w] = 0;
          }
        }
      }
    }
  }
  // for data, num_filters = 1, for filters, w_col * h_col = 1
  return {channels_col, width_col * height_col * num_filters};
}

// DeCor : Conv2D diff prop algorithm.
void Conv2D::diff(){
  // 1. calculate eigen for each op; 2. calc avg; 3. diff = avg * parentop.difference.

	// --- Setting the host matrix
	int M = inputs[1].dim[0]; // number of filters
  int N = inputs[1].dim[1] * inputs[1].dim[2] * inputs[1].dim[3]; // c*h*w
  float *h_A = (float *)malloc(M * N * sizeof(float));
    
	// --- Setting the device matrix and moving the host matrix to the device
  float *d_A;           checkCUDA(cudaMalloc(&d_A, M * N * sizeof(float)));
	cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);

  // --- host side SVD results space
  int num_sgv = std::min(M, N);
	float *h_S = (float *)malloc(num_sgv * sizeof(float));

	// --- device side SVD workspace and matrices
	int *devInfo;       checkCUDA(cudaMalloc(&devInfo, sizeof(int)));
  float *d_S;         checkCUDA(cudaMalloc(&d_S, num_sgv * sizeof(float)));

	// --- CUDA solver initialization
  // cusolverDnHandle_t solver is initialized in Model (ops_cudnn.cu)
  int work_size = 0;
	checkCUDA(cusolverDnSgesvd_bufferSize(model->solver, M, N, &work_size));
	float *work;    checkCUDA(cudaMalloc(&work, work_size * sizeof(float)));

	// --- CUDA SVD execution - Singular values only
	checkCUDA(cusolverDnSgesvd(model->solver, 'N', 'N', M, N, d_A, M, d_S, NULL, M, NULL, N, work, work_size, NULL, devInfo));

	int devInfo_h = 0;
	checkCUDA(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_h == 0)
		printf("SVD successfull for the singular values calculation only\n\n");
	else if (devInfo_h < 0)
		printf("SVD unsuccessfull for the singular values calculation only. Parameter %i is wrong\n", -devInfo_h);
	else
		printf("SVD unsuccessfull for the singular values calculation only. A number of %i superdiagonals of an intermediate bidiagonal form did not converge to zero\n", devInfo_h);

	// --- Moving the results from device to host
	checkCUDA(cudaMemcpy(h_S, d_S, num_sgv * sizeof(float), cudaMemcpyDeviceToHost));
  
  // check these sgv into the difference.
  float total_diff = 0;
  for (int i = 0; i < num_sgv; i++) {
    total_diff += h_S[i];
  }
  difference = total_diff / (float)num_sgv * inputs[0].op.ptr->difference;
}

void Matmul::diff() {
  // --- Setting the host matrix
  int numDim = outputs[0].numDim;
	int M = inputs[0].dim[numDim-1]; // num_rows of the weight matrix = num_cols of the data mat.
  int N = inputs[1].dim[numDim-1]; // num_cols of weight mat.
  float *h_A = (float *)malloc(M * N * sizeof(float));
    
	// --- Setting the device matrix and moving the host matrix to the device
  float *d_A;           checkCUDA(cudaMalloc(&d_A, M * N * sizeof(float)));
	cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);

  // --- host side SVD results space
  int num_sgv = std::min(M, N);
	float *h_S = (float *)malloc(num_sgv * sizeof(float));

	// --- device side SVD workspace and matrices
	int *devInfo;       checkCUDA(cudaMalloc(&devInfo, sizeof(int)));
  float *d_S;         checkCUDA(cudaMalloc(&d_S, num_sgv * sizeof(float)));

	// --- CUDA solver initialization
  // cusolverDnHandle_t solver is initialized in Model (ops_cudnn.cu)
  int work_size = 0;
	checkCUDA(cusolverDnSgesvd_bufferSize(model->solver, M, N, &work_size));
	float *work;    checkCUDA(cudaMalloc(&work, work_size * sizeof(float)));

	// --- CUDA SVD execution - Singular values only
	checkCUDA(cusolverDnSgesvd(model->solver, 'N', 'N', M, N, d_A, M, d_S, NULL, M, NULL, N, work, work_size, NULL, devInfo));

	int devInfo_h = 0;
	checkCUDA(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_h == 0)
		printf("SVD successfull for the singular values calculation only\n\n");
	else if (devInfo_h < 0)
		printf("SVD unsuccessfull for the singular values calculation only. Parameter %i is wrong\n", -devInfo_h);
	else
		printf("SVD unsuccessfull for the singular values calculation only. A number of %i superdiagonals of an intermediate bidiagonal form did not converge to zero\n", devInfo_h);

	// --- Moving the results from device to host
	checkCUDA(cudaMemcpy(h_S, d_S, num_sgv * sizeof(float), cudaMemcpyDeviceToHost));
  
  // check these sgv into the difference.
  float total_diff = 0;
  for (int i = 0; i < num_sgv; i++) {
    total_diff += h_S[i];
  }
  difference = total_diff / (float)num_sgv * inputs[0].op.ptr->difference;
}



