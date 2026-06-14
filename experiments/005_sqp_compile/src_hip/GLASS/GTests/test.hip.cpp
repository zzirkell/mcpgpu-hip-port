#include "hip/hip_runtime.h"
#include <iostream>
#include <hip/hip_runtime.h>

#include "../glass.hip.hpp"
#include "./global_glass.hip.hpp"
#include "gtest/gtest.h"


class L1Test : public ::testing::Test{

	protected:
		void SetUp() override {
			n = 100;
			h_a = new int[n];
			h_b = new int[n];
			h_c = new int;
			for(int i = 0; i < n; i++){
					h_a[i] = i;
					h_b[i] = 2 * i;
			}
			hipMalloc(&d_a, n * sizeof(int));
			hipMalloc(&d_b, n * sizeof(int));
			hipMalloc(&d_c, n * sizeof(int));
			hipMemcpy(d_a, h_a, n * sizeof(int), hipMemcpyHostToDevice);
			hipMemcpy(d_b, h_b, n * sizeof(int), hipMemcpyHostToDevice);
			hipDeviceSynchronize();
		}
		void TearDown() override {
			// Code here will be called immediately after each test (right
			// before the destructor).
			hipFree(d_a);
			hipFree(d_b);
			hipFree(d_c);
			delete h_a;
			delete h_b;
			delete h_c;
		}

	int n;
	int * h_a;
	int * h_b;
	int * h_c;
	int * d_a, *d_b, *d_c;
};

class L2Test : public ::testing::Test{

	protected:
		void SetUp() override {
			m = 5;
			n = 7;
			h_a = new int[m*n];
			h_b = new int[n];
			h_c = new int[m];
			for(int i = 0; i < m*n; i++){
					h_a[i] = i;
			}
			for (size_t i = 0; i < n; i++)
			{
				h_b[i] = 2 * i;
			}
			hipMalloc(&d_a, m*n * sizeof(int));
			hipMalloc(&d_b, n * sizeof(int));
			hipMalloc(&d_c, m * sizeof(int));
			hipMemcpy(d_a, h_a, m*n * sizeof(int), hipMemcpyHostToDevice);
			hipMemcpy(d_b, h_b, n * sizeof(int), hipMemcpyHostToDevice);
			hipDeviceSynchronize();
		}
		void TearDown() override {
			// Code here will be called immediately after each test (right
			// before the destructor).
			hipFree(d_a);
			hipFree(d_b);
			hipFree(d_c);
			delete h_a;
			delete h_b;
			delete h_c;
		}

	int n, m;
	int * h_a;
	int * h_b;
	int * h_c;
	int * d_a, *d_b, *d_c;
};

class L3Test : public ::testing::Test{

	protected:
		void SetUp() override {
			m = 5;
			n = 4;
			k = 3;
			h_a = new int[m*n];
			h_b = new int[n*k];
			h_c = new int[m*k];
			for(int i = 0; i < m*n; i++){
					h_a[i] = i;
			}
			for(int i = 0; i < n*k; i++){
					h_b[i] = 2 * i;
			}
			hipMalloc(&d_a, m*n * sizeof(int));
			hipMalloc(&d_b, n*k * sizeof(int));
			hipMalloc(&d_c, m*k * sizeof(int));
			hipMemcpy(d_a, h_a, m*n * sizeof(int), hipMemcpyHostToDevice);
			hipMemcpy(d_b, h_b, n*k * sizeof(int), hipMemcpyHostToDevice);
			hipDeviceSynchronize();
		}
		void TearDown() override {
			// Code here will be called immediately after each test (right
			// before the destructor).
			hipFree(d_a);
			hipFree(d_b);
			hipFree(d_c);
			delete h_a;
			delete h_b;
			delete h_c;
		}

	int n, m, k;
	int * h_a, *h_b, *h_c;
	int * d_a, *d_b, *d_c;
};

class L3InvTest : public ::testing::Test{

	protected:
		void SetUp() override {
			m = 5;
			h_a = new double[2*m*m] {
				10, 2,  4,  5,  3,
				11, 6,  12, 7,  13,
				8,  9,  14, 15, 16,
				17, 18, 19, 20, 21,
				22, 23, 24, 25, 26,
			};

			hipMalloc(&d_a, 2*m*m * sizeof(*d_a));
			hipMalloc(&d_b, 2*m*m * sizeof(*d_b));
			hipMalloc(&d_temp, (2*m + 1) * sizeof(*d_temp));
			hipMemcpy(d_a, h_a, 2*m*m * sizeof(*d_a), hipMemcpyHostToDevice);
			hipDeviceSynchronize();
		}
		void TearDown() override {
			hipFree(d_a);
			hipFree(d_temp);
			delete h_a;
		}

	int m;
	double *h_a;
	double *d_a, *d_b;
	double *d_temp;
};

TEST_F(L1Test, DotProduct){
	global_dot<<<1, n>>>(d_c, n, d_a, d_b);
	hipDeviceSynchronize();
	// copy the memory back
	hipMemcpy(h_c, d_c, sizeof(int), hipMemcpyDeviceToHost);
	EXPECT_EQ(*h_c, 656700);
}

TEST_F(L1Test, DotProductMultiBlock){
	global_dot<<<dim3(2,2,2), dim3(2,2,2)>>>(d_c, n, d_a, d_b);
	hipDeviceSynchronize();
	// copy the memory back
	hipMemcpy(h_c, d_c, sizeof(int), hipMemcpyDeviceToHost);
	EXPECT_EQ(*h_c, 656700);
}

TEST_F(L1Test, axpy) {
	int res[n];
	int alpha = n;

	for (int i=0; i<n; i++) {
		res[i]= h_a[i]*alpha+h_b[i];
	}
	global_axpy<<<1,n>>>(n, alpha, d_a, d_b);
	hipDeviceSynchronize();
	hipMemcpy(h_b, d_b, n*sizeof(int), hipMemcpyDeviceToHost);
	for (int i=0; i<n; i++){
		EXPECT_EQ(h_b[i], res[i]);
	}
}

TEST_F(L1Test, clip) {
	global_clip<<<1,n>>>(n, d_a, d_b, d_b);
	hipDeviceSynchronize();
	hipMemcpy(h_a, d_a, n*sizeof(int), hipMemcpyDeviceToHost);
	for (int i=0; i<n; i++){
		EXPECT_EQ(h_a[i], h_b[i]);
	}
}

TEST_F(L1Test, copy) {
	global_copy<<<1,n>>>(n, d_a, d_b);
	hipDeviceSynchronize();
	hipMemcpy(h_b, d_b, n*sizeof(int), hipMemcpyDeviceToHost);
	for (int i=0; i<n; i++){
		EXPECT_EQ(h_a[i], h_b[i]);
	}
}

TEST_F(L1Test, copyMultiBlock) {
	global_copy<<<dim3(2,2,2), dim3(2,2,2)>>>(n, d_a, d_b);
	hipDeviceSynchronize();
	hipMemcpy(h_b, d_b, n*sizeof(int), hipMemcpyDeviceToHost);
	for (int i=0; i<n; i++) {
		EXPECT_EQ(h_a[i], h_b[i]);
	}
}

TEST_F(L1Test, scaledCopy) {
	int alpha = 4;

	global_copy<<<1,n>>>(n, alpha, d_a, d_b);
	hipDeviceSynchronize();
	hipMemcpy(h_b, d_b, n*sizeof(int), hipMemcpyDeviceToHost);
	for (int i=0; i<n; i++) {
		EXPECT_EQ(alpha*h_a[i], h_b[i]);
	}
}

TEST_F(L1Test, loadIdentity) {
	int dim = (int)sqrt(n);
	global_loadIdentity<<<1,n>>>(dim, d_a);
	hipDeviceSynchronize();
	hipMemcpy(h_a, d_a, n*sizeof(int), hipMemcpyDeviceToHost);
	for (int i=0; i<n; i++) {
		EXPECT_EQ((i%dim == i/dim), h_a[i]);
	}
}

TEST_F(L1Test, loadIdentity2) {
	int dim = (int)sqrt(n);
	global_loadIdentity<<<1,n>>>(dim, d_a, dim, d_b);
	hipDeviceSynchronize();
	hipMemcpy(h_a, d_a, n*sizeof(int), hipMemcpyDeviceToHost);
	hipMemcpy(h_b, d_b, n*sizeof(int), hipMemcpyDeviceToHost);
	for (int i=0; i<n; i++) {
		EXPECT_EQ((i%dim == i/dim), h_a[i]);
		EXPECT_EQ((i%dim == i/dim), h_b[i]);
	}
}

TEST_F(L1Test, addI) {
	int dim = (int)sqrt(n);
	int alpha = 4;
	int res[n];

	for (int i=0; i<n; i++) {
		if (i%dim == i/dim)
			res[i] = h_a[i] + alpha;
		else
			res[i] = h_a[i];
	}
	global_addI<<<1,n>>>(dim, d_a, alpha);
	hipDeviceSynchronize();
	hipMemcpy(h_a, d_a, n*sizeof(int), hipMemcpyDeviceToHost);
	for (int i=0; i<n; i++){
		EXPECT_EQ(res[i], h_a[i]);
	}
}

TEST_F(L1Test, infnorm) {
	global_infnorm<<<1,n>>>(n, d_b);
	hipDeviceSynchronize();
	hipMemcpy(h_b, d_b, sizeof(int), hipMemcpyDeviceToHost);
	EXPECT_EQ(198, h_b[0]);
}

TEST_F(L1Test, reduce) {
	int expected_sum = 0;
	for (int i = 0; i < n; i++) { expected_sum += h_a[i]; }
	global_reduce<<<1, n>>>(n, d_a);
	hipDeviceSynchronize();
	hipMemcpy(h_a, d_a, sizeof(*h_a), hipMemcpyDeviceToHost);
	EXPECT_EQ(h_a[0], expected_sum);
}

TEST_F(L1Test, l2norm) {
	int expected_sum = 0;
	for (int i = 0; i < n; i++) { expected_sum += h_a[i] * h_a[i]; }
	global_l2norm<<<1, n>>>(n, d_a);
	hipDeviceSynchronize();
	hipMemcpy(h_a, d_a, sizeof(*h_a), hipMemcpyDeviceToHost);
	EXPECT_EQ(h_a[0], floor(sqrtf(expected_sum)));
}

TEST_F(L1Test, scal) {
	int expected[n];
	for (int i = 0; i < n; i++) { expected[i] = h_a[i] * 2; }
	global_scal<<<1, n>>>(n, 2, d_a);
	hipDeviceSynchronize();
	hipMemcpy(h_a, d_a, sizeof(*h_a) * n, hipMemcpyDeviceToHost);
	for (int i = 0; i < n; i++) {
		EXPECT_EQ(h_a[i], expected[i]);
	}
}

TEST_F(L1Test, setConst) {
	global_set_const<<<1, n>>>(n, n, d_a);
	hipDeviceSynchronize();
	hipMemcpy(h_a, d_a, sizeof(*h_a) * n, hipMemcpyDeviceToHost);
	for (int i = 0; i < n; i++) {
		EXPECT_EQ(h_a[i], n);
	}
}

TEST_F(L1Test, swap) {
	int expected_a[n], expected_b[n];
	for (int i = 0; i < n; i++) {
		expected_a[i] = h_b[i];
		expected_b[i] = h_a[i];
	}
	global_swap<<<1,n>>>(n, 1, d_a, d_b);
	hipDeviceSynchronize();
	hipMemcpy(h_a, d_a, sizeof(*h_a) * n, hipMemcpyDeviceToHost);
	hipMemcpy(h_b, d_b, sizeof(*h_b) * n, hipMemcpyDeviceToHost);
	for (int i = 0; i < n; i++) {
		EXPECT_EQ(h_a[i], expected_a[i]);
		EXPECT_EQ(h_b[i], expected_b[i]);
	}
}

TEST_F(L2Test, gemv){
	int res[] = {910,952,994,1036,1078};
	int res_transpose[] = {182,476,770,1064,1358};
	global_gemv<int, false><<<1, 64>>>(m, n, static_cast<int>(1), d_a, d_b, d_c);
	hipDeviceSynchronize();
	hipMemcpy(h_c, d_c, m*sizeof(int), hipMemcpyDeviceToHost);
	for(int i=0; i<m; i++){
		EXPECT_EQ(h_c[i], res[i]);
	}

	// transpose
	global_gemv<int, true><<<1, 64>>>(m, n, static_cast<int>(1), d_a, d_b, d_c);
	hipDeviceSynchronize();
	hipMemcpy(h_c, d_c, m*sizeof(int), hipMemcpyDeviceToHost);
	for(int i=0; i<m; i++){
		EXPECT_EQ(h_c[i], res_transpose[i]);
	}
}

TEST_F(L2Test, gemvMultiBlock){
	int res[] = {910,952,994,1036,1078};
	int res_transpose[] = {182,476,770,1064,1358};
	global_gemv<int, false><<<dim3(2,2,2), dim3(2,2,2)>>>(m, n, static_cast<int>(1), d_a, d_b, d_c);
	hipDeviceSynchronize();
	hipMemcpy(h_c, d_c, m*sizeof(int), hipMemcpyDeviceToHost);
	for(int i=0; i<m; i++){
		EXPECT_EQ(h_c[i], res[i]);
	}

	// transpose
	global_gemv<int, true><<<dim3(2,2,2), dim3(2,2,2)>>>(m, n, static_cast<int>(1), d_a, d_b, d_c);
	hipDeviceSynchronize();
	hipMemcpy(h_c, d_c, m*sizeof(int), hipMemcpyDeviceToHost);
	for(int i=0; i<m; i++){
		EXPECT_EQ(h_c[i], res_transpose[i]);
	}
}

TEST_F(L3Test, chol) {
	double h_d[] = {10, 5, 2, 5, 3, 2, 2, 2, 3};
	double res[] = {pow(10,0.5), 5/pow(10,0.5), 2/pow(10,0.5), 5, 
					1/pow(2,0.5), pow(2,0.5), 2, 2, pow(3,0.5)/pow(5,0.5)};
	double *d_d;

	hipMalloc(&d_d, 9 * sizeof(double));
	hipMemcpy(d_d, h_d, 9 * sizeof(double), hipMemcpyHostToDevice);
	global_cholDecomp_InPlace_c<<<1,9>>>(3, d_d);
	hipDeviceSynchronize();
	hipMemcpy(h_d, d_d, 9*sizeof(double), hipMemcpyDeviceToHost);

	for (int i = 0; i < 9; i++) {
		EXPECT_FLOAT_EQ(h_d[i], res[i]);
	}
	hipFree(d_d);
}

TEST_F(L3Test, gemm){
	int res[] = {140,152,164,176,188,380,424,468,512,556,620,696,772,848,924};
	int res_transpose[] = {420,456,492,528,564,480,524,568,612,656,540,592,644,696,748};
	global_gemm<int, false><<<1, 64>>>(m, n, k, static_cast<int>(1), d_a, d_b, d_c);
	hipDeviceSynchronize();
	hipMemcpy(h_c, d_c, m*k*sizeof(int), hipMemcpyDeviceToHost);
	for(int i=0; i<m*k; i++){
		EXPECT_EQ(h_c[i], res[i]);
	}

	// transpose
	global_gemm<int, true><<<1, 64>>>(m, n, k, static_cast<int>(1), d_a, d_b, d_c);
	hipDeviceSynchronize();
	hipMemcpy(h_c, d_c, m*k*sizeof(int), hipMemcpyDeviceToHost);
	for(int i=0; i<m*k; i++){
		EXPECT_EQ(h_c[i], res_transpose[i]);
	}
}

TEST_F(L3InvTest, invSingle) {

	double res[] = {
		1.0/9, 1.0/33, -1.0/9, 26.0/45, -211.0/495,
		-1.0/9, -1.0/33, -5.0/36, -7.0/90, 349.0/1980,
		-1.0/9, 2.0/33, 13.0/36, -331.0/90, 5407.0/1980,
		1.0/9, -5.0/33, 5.0/36, 7.0/90, -169.0/1980,
		0.0, 1.0/11, -1.0/4, 29.0/10, -483.0/220
	};

	// load identity:	[d_a 	| identity]
	global_loadIdentity<<<1, 64>>>(m, d_a + m*m);
	hipDeviceSynchronize();

	// invert d_a:		[ident w error? | d_a inv]
	global_invertMatrix<<<1, 64>>>(m, d_a, d_temp);
	hipDeviceSynchronize();

	hipMemcpy(h_a, d_a + m*m, m*m * sizeof(*d_a), hipMemcpyDeviceToHost);

	for (int i = 0; i < m*m; i++) {
		EXPECT_LT(abs(h_a[i] - res[i]), 1e-13);
	}
}

TEST_F(L3InvTest, invSingleAndMultiply) {
	// load identity:	[d_a 	| identity]
	global_loadIdentity<<<1, 64>>>(m, d_a + m*m);
	hipDeviceSynchronize();

	// invert d_a:		[ident w error? | d_a inv]
	global_invertMatrix<<<1, 64>>>(m, d_a, d_temp);
	hipDeviceSynchronize();

	// load into d_a again:	[d_a	| d_a inv]
	hipMemcpy(d_a, h_a, m*m * sizeof(*d_a), hipMemcpyHostToDevice);

	// multiply d_a * d_a inv, store result in d_b
	global_gemm<double, false><<<1, 64>>>(m, m, m, 1.0, d_a, d_a + m*m, d_b),
	hipDeviceSynchronize();

	hipMemcpy(h_a, d_b, m*m * sizeof(*d_b), hipMemcpyDeviceToHost);

	// result should be identity
	for (int i = 0; i < m*m; i++) {
		EXPECT_LT(abs(h_a[i] - (i%m == i/m)), 1e-13);
	}
}

TEST_F(L3Test, gemmMultiBlock){
	int res[] = {140,152,164,176,188,380,424,468,512,556,620,696,772,848,924};
	int res_transpose[] = {420,456,492,528,564,480,524,568,612,656,540,592,644,696,748};
	global_gemm<int, false><<<dim3(2,2,2), dim3(2,2,2)>>>(m, n, k, static_cast<int>(1), d_a, d_b, d_c);
	hipDeviceSynchronize();
	hipMemcpy(h_c, d_c, m*k*sizeof(int), hipMemcpyDeviceToHost);
	for(int i=0; i<m*k; i++){
		EXPECT_EQ(h_c[i], res[i]);
	}

	// transpose
	global_gemm<int, true><<<dim3(2,2,2), dim3(2,2,2)>>>(m, n, k, static_cast<int>(1), d_a, d_b, d_c);
	hipDeviceSynchronize();
	hipMemcpy(h_c, d_c, m*k*sizeof(int), hipMemcpyDeviceToHost);
	for(int i=0; i<m*k; i++){
		EXPECT_EQ(h_c[i], res_transpose[i]);
	}
}

int main(){
        ::testing::InitGoogleTest();
        return RUN_ALL_TESTS();
}