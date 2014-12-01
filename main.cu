//#include <stdio.h>
#include <iostream>
#include<algorithm>
#include <stdlib.h>
#include <stdint.h>
#include <sstream>
#include <fstream>
#include "defs.h"
#include "support.h"
#include "kernel.cu"
#include<vector>
#include<utility>
using namespace std;


bool pair_compare(const pair<short unsigned int, unsigned int>& p1,const pair<short unsigned int, unsigned int>& p2);
int main(int argc, char* argv[])
{
    char *line = NULL;
    size_t len = 0;
    unsigned int lines = 0;
    unsigned int count = 0;
    char *ln, *nptr;

    unsigned int *transactions = NULL;
    unsigned int *trans_offset = NULL;
    unsigned int *ci_h = NULL;//bins array for histogram op
    //unsigned int *flist = NULL;
    //unsigned short *flist_key_16_index = NULL;

    unsigned int element_id = 0;

    transactions = (unsigned int *) malloc(MAX_NUM_ELEMENTS * sizeof(unsigned int));
    trans_offset = (unsigned int *) malloc((MAX_NUM_ELEMENTS + 1) * sizeof(unsigned int));
    ci_h = (unsigned int *) malloc(MAX_UNIQUE_ITEMS * sizeof(unsigned int));
    //flist = (unsigned int *) malloc(max_unique_items * sizeof(unsigned int));
    //flist_key_16_index = (unsigned short*) malloc(max_unique_items * sizeof(unsigned short));

//    memset(flist_key_16_index, 0xFFFF, max_unique_items * sizeof(unsigned short));

    lines = 0;
    element_id = 0;
    ifstream fp1("topic-3.txt");
    string curline, space(" ");
    if (fp1.is_open()) {
        cout<<"file opened"<<endl;
        trans_offset[0] = 0;
        while(getline(fp1, curline) && lines < MAX_TRANSACTIONS) {
            count = 0;
            istringstream s(curline); 
            string st;
            while(getline(s, st, ' ') && count < MAX_ITEM_PER_TRANSACTION) {
                int item = atol(st.c_str());
                if (item < MAX_UNIQUE_ITEMS) {
                    // add an item only if it is in the range [0,max_unique_items)
                    transactions[element_id++] = atol(st.c_str());
                    count++;
                }
            }
            if (count > 0) {
                // consider this transaction if there is atleast one item
                trans_offset[lines + 1] = trans_offset[lines] + count;
                lines++;
            }
        }
    } else {
        cout<<"error in reading from file"<<endl;
        return 0;
    }
    fp1.close();
    unsigned int num_elements = element_id;
    unsigned int num_transactions = lines;
    cout<<"Number of Transactions = "<<num_transactions<<endl;
    cout<<"num_elements in transactions array = "<<num_elements<<endl;
    #ifdef TEST_PARAMS
    for (int i = 0; i < num_elements; i++){
        cout<<transactions[i]<<" ";
    }
    cout<<endl;
    for (int i = 0; i <= num_transactions; i++) {
       cout<<"(i,offset)"<<i<<","<<trans_offset[i]; 
    }
    #endif

    //calculate max power
    int power = 1;
    while ((MAX_UNIQUE_ITEMS / (int)(pow(10.0, (double)power))) != 0) {
        power += 1;
    }
    cout<<"max power = "<<power<<endl;

    //check for max item exceed
    if (num_elements > MAX_NUM_ELEMENTS) {
        cout<<"Error: Elements exceeding NUM_ELEMENTS. Exiting...";
        return -1;
    }
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////// Device Variables Initializations ///////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    unsigned int *d_input;//
    unsigned int *d_offsets;
    unsigned int *ci_d;//bins array - each index corrosponds to an item
    cudaDeviceProp deviceProp;
    Timer timer;
    cudaError_t cuda_ret;
    cudaGetDeviceProperties(&deviceProp, 0);
    cout<<"Allocating device variables...";
    startTime(&timer);
    cuda_ret = cudaMalloc((void**)&d_input, num_elements * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&d_offsets, (num_transactions + 1) * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
    cuda_ret = cudaMalloc((void**)&ci_d, MAX_UNIQUE_ITEMS * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMemset(ci_d, 0, MAX_UNIQUE_ITEMS * sizeof(unsigned int));
    cuda_ret = cudaMemcpy(d_input, transactions, num_elements * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy input to the device");
    cuda_ret = cudaMemcpy(d_offsets, trans_offset, (num_transactions+1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy input to the device");
    
    dim3 grid_dim, block_dim;
    block_dim.x = BLOCK_SIZE; 
    block_dim.y = 1; block_dim.z = 1;
    grid_dim.x = ceil(num_elements / (1.0 * BLOCK_SIZE)); 
    grid_dim.y = 1; grid_dim.z = 1;
    cout<<"launching histogram kernel(grid, block):"<<grid_dim.x<<","<<block_dim.x<<endl;
    startTime(&timer);
    histogram_kernel<<<grid_dim, block_dim, MAX_UNIQUE_ITEMS * sizeof(unsigned int)>>>(d_input, ci_d, num_elements);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch Histogram kernel");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
    // prune the histogram op 
    block_dim.x = BLOCK_SIZE; 
    block_dim.y = 1; block_dim.z = 1;
    grid_dim.x = ceil(MAX_UNIQUE_ITEMS / (1.0 * BLOCK_SIZE)); 
    grid_dim.y = 1; grid_dim.z = 1;
    cout<<"launching pruning kernel(grid, block):"<<grid_dim.x<<","<<block_dim.x<<endl;
    startTime(&timer);
    pruneGPU_kernel<<<grid_dim, block_dim>>>(ci_d, MAX_UNIQUE_ITEMS, MIN_SUPPORT);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy histogram op back to host");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
    
    cout<<"copying hist op back to host"<<endl;
    startTime(&timer);
    cudaMemcpy(ci_h, ci_d, MAX_UNIQUE_ITEMS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy histogram op back to host");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
#ifdef TEST_PARAMS
    cout<<"histogram output after pruning:"<<endl;
    for (int i = 0; i < MAX_UNIQUE_ITEMS; i++) {
        cout<<"ci_h["<<i<<"]="<<ci_h[i]<<endl;   
    }     
#endif
    unsigned int *li_h; // this list contains the actual items which passed min support test
    unsigned int  k = 0; //count of actual items which passed min support test
    for (int i = 0;i<MAX_UNIQUE_ITEMS;i++) {
        if (ci_h[i] != 0) {
            k++;    
        }    
    }
    cout<<"num items with good support count="<<k<<endl;
    li_h = (unsigned int *) malloc(k * sizeof(unsigned int));
    /*if (li_h ==  NULL) {
        cout<<"faild to alloc li_h...exiting!"<<endl;
        goto exit;    
    }*/
    int li_count = 0;
    for (int i = 0;i<MAX_UNIQUE_ITEMS;i++) {
        if (ci_h[i] != 0) {
            li_h[li_count++] = ci_h[i]; 
        } 
    }

//#ifdef TEST_PARAMS
    cout<<"li_h after pruning:"<<endl;
    for (int i = 0; i < k; i++) {
        cout<<"li_h["<<i<<"]="<<li_h[i]<<endl;   
    }
//#endif
    unsigned int *li_d;
    cuda_ret = cudaMalloc((void**)&li_d, k * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cudaMemcpy(li_d, li_h, k * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy li_h to device");
    
    int maskLength = pow(float(k), 2);
    cout <<"maskLength ="<<maskLength<<endl;
    int *mask_h = (int*)malloc(maskLength * sizeof(int));
    int* mask_d;//mask matrix
    cout<<"alloc mask matrix"<<endl;
    startTime(&timer);
    cuda_ret = cudaMalloc((void**)&mask_d, maskLength * sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
    
    block_dim.x = BLOCK_SIZE;
    block_dim.y = 1;
    block_dim.y = 1;
    grid_dim.x = (int) ceil((maskLength) / (1.0 * block_dim.x));
    grid_dim.y = 1;
    grid_dim.z = 1;
    cout<<"init mask"<<endl;
    startTime(&timer);
    initializeMaskArray<<<grid_dim, block_dim>>>(mask_d, maskLength);
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
    cudaDeviceSynchronize();

    block_dim.x = BLOCK_SIZE;
    block_dim.y = 1;
    block_dim.y = 1;
    grid_dim.x = (int) ceil(k / (1.0 * MAX_ITEM_PER_SM));
    grid_dim.y = 1;
    grid_dim.z = 1;
    cout<<"self join launched with <grid,block>"<<grid_dim.x<<","<<block_dim.x<<endl;
    startTime(&timer);
    selfJoinKernel<<<grid_dim, block_dim>>>(li_d, mask_d, k, power);
    cudaDeviceSynchronize();
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
  
    cout<<"copy mask back to host"<<endl;
    startTime(&timer);
    cudaMemcpy(mask_h, mask_d, maskLength * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy histogram op back to host");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
//#ifdef TEST_PARAMS
    cout<<"mask_h"<<endl;
    for (int i = 0;i < maskLength; i++) {
        cout<<"mask["<<i<<"]="<<mask_h[i]<<endl;   
        
    }
//#endif
#if 0

    unsigned int *d_flist, *d_flist_key_16;
    unsigned short *d_flist_key_16_index;
    int SM_PER_BLOCK = deviceProp.sharedMemPerBlock;
    int CONST_MEM_GPU = deviceProp.totalConstMem;
    // Allocate device variables ----------------------------------------------


    cuda_ret = cudaMalloc((void**)&d_flist, max_unique_items * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&d_flist_key_16_index, max_unique_items * sizeof(unsigned short));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&d_flist_key_16, max_unique_items * sizeof(unsigned short));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");


    cudaDeviceSynchronize();
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
    cuda_ret = cudaMemcpy(d_transactions, transactions, num_items_in_transactions * sizeof(unsigned int),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    // to test
	cuda_ret = cudaMemcpy(d_trans_offsets, trans_offset, (num_transactions + 1) * sizeof(unsigned int),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    
    cuda_ret = cudaMemset(d_flist, 0, max_unique_items * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");
    
    startTime(&timer);
    cout<<"histogram kernel\n";
    make_flist(d_trans_offsets, d_transactions, d_flist, num_transactions, num_items_in_transactions, SM_PER_BLOCK);
    // now prune flist
    dim3 block_dim;
    dim3 grid_dim;
    
    block_dim.x = BLOCK_SIZE;
    block_dim.y = 1;
    block_dim.y = 1;

    grid_dim.x = (int) ceil(max_unique_items / (1.0 * block_dim.x));
    grid_dim.y = 1;
    grid_dim.z = 1;

    pruneList<<<grid_dim, block_dim>>>(d_flist, max_unique_items, support);
    cout<<"constant mem available:"<<CONST_MEM_GPU<<endl;
    startTime(&timer);
    cout<<"copying flist form dev to host\n";
    cuda_ret = cudaMemcpy(flist, d_flist, max_unique_items * sizeof(unsigned int),
        cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");
    cudaDeviceSynchronize();
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;

    // now keep track of only actual number of items which have passed support count
    int *li_h = (int *) malloc(max_unique_items * sizeof(int));
    int actualNumItems = 0;
    for (int i =0; i < max_unique_items;i++) {
        if (flist[i] != 0) {
            li_h[actualNumItems++] = i;
        } 
    }
    
    int*li_d;
    cuda_ret = cudaMalloc((void**)&li_d, actualNumItems * sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    // Free memory ------------------------------------------------------------

    #ifdef TEST_MODE
    int*h_mask;
    h_mask = (int*)malloc(maskLength * sizeof(int));
    cudaMemcpy(h_mask, d_mask, maskLength * sizeof(int),
       cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    for (int i =0;i < maskLength;i++) {
        cout<<h_mask[i]<<",";    
    }
        /*for (int i =0;i < actualNumItems;i++) {
            cout<<li_h[i]<<",";
        }*/
    free(h_mask);
    #endif
    cudaFree(d_flist);
    cudaFree(d_flist_key_16);
    cudaFree(d_flist_key_16_index);
    cudaFree(d_mask);
    
    free(li_h);
    free(flist);
    //free(flist_key_16);
    free(flist_key_16_index);
#endif
exit:
    if (trans_offset) {
        free(trans_offset);
    }
    if (transactions) {
        free(transactions);
    }
    if (ci_h) {
        free(ci_h);
    }
    if (li_h) {
        free(li_h);    
    }
    if (mask_h) {
        free(mask_h);    
    }
    cudaFree(d_offsets);
    cudaFree(d_input);
    cudaFree(ci_d);
    cudaFree(li_d);
    cudaFree(mask_d);
    cout<<"program end";

}

bool pair_compare(const pair<short unsigned int, unsigned int>& p1,const pair<short unsigned int, unsigned int>& p2) {
    return p1.second < p2.second;    
}
