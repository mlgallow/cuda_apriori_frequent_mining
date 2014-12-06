/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE


#include "defs.h"
#include "support.h"
#include<iostream>
#include<stdio.h>
using namespace std;
/*__constant__ unsigned short dc_flist_key_16_index[max_unique_items];
__global__ void histogram_kernel_naive(unsigned int* input, unsigned int* bins,
        unsigned int num_elements, unsigned int num_bins) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    while (i < num_elements) {
        int bin_num = input[i];
        if (bin_num < num_bins) {
            atomicAdd(&bins[bin_num], 1);
        }
        i+=stride;
    }
}*/
__global__ void histogram_kernel(unsigned int* input, unsigned int* bins,
        unsigned int num_elements) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int index_x = 0;
    extern __shared__ unsigned int hist_priv[];
    for (int i = 0; i < ceil(MAX_UNIQUE_ITEMS / (1.0 * blockDim.x)); i++){
        index_x = threadIdx.x + i * blockDim.x;
        if (index_x < MAX_UNIQUE_ITEMS)
            hist_priv[index_x] = 0;
    }

    __syncthreads();
    unsigned int stride = blockDim.x * gridDim.x;
    while (i < num_elements) {
        int bin_num = input[i];
        if (bin_num < MAX_UNIQUE_ITEMS) {
            atomicAdd(&hist_priv[bin_num], 1);
        }
        i+=stride;
    }
    __syncthreads();
    for (int i = 0; i < ceil(MAX_UNIQUE_ITEMS / (1.0 * blockDim.x)); i++){
        index_x = threadIdx.x + i * blockDim.x;
        if (index_x < MAX_UNIQUE_ITEMS) {
            atomicAdd(&bins[index_x], hist_priv[index_x]);
        }
    }
}

__global__ void pruneGPU_kernel(unsigned int* input, int num_elements, int min_sup) {
    int tx = threadIdx.x;
    int index = tx + blockDim.x * blockIdx.x;
    if (index < num_elements) {
        if (input[index] < min_sup) {
            input[index] = 0;    
        } 
    }
}
__global__ void  initializeMaskArray(int *mask_d, int maskLength) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < maskLength) {
        mask_d[index] = -1;
    }
}
__global__ void selfJoinKernel(unsigned int *input_d, int *output_d, int num_elements, int power) {
    int tx = threadIdx.x;
    int start = blockIdx.x * MAX_ITEM_PER_SM; 
    __shared__ int sm1[MAX_ITEM_PER_SM];   
    __shared__ int sm2[MAX_ITEM_PER_SM];

    int actual_items_per_sm = num_elements - start;
    if (actual_items_per_sm >= MAX_ITEM_PER_SM) {
        actual_items_per_sm = MAX_ITEM_PER_SM;
    }
    

    int location_x = 0;   
    for (int i = 0; i < ceil(MAX_ITEM_PER_SM/ (1.0 * BLOCK_SIZE));i++) {
        location_x = tx + i * BLOCK_SIZE;
        if (location_x < actual_items_per_sm && (start + location_x) < num_elements) {
            sm1[location_x] = input_d[start + location_x];    
        } else {
            sm1[location_x] = 0; 
        }
    }
    __syncthreads();

    // self join of 1st block
    int loop_tx = 0;
    for (int i = 0; i < ceil(MAX_ITEM_PER_SM/ (1.0 * BLOCK_SIZE));i++) {
        loop_tx = tx + i * BLOCK_SIZE;
        if (loop_tx < actual_items_per_sm) {
            for (int j = loop_tx + 1;j < actual_items_per_sm;j++) {
                if (sm1[loop_tx] / (int)(pow(10.0, (double)power)) == sm1[j] / (int)(pow(10.0, (double)power))) {
                //if (sm1[loop_tx] / 10 == sm1[j] / 10) {
                    output_d[(start + loop_tx) * num_elements + (start + j)] = 0;
               } 
            }   
        }
    }

    __syncthreads();
    if ((blockIdx.x + 1) < ceil(num_elements / (1.0 * MAX_ITEM_PER_SM))) {
        int current_smid = 0;
        for (int smid = blockIdx.x + 1; smid < ceil(num_elements / (1.0 * MAX_ITEM_PER_SM));smid++) {
            int actual_items_per_secondary_sm = num_elements - current_smid * MAX_ITEM_PER_SM - start - MAX_ITEM_PER_SM;
            if (actual_items_per_secondary_sm > MAX_ITEM_PER_SM)
                actual_items_per_secondary_sm = MAX_ITEM_PER_SM;

            for (int i = 0; i < ceil(MAX_ITEM_PER_SM/ (1.0 * BLOCK_SIZE));i++) {
                int location_x = tx + i * BLOCK_SIZE;
                if (location_x < actual_items_per_secondary_sm and (current_smid * MAX_ITEM_PER_SM + start + location_x) < num_elements) {
                    sm2[location_x] = input_d[(current_smid + 1) * MAX_ITEM_PER_SM + start + location_x];
                } else {
                    sm2[location_x] = 0;
                }
            }
            __syncthreads();
                            
            for (int i = 0; i < ceil(MAX_ITEM_PER_SM/ (1.0 * BLOCK_SIZE));i++) {
                if (loop_tx < actual_items_per_sm) {
                    for (int j = 0;j < actual_items_per_secondary_sm;j++) {
                        if (sm1[loop_tx] / (int)(pow(10.0, (double)power)) == sm2[j] / (int)(pow(10.0, (double)power))) {
                        //if (sm1[loop_tx] / 10 == sm2[j] / 10) {
                            output_d[(start + loop_tx) * num_elements + (current_smid + 1) * MAX_ITEM_PER_SM + start + j] = 0;
                       } 
                    }   
                    
                }
            }
        }
        current_smid++;    
    }
}
__global__ void findFrequencyGPU_kernel(unsigned int *d_transactions, 
                                 unsigned int *d_offsets,
                                 int num_transactions,
                                 int num_elements,
                                 unsigned int* d_keyIndex,
                                 int* d_mask,
                                 int num_patterns,
                                 int maskLength) {
    __shared__ unsigned int Ts[MAX_TRANSACTION_PER_SM][MAX_ITEM_PER_TRANSACTION];
    int tx = threadIdx.x;
    
    int index = tx + blockDim.x * blockIdx.x;
    int trans_index = blockIdx.x * MAX_TRANSACTION_PER_SM; 
    //init the SM
    for (int i = 0;i < MAX_TRANSACTION_PER_SM; i++) {
        if (tx < MAX_ITEM_PER_TRANSACTION) {
            Ts[i][tx] = -1; 
        }
    }
    __syncthreads();
    // bring the trnsactions to the SM 
    for (int i = 0;i < MAX_TRANSACTION_PER_SM; i++) {
        int item_ends = num_elements;
        if ((trans_index + i + 1) == num_transactions) {
            item_ends = num_elements;
        } else if ((trans_index + i + 1) < num_transactions) {
            item_ends = d_offsets[trans_index + i + 1];
        } else
            continue;
       if ((tx + d_offsets[trans_index + i]) < item_ends && tx < MAX_ITEM_PER_TRANSACTION) {
           Ts[i][tx] = d_transactions[d_offsets[trans_index + i] + tx];
       }
    }

    __syncthreads();

   for (int maskid = 0; maskid < int(ceil(num_patterns/(1.0 * blockDim.x)));maskid++) {
       int loop_tx = tx + maskid * blockDim.x;
       if (loop_tx >= num_patterns) continue;
       
       for (int last_seen = 0; last_seen < num_patterns; last_seen++) {
           if (loop_tx * num_patterns + last_seen >= maskLength) {
               break;
           }
          if (d_mask[loop_tx * num_patterns + last_seen] < 0) continue;
           
           int item1 = d_keyIndex[loop_tx];
           int item2 = d_keyIndex[last_seen];
           //if (blockIdx.x == 0 && tx == 0)
           //printf("(tx=%d,bx=%d,item1=%d,item2=%d)\n", tx, blockIdx.x, item1, item2);
           for (int tid = 0; tid < MAX_TRANSACTION_PER_SM;tid++) {
               bool flag1 = false;
               bool flag2 = false;
               for (int titem = 0;titem < MAX_ITEM_PER_TRANSACTION;titem++) {
                   //if (blockIdx.x == 0 && tx==0)
                   //printf("(tx=%d,titem=%d)\n", tx, Ts[tid][titem]);
                   if (Ts[tid][titem] == item1) {
                       flag1 = true;
                   } else if (Ts[tid][titem] == item2) {
                       flag2 = true;
                   }
               }
               bool present_flag = flag1 & flag2;
               if (present_flag)
                   atomicAdd(&d_mask[loop_tx * num_patterns + last_seen], 1);
           }
       }    
   }
   
}
__global__ void pruneMultipleGPU_kernel(int *mask_d, int num_patterns, int min_sup) { 
    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
    int index_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (index_x < num_patterns && index_y < num_patterns) {
        int data_index = index_y * num_patterns + index_x;    
        //printf("index%d,indexy%d,index%d\n", index_x, index_y,data_index);
        if (mask_d[data_index] < min_sup) {
            mask_d[data_index] = 0;    
        }
    }
}

__global__ void combinationsAvailable_kernel(int*mask_d, unsigned int*ci_dn, int num_patterns, int maskLength) {
    int index_x = threadIdx.x + blockDim.x * blockIdx.x;
    int index_y = threadIdx.y + blockDim.y * blockIdx.y;
    int mask_index = index_y * num_patterns + index_x;
    if (mask_index < maskLength) {
        if (index_x < num_patterns && index_y < num_patterns && mask_d[mask_index] > 0) {
            atomicAdd(&ci_dn[index_y], 1);
        }
    }
}
__global__ void convert2Sparse(int *input_d,
                               unsigned int *offset_d,
                               unsigned int *output_d, 
                               unsigned int num_patterns,
                               unsigned int k) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (index < (k-1)) {
        int col_start = offset_d[index];
        for (int i =0;i < k;i++) {
            int support_value = input_d[index * k + i];
            if (support_value > 0) {
                output_d[col_start] = index;
                output_d[col_start + num_patterns] = i;
                output_d[col_start + 2 * num_patterns] = support_value;
                col_start++;
            }
        }
    }        
}

__global__ void findHigherPatternFrequencyGPU(unsigned int *d_transactions, unsigned int *d_offsets,
                                              int num_transactions, int num_elements, unsigned int* d_keyIndex,
                                              int *d_mask, int num_patterns, unsigned int *api_d,unsigned  int *iil_d, int power,
                                              int size_api_d, int size_iil_d, int maskLength) {
    
    __shared__ unsigned int Ts[MAX_TRANSACTION_PER_SM][MAX_ITEM_PER_TRANSACTION];
    int tx = threadIdx.x;
    int index = tx + blockDim.x * blockIdx.x;
    int trans_index = blockIdx.x * MAX_TRANSACTION_PER_SM;

    for (int i = 0; i < MAX_TRANSACTION_PER_SM;i++) {
        if (tx < MAX_ITEM_PER_TRANSACTION)  {
            Ts[i][tx] = -1;   
        } 
    } 
    __syncthreads();
    // bring items in SM 
    for (int i = 0;i < MAX_TRANSACTION_PER_SM; i++) {
        int item_ends = num_elements;
        if ((trans_index + i + 1) == num_transactions) {
            item_ends = num_elements;
        } else if ((trans_index + i + 1) < num_transactions) {
            item_ends = d_offsets[trans_index + i + 1];
        } else
            continue;
       if ((tx + d_offsets[trans_index + i]) < item_ends && tx < MAX_ITEM_PER_TRANSACTION) {
           Ts[i][tx] = d_transactions[d_offsets[trans_index + i] + tx];
       }
    }
    __syncthreads();

   for (int maskid = 0; maskid < int(ceil(num_patterns/(1.0 * blockDim.x)));maskid++) {
       int loop_tx = tx + maskid * blockDim.x;
       if (loop_tx >= num_patterns) continue;
       
       for (int last_seen = 0; last_seen < num_patterns; last_seen++) {
          //extra check
          if (loop_tx * num_patterns + last_seen >= maskLength) {
              break;
          }
          if (d_mask[loop_tx * num_patterns + last_seen] < 0) continue;
           
           int hp1 = d_keyIndex[loop_tx];
           int hp2 = d_keyIndex[last_seen];
           int divisor = (int)(pow(10.0, (double)power));
           int vitem1 = hp1 % divisor; 
           int vitem2 = hp2 % divisor;
           //if (blockIdx.x == 0 && tx == 0)
           //printf("(tx=%d,bx=%d,item1=%d,item2=%d)\n", tx, blockIdx.x, item1, item2);
           // now decode virtual item
           int index_item1 = 0;
           int index_item2 = 0;
           int item1 = 0;
           int item2 = 0;
           if ( ((vitem1 - 1) * 3 + 1) < size_iil_d 
               && ((vitem2 - 1) * 3 + 1) < size_iil_d) {
               index_item1 = iil_d[(vitem1 - 1) * 3 + 1];
               index_item2 = iil_d[(vitem2 - 1) * 3 + 1];

               if (index_item1 < size_api_d && index_item2 < size_api_d) {
                   item1 = api_d[index_item1];
                   item2 = api_d[index_item2];
               } else continue;
           } else continue;

           int vcommon_pattern = hp1 / divisor;
           int vpat1 = 0;
           if (((vcommon_pattern - 1) * 3 + 1) < size_iil_d) {
               int index_vpat1 = iil_d[(vcommon_pattern - 1) * 3 + 1];
               if (index_vpat1 < size_api_d) {
                   vpat1 = api_d[index_vpat1];
               } else continue;
           } else continue;


           for (int tid = 0; tid < MAX_TRANSACTION_PER_SM;tid++) {
               // TBD: define a flag array of 
               bool flag1 = false;
               bool flag2 = false;
               bool fpat1 = false;
               for (int titem = 0;titem < MAX_ITEM_PER_TRANSACTION;titem++) {
                   //if (blockIdx.x == 0 && tx==0)
                   //printf("(tx=%d,titem=%d)\n", tx, Ts[tid][titem]);
                   if (Ts[tid][titem] == item1) {
                       flag1 = true;
                   } else if (Ts[tid][titem] == item2) {
                       flag2 = true;
                   } else {
                       fpat1 = true;    
                   }
               }
               bool present_flag = flag1 & flag2& fpat1;
               if (present_flag)
                   atomicAdd(&d_mask[loop_tx * num_patterns + last_seen], 1);
           }
       }    
   }
}
