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
#include "kernel_prescan.cu"
#include<vector>
#include<map>
#include<utility>
#include<algorithm>
using namespace std;

class map_pair{
    public:
    unsigned int row_item;
    unsigned int col_item;
    map_pair(unsigned int row, unsigned int col) {
        this->row_item = row;
        this->col_item = col;    
    }
    bool operator < (const map_pair& another) const {
        if (this->row_item == another.row_item) return this->col_item < another.col_item;
        else return this->row_item < another.row_item;
    }
    ~map_pair() {}
};

class tuple {
    // ensure it doesnt exceed MAX_PATTERN
    public:
    vector<int> values;
    tuple(int val1) {
        values.push_back(val1);
    }
    tuple() {}
    tuple(int val1, int val2) {
        values.push_back(val1);
        values.push_back(val2);
        std::sort(values.begin(), values.end());    
    }
    tuple(int val1, int val2, int val3) {
        values.push_back(val1);
        values.push_back(val2);
        values.push_back(val3);
        std::sort(values.begin(), values.end());    
    }
   
    int get(int index) {
        if (index >= values.size()) index = values.size();
        else if (index < 0) index = 0;
        
        return values[index];  
    } 
    
    int size() {
       return values.size(); 
    }
    
    void print() {
        vector<int>::iterator it = values.begin();
        cout<<"(";
        while(it != values.end()) {
            cout<<*it<<",";
            it++; 
        }
        cout<<")";
    }
    bool operator==(const tuple &other) const {
        if (values.size() != (other.values).size()) return false;
        vector<int>::const_iterator it, it_other;
        for (it = values.begin(), it_other = other.values.begin(); it != values.end();it++, it_other++) {
            if (*it != *it_other) return false;
        }
        return true;
    }
    
    bool insertValues(int item) {
        if (values.size() >= MAX_PATTERN_SEARCH) return false;
        values.push_back(item);
        return true;
    }
    
    tuple getFirstNitems(int n) {
        if (n > values.size()) n = values.size();
        vector<int>::iterator it = values.begin();
        tuple op;
        for (int i = 0;i < n;i++) {
            op.insertValues(*it++);
        }
        return op;
    }

    tuple getLastItem() {
        return values[values.size() - 1];
    }
};

bool isTuplePresent(const vector<std::pair<tuple, int> >&list, const tuple &t) {
    if (list.size() == 0) return false;
    vector<std::pair<tuple, int> >::const_iterator it = list.begin();
    while(it != list.end()) {
        tuple cur_tuple = it->first;
        if (cur_tuple == t) return true;
        it++;
    }
    return false;
}

int  getTupleValue(const vector<std::pair<tuple, int> >&list, const tuple &t) {
    if (list.size() == 0) return INVALID;
    vector<std::pair<tuple, int> >::const_iterator it = list.begin();
    while(it != list.end()) {
        tuple cur_tuple = it->first;
        if (cur_tuple == t) {
            return it->second;  
        }
        it++;
    }
    return false;
}

int compare(const void *a, const void *b) {
    int a1 = *((int*)a);
    int b1 = *((int*)b);
    if (a1 == b1) return 0;
    else if (a1 < b1) return -1;
    else return 1;
}
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

    unsigned int element_id = 0;

    transactions = (unsigned int *) malloc(MAX_NUM_ELEMENTS * sizeof(unsigned int));
    trans_offset = (unsigned int *) malloc((MAX_NUM_ELEMENTS + 1) * sizeof(unsigned int));
    ci_h = (unsigned int *) malloc(MAX_UNIQUE_ITEMS * sizeof(unsigned int));

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
    /*for (int i = 0; i < num_elements; i++){
        cout<<transactions[i]<<" ";
    }
    cout<<endl;
    for (int i = 0; i <= num_transactions; i++) {
       cout<<"(i,offset)"<<i<<","<<trans_offset[i]; 
    }*/
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
            li_h[li_count++] = i; 
        } 
    }

#ifdef TEST_PARAMS
    cout<<"li_h after pruning:"<<endl;
    for (int i = 0; i < k; i++) {
        cout<<"li_h["<<i<<"]="<<li_h[i]<<endl;   
    }
#endif
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
  
    // TBD:to test. remove in final code
    cout<<"copy mask back to host"<<endl;
    startTime(&timer);
    cudaMemcpy(mask_h, mask_d, maskLength * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy histogram op back to host");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
#ifdef TEST_PARAMS
    /*cout<<"################mask_h after join#############"<<endl;
    for (int i = 0;i < maskLength; i++) {
        cout<<"mask["<<i<<"]="<<mask_h[i]<<endl;   
        
    }*/
#endif
    block_dim.x = BLOCK_SIZE;
    block_dim.y = 1;
    block_dim.y = 1;
    grid_dim.x = (int) ceil((num_transactions) / (1.0 * MAX_TRANSACTION_PER_SM));
    grid_dim.y = 1;
    grid_dim.z = 1;
    cout<<"findFrequencyGPU <grid,block>"<<grid_dim.x<<","<<block_dim.x<<endl;
    startTime(&timer);
    findFrequencyGPU_kernel<<<grid_dim, block_dim>>>(d_input, d_offsets, num_transactions, num_elements, li_d, mask_d, k, maskLength);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch findFrequencyGPU_kernel");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
    //prune the 2d mask matrix
    block_dim.x = BLOCK_SIZE;
    block_dim.y = BLOCK_SIZE;
    block_dim.y = 1;
    grid_dim.x = (int) ceil(k / (1.0 * block_dim.x));
    grid_dim.y = (int) ceil(k / (1.0 * block_dim.y));
    grid_dim.z = 1;
    //cout<<"gridy"<<grid_dim.y<<endl;
    cout<<"pruneMultipleGPU <grid,block>"<<grid_dim.x<<","<<block_dim.x<<endl;
    startTime(&timer);
    pruneMultipleGPU_kernel<<<grid_dim, block_dim>>>(mask_d, k, MIN_SUPPORT);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch pruneMultipleGPU");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;

    cout<<"copy mask back to host"<<endl;
    startTime(&timer);
    cudaMemcpy(mask_h, mask_d, maskLength * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy histogram op back to host");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
#ifdef TEST_PARAMS
    /*cout<<"################mask_h after findFrequencyGPU_kernel and Prune#############"<<endl;
    for (int i = 0;i < maskLength; i++) {
        cout<<"mask["<<i<<"]="<<mask_h[i]<<endl;   
        
    }*/
#endif
    //now we need to convert the mask array to a sparse matrix in parallel
    // this means we need to find number of non zero entries in each row of mask matrix
    // and the allocate memory equal to total number of non zero items.
    // each thread can then directly work on an offset into the array, 
    // obtained by perorming a exclusive scan.
    unsigned int *ci_dn;
    unsigned int *ci_hn;
    ci_hn = (unsigned int*) malloc(k * sizeof (unsigned int));
    cuda_ret = cudaMalloc((void**)&ci_dn, k * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cudaMemset(ci_dn, 0, k * sizeof(unsigned int));
    
    block_dim.x = BLOCK_SIZE;
    block_dim.y = BLOCK_SIZE;
    block_dim.y = 1;
    grid_dim.x = (int) ceil(k / (1.0 * block_dim.x));
    grid_dim.y = (int) ceil(k / (1.0 * block_dim.y));
    grid_dim.z = 1;

    cout<<"combinationsAvailable_kernel <grid,block>"<<grid_dim.x<<","<<block_dim.x<<endl;
    startTime(&timer);
    combinationsAvailable_kernel<<<grid_dim, block_dim>>>(mask_d, ci_dn, k, maskLength);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch combinationsAvailable_kernel");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
    cudaMemcpy(ci_hn, ci_dn, k * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy histogram op back to host");
//#ifdef TEST_PARAMS
    for (int i = 0; i < k; i++) {
        cout<<"ci_dn["<<i<<"]="<<ci_hn[i]<<endl;    
    }
//#endif
    // prescan the ci_hn array
    unsigned int *ci_hnx;
    unsigned int *ci_dnx;
    ci_hnx = (unsigned int*) malloc(k * sizeof (unsigned int));
    cuda_ret = cudaMalloc((void**)&ci_dnx, k * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cudaMemset(ci_dnx, 0, k * sizeof(unsigned int));
    preScan(ci_dnx, ci_dn, k);
    cudaMemcpy(ci_hnx, ci_dnx, k * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy histogram op back to host");
//#ifdef TEST_PARAMS
    cout<<"scan op"<<endl;
    for (int i = 0; i < k; i++) {
        cout<<"ci_dnx["<<i<<"]="<<ci_hnx[i]<<endl;    
    }
//#endif

    unsigned int *sparseM_h;
    unsigned int *sparseM_d;
    unsigned int sparse_matrix_size = ci_hnx[k-1];
    cout<<"allocating sparse matrix for size"<<sparse_matrix_size<<endl; 
    sparseM_h = (unsigned int*) malloc(3 * sparse_matrix_size * sizeof (unsigned int));
    cuda_ret = cudaMalloc((void**)&sparseM_d, 3 * sparse_matrix_size * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cudaMemset(sparseM_d, 0, 3 * sparse_matrix_size * sizeof(unsigned int));
    block_dim.x = BLOCK_SIZE;
    block_dim.y = 1;
    block_dim.y = 1;
    grid_dim.x = (int) ceil(k / (1.0 * block_dim.x));
    grid_dim.y = 1;
    grid_dim.z = 1;
    cout<<" convert2Sparse kernel <grid,block>"<<grid_dim.x<<","<<block_dim.x<<endl;
    startTime(&timer);
    convert2Sparse<<<grid_dim, block_dim>>>(mask_d, ci_dnx, sparseM_d, sparse_matrix_size, k);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch convert2Sparse");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
    
    cudaMemcpy(sparseM_h, sparseM_d, 3 * sparse_matrix_size * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy histogram op back to host");
#ifdef TEST_PARAMS
    /*cout<<"sparse op(row,col,val)"<<endl;
    for (int i = 0; i < sparse_matrix_size; i++) {
        cout<<"sparse("<<sparseM_h[i]<<","<<sparseM_h[i + sparse_matrix_size]<<","<<sparseM_h[i + 2*sparse_matrix_size]<<")"<<endl;    
    }*/
#endif
    //now create a STL map and add the sparse matrix values to the map
    //map<map_pair, unsigned int> patterns;
    vector<std::pair<tuple, int> > patterns;
    cout<<"build vector from sparse array of length = "<<sparse_matrix_size<<endl;
    for (int i = 0; i< sparse_matrix_size;i++) {
        tuple t(sparseM_h[i], sparseM_h[i + sparse_matrix_size]);
        int item = sparseM_h[i + 2 * sparse_matrix_size];
        patterns.push_back(std::pair<tuple, unsigned int>(t, item));    
    }
    cout<<"map size"<<patterns.size()<<endl;
#ifdef TEST_PARAMS
    vector<std::pair<tuple, int> >::iterator it;
    for (it = patterns.begin(); it != patterns.end();it++) {
        it->first.print();
        cout<<"="<<it->second<<endl;    
    }
#endif
    vector<std::pair<tuple, int> > new_modulo_map;
    //cout<<isTuplePresent(dict, t);
    vector<std::pair<tuple, int> >::iterator it_modulo_map;
    int index_id = 1;
    int actual_patterns_items_size = 0;
    for (it_modulo_map = patterns.begin();it_modulo_map != patterns.end();it_modulo_map++) {
        tuple t = it_modulo_map->first;
        cout<<"tuple:";
        t.print();
        cout<<"----";
        //since now there is only 2 items in the tuple.
        tuple op = t.getFirstNitems(1);
        tuple op1 = t.getLastItem();
        cout<<"split tuple=";
        op.print();
        cout<< "---";
        op1.print();
        cout<<endl;
        if (!isTuplePresent(new_modulo_map, op)) {
            cout<<"adding tuple to api_h=";
            op.print();
            cout<<"id assigned="<<index_id<<endl;
            actual_patterns_items_size += op.size();
            new_modulo_map.push_back(std::pair<tuple, int>(op, index_id));
            index_id++;
        }
        if (!isTuplePresent(new_modulo_map, op1)) {
            cout<<"adding tuple to api_h=";
            op1.print();
            cout<<"id assigned="<<index_id<<endl;
            actual_patterns_items_size += op1.size();
            new_modulo_map.push_back(std::pair<tuple, int>(op1, index_id)); 
            index_id++;
        }
    }

#ifdef TEST_PARAMS
    for (it_modulo_map = new_modulo_map.begin(); it_modulo_map != new_modulo_map.end();it_modulo_map++) {
        cout<<"id[";
        it_modulo_map->first.print();
        cout<<"]="<<it_modulo_map->second<<endl; 
    }
#endif
    cout<<"actual_patterns_items_size:"<<actual_patterns_items_size<<endl;
    int index_items_lookup_size = 3 * new_modulo_map.size();// (index_id, start, length)
    cout<<"index_items_lookup_size :"<<index_items_lookup_size<<endl;
    unsigned int *actual_patterns_items = (unsigned int *) malloc(actual_patterns_items_size * sizeof (unsigned int));
    unsigned int *index_items_lookup = (unsigned int *) malloc(index_items_lookup_size * sizeof (unsigned int));
    int start_offset = 0;
    int counter = 0;
    for (it_modulo_map = new_modulo_map.begin(); it_modulo_map != new_modulo_map.end();it_modulo_map++) {
        tuple t = it_modulo_map->first;
        //cout<<"makeiid tuple:";
        //t.print();
        //cout<<"--- index_id="<<it_modulo_map->second<<" start="<<start_offset<<"length="<<t.size()<<endl;
        index_items_lookup[counter] = it_modulo_map->second;
        index_items_lookup[counter+1] = start_offset; 
        index_items_lookup[counter+2] = t.size();
        //cout<<"--- index_id_tuple=("<<index_items_lookup[counter]<<","<<index_items_lookup[counter+1]<<","<<index_items_lookup[counter+2]<<endl;
        counter +=3;
        for (int i =0; i < t.size();i++) {
            actual_patterns_items[start_offset] = t.get(i);
            cout<<"api_h["<<start_offset<<"]="<<actual_patterns_items[start_offset]<<endl;
            start_offset++;
        }
    }
#ifdef TEST_PARAMS
    for (int i = 0;i < index_items_lookup_size;i+=3) {
        cout<<"iil_h["<<i<<"]="<<"("<<index_items_lookup[i]<<","<<index_items_lookup[i+1]<<","<<index_items_lookup[i+2]<<")"<<endl;
    }
#endif
#if 0
    // now create the new encoded array
    unsigned int *new_new_patterns;
    unsigned int *new_new_patterns_d;
    int new_new_patterns_size = patterns.size(); 
    new_new_patterns = (unsigned int *) malloc(new_new_patterns_size * sizeof (unsigned int));
    counter = 0;
    int mul_factor = (int)(pow(10.0, (double)power));
    for (it_modulo_map = patterns.begin();it_modulo_map != patterns.end();it_modulo_map++) {
        tuple t = it_modulo_map->first;
        //since now there is only 2 items in the tuple.
        tuple op = t.getFirstNitems(1);
        tuple op1 = t.getLastItem();
        int code1 = getTupleValue(new_modulo_map, op);
        int code2 = getTupleValue(new_modulo_map, op1);
        if (code1 == INVALID || code2 == INVALID) continue;
        
        int newcode = code1 * mul_factor + code2;
        new_new_patterns[counter++] = newcode;
    }
   
    //may apply radix sort to sort it
    qsort(new_new_patterns, sizeof(new_new_patterns)/sizeof(new_new_patterns[0]), sizeof(new_new_patterns[0]), compare);
    //send the array to device
    cuda_ret = cudaMalloc((void**)&new_new_patterns_d, new_new_patterns_size * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMemcpy(new_new_patterns_d, new_new_patterns_d, new_new_patterns_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    //#########################################################//
    //############# start of second phase######################// 
    // calculate parameters again for second phase
    k = counter;
    maskLength = pow(float(k), 2);
    cout <<"maskLength ="<<maskLength<<endl;
    cuda_ret = cudaMemset(mask_d, -1, maskLength * sizeof(int));
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
    
    //TBD:remove. only for test
    cout<<"copy mask back to host"<<endl;
    startTime(&timer);
    cudaMemcpy(mask_h, mask_d, maskLength * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy histogram op back to host");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
#ifdef TEST_PARAMS
    cout<<"################mask_h after join#############"<<endl;
    for (int i = 0;i < maskLength; i++) {
        cout<<"mask["<<i<<"]="<<mask_h[i]<<endl;   
        
    }
#endif
    unsigned int *actual_patterns_items_d;
    unsigned int *index_items_lookup_d;
    cuda_ret = cudaMalloc((void**)&actual_patterns_items_d, actual_patterns_items_size * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&index_items_lookup_d, index_items_lookup_size * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cuda_ret = cudaMemcpy(actual_patterns_items_d, actual_patterns_items, actual_patterns_items_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy input to the device");
    cuda_ret = cudaMemcpy(index_items_lookup_d, index_items_lookup, index_items_lookup_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy input to the device");

    block_dim.x = BLOCK_SIZE;
    block_dim.y = 1;
    block_dim.y = 1;
    grid_dim.x = (int) ceil((num_transactions) / (1.0 * MAX_TRANSACTION_PER_SM));
    grid_dim.y = 1;
    grid_dim.z = 1;
    cout<<"findHigherPatternFrequencyGPU launched with <grid,block>"<<grid_dim.x<<","<<block_dim.x<<endl;
    startTime(&timer);
    findHigherPatternFrequencyGPU<<<grid_dim, block_dim>>>(d_input, d_offsets,
                                  num_transactions, 
                                  num_elements, new_new_patterns_d,
                                  mask_d, k, actual_patterns_items_d,
                                  index_items_lookup_d, power,
                                  actual_patterns_items_size,
                                  index_items_lookup_size, maskLength);
    cudaDeviceSynchronize();
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;

    // prune the matrix
    block_dim.x = BLOCK_SIZE;
    block_dim.y = BLOCK_SIZE;
    block_dim.y = 1;
    grid_dim.x = (int) ceil(k / (1.0 * block_dim.x));
    grid_dim.y = (int) ceil(k / (1.0 * block_dim.y));
    grid_dim.z = 1;
    cout<<"pruneMultipleGPU <grid,block>"<<grid_dim.x<<","<<block_dim.x<<endl;
    startTime(&timer);
    pruneMultipleGPU_kernel<<<grid_dim, block_dim>>>(mask_d, k, MIN_SUPPORT);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch pruneMultipleGPU");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;

    //find combinations available
    block_dim.x = BLOCK_SIZE;
    block_dim.y = BLOCK_SIZE;
    block_dim.y = 1;
    grid_dim.x = (int) ceil(k / (1.0 * block_dim.x));
    grid_dim.y = (int) ceil(k / (1.0 * block_dim.y));
    grid_dim.z = 1;

    cudaMemset(ci_dn, 0, k * sizeof(unsigned int));
    cout<<"combinationsAvailable_kernel <grid,block>"<<grid_dim.x<<","<<block_dim.x<<endl;
    startTime(&timer);
    combinationsAvailable_kernel<<<grid_dim, block_dim>>>(mask_d, ci_dn, k, maskLength);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch combinationsAvailable_kernel");
    
    // prescan it to get offsets
    cudaMemset(ci_dnx, 0, k * sizeof(unsigned int));
    preScan(ci_dnx, ci_dn, k);
    cudaMemcpy(ci_hnx, ci_dnx, k * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy histogram op back to host");
    
    unsigned int *sparseM_h1;
    unsigned int *sparseM_d1;
    unsigned int sparse_matrix_size1 = ci_hnx[k-1];
    cout<<"allocating sparse matrix for size"<<sparse_matrix_size1<<endl; 
    sparseM_h1 = (unsigned int*) malloc(3 * sparse_matrix_size1 * sizeof (unsigned int));
    cuda_ret = cudaMalloc((void**)&sparseM_d1, 3 * sparse_matrix_size1 * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cudaMemset(sparseM_d1, 0, 3 * sparse_matrix_size1 * sizeof(unsigned int));
    block_dim.x = BLOCK_SIZE;
    block_dim.y = 1;
    block_dim.y = 1;
    grid_dim.x = (int) ceil(k / (1.0 * block_dim.x));
    grid_dim.y = 1;
    grid_dim.z = 1;
    cout<<" convert2Sparse kernel <grid,block>"<<grid_dim.x<<","<<block_dim.x<<endl;
    startTime(&timer);
    convert2Sparse<<<grid_dim, block_dim>>>(mask_d, ci_dnx, sparseM_d1, sparse_matrix_size1, k);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch convert2Sparse");
    stopTime(&timer); cout<<elapsedTime(timer)<<endl;
    
    cudaMemcpy(sparseM_h1, sparseM_d1, 3 * sparse_matrix_size1 * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy histogram op back to host");
    
    
    // make the sparse array
    vector<std::pair<tuple, int> > patterns1;
    cout<<"build vector from sparse array of length = "<<sparse_matrix_size<<endl;
    for (int i = 0; i< sparse_matrix_size1;i++) {
        tuple t(sparseM_h1[i], sparseM_h1[i + sparse_matrix_size1]);
        int item = sparseM_h1[i + 2 * sparse_matrix_size1];
        patterns.push_back(std::pair<tuple, unsigned int>(t, item));    
    }
    cout<<"map size"<<patterns1.size()<<endl;
#ifdef TEST_PARAMS
    for (it = patterns1.begin(); it != patterns1.end();it++) {
        it->first.print();
        cout<<"="<<it->second<<endl;    
    }
#endif
exit:
    ///////////////////////////////////
    free(new_new_patterns);
    free(sparseM_h1);
    cudaFree(sparseM_d1);
    cudaFree(actual_patterns_items_d);
    cudaFree(index_items_lookup_d);
#endif
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
    if (ci_hn) {
        free(ci_hn);    
    }
    if (ci_hnx) {
        free(ci_hnx);    
    }
    if (sparseM_h) {
        free(sparseM_h);
    }
    if(actual_patterns_items) {
        free(actual_patterns_items);
    }
    if (index_items_lookup) {
        free(index_items_lookup);
    }
    cudaFree(d_offsets);
    cudaFree(d_input);
    cudaFree(ci_d);
    cudaFree(li_d);
    cudaFree(mask_d);
    cudaFree(ci_dn);
    cudaFree(ci_dnx);
    cudaFree(sparseM_d);
    cout<<"program end";

}

bool pair_compare(const pair<short unsigned int, unsigned int>& p1,const pair<short unsigned int, unsigned int>& p2) {
    return p1.second < p2.second;    
}
