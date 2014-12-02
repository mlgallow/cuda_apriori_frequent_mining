#define TEST_PARAMS
#ifdef TEST_PARAMS

#define MAX_UNIQUE_ITEMS 64
#define MAX_PATTERN_SEARCH 5 //max n-frequent items to search
#define BLOCK_SIZE 8 // must be greater than MAX_ITEM_PER_TRANSACTION
#define MAX_ITEM_PER_SM 32// can be a maximum of 6k items, each 4 bytes
// this is number of transactions per SM
// where eahc transaction can have MAX_ITEM_PER_TRANSACTION items
#define MAX_TRANSACTION_PER_SM 32
#define MAX_ITEM_PER_TRANSACTION 16
#define MIN_SUPPORT 2
#define MAX_TRANSACTIONS 32//1000
#define MAX_NUM_ELEMENTS (MAX_TRANSACTIONS * MAX_ITEM_PER_TRANSACTION)

#else

#define MAX_UNIQUE_ITEMS 512
#define MAX_PATTERN_SEARCH 5 
#define BLOCK_SIZE 512
#define MAX_ITEM_PER_SM 1024
#define MAX_TRANSACTION_PER_SM 128
#define MAX_ITEM_PER_TRANSACTION 32
#define MIN_SUPPORT 3
#define MAX_TRANSACTIONS 10000
#define MAX_NUM_ELEMENTS (MAX_TRANSACTIONS * MAX_ITEM_PER_TRANSACTION)

#endif

