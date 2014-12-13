//#define TEST_PARAMS
#define INVALID 99999
#ifdef TEST_PARAMS
#define MAX_UNIQUE_ITEMS 32
#define MAX_PATTERN_SEARCH 5 //max n-frequent items to search
#define BLOCK_SIZE 32 // must be greater than MAX_ITEM_PER_TRANSACTION
#define MAX_ITEM_PER_SM 32// can be a maximum of 6k items, each 4 bytes
// this is number of transactions per SM
// where eahc transaction can have MAX_ITEM_PER_TRANSACTION items
#define MAX_TRANSACTION_PER_SM 32
#define MAX_ITEM_PER_TRANSACTION 32
#define MIN_SUPPORT 2
#define MAX_TRANSACTIONS 10
#define MAX_NUM_ELEMENTS (MAX_TRANSACTIONS * MAX_ITEM_PER_TRANSACTION)
#define mAX_PATTERN_SEARCH 5
#else

#define MAX_UNIQUE_ITEMS 64
#define MAX_PATTERN_SEARCH 5 
#define BLOCK_SIZE 512
#define MAX_ITEM_PER_SM 1024
#define MAX_TRANSACTION_PER_SM 64
#define MAX_ITEM_PER_TRANSACTION 64
#define MIN_SUPPORT 3
#define MAX_TRANSACTIONS 100000
#define MAX_NUM_ELEMENTS (MAX_TRANSACTIONS * MAX_ITEM_PER_TRANSACTION)
#define MAX_PATTERN_SEARCH 5
#endif

