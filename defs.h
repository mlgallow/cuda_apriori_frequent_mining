//#define TEST_PARAMS
#ifdef TEST_PARAMS

#define MAX_UNIQUE_ITEMS 16
#define MAX_PATTERN_SEARCH 5 //max n-frequent items to search
#define BLOCK_SIZE 8 // must be greater than MAX_ITEM_PER_TRANSACTION
#define SM_SIZE 8//128
#define MAX_ITEM_PER_SM 16// can be a maximum of 6k items, each 4 bytes
        // this is number of transactions per SM
        // where eahc transaction can have MAX_ITEM_PER_TRANSACTION items
#define MAX_TRANSACTION_PER_SM 8
#define MAX_ITEM_PER_TRANSACTION 6
#define SM_SHAPE (MAX_TRANSACTION_PER_SM * MAX_ITEM_PER_TRANSACTION)
#define MIN_SUPPROT 3
#define MAX_TRANSACTIONS 32
#define MAX_NUM_ELEMENTS (MAX_TRANSACTIONS * MAX_ITEM_PER_TRANSACTION)

#else

#define MAX_UNIQUE_ITEMS 512
#define MAX_PATTERN_SEARCH 5 
#define BLOCK_SIZE 1024
#define SM_SIZE 8//128
#define MAX_ITEM_PER_SM 16
#define MAX_TRANSACTION_PER_SM 128
#define MAX_ITEM_PER_TRANSACTION 32
#define SM_SHAPE (MAX_TRANSACTION_PER_SM * MAX_ITEM_PER_TRANSACTION)
#define MIN_SUPPROT 3
#define MAX_TRANSACTIONS 10000
#define MAX_NUM_ELEMENTS (MAX_TRANSACTIONS * MAX_ITEM_PER_TRANSACTION)

#endif

