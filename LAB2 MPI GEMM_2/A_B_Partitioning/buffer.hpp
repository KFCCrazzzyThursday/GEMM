class Matrix;
class BUFFER
{
public:
    double *buffer;
    int A_rows_per_process;
    int A_cols_per_process;
    int B_cols_per_process;
    int comm_size;
    int A_blocks;
    int B_blocks;

public:
    BUFFER() : buffer(nullptr), A_rows_per_process(0), A_cols_per_process(0), B_cols_per_process(0), A_blocks(0), B_blocks(0){};
    BUFFER(int ar, int ac, int bc, int cs) : buffer(nullptr),
                                             A_rows_per_process(ar), A_cols_per_process(ac), B_cols_per_process(bc),
                                             comm_size(cs),
                                             A_blocks(0), B_blocks(0){};
    ~BUFFER()
    {
        if (buffer != nullptr) {
            delete[] buffer;
            buffer = nullptr;
        }
    }
    int m_alignment();
    int k_alignment();
    void partition();
    void buffer_init(Matrix &A, const Matrix &B, const int A_pad_rows, const int B_pad_cols);
    inline int Get_A_size_per_process() const;
    inline int Get_B_size_per_process() const;
    inline int Get_Buffer_size() const;
};

inline int BUFFER::Get_A_size_per_process() const
{
    return A_rows_per_process * A_cols_per_process;
};

inline int BUFFER::Get_B_size_per_process() const
{
    return A_cols_per_process * B_cols_per_process;
};

inline int BUFFER::Get_Buffer_size() const
{
    return Get_A_size_per_process() + Get_B_size_per_process();
};