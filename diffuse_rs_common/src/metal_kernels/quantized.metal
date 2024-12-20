#include <metal_stdlib>

using namespace metal;

#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;

#else

/////////////////////////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////////////////////////

constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {
  // Check for nan
  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask) {
    return uint16_t(as_type<uint32_t>(0x7FC0));
  }
  // Take bits
  uint32_t float_bits = as_type<uint32_t>(x);

  // Round to nearest even
  float_bits += ((float_bits >> 16) & 1) + as_type<uint32_t>(0x7FFF);

  // Take upper 16 bits
  return float_bits >> 16;
}

constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {
  // Upper 16 bits are the data and lower 16 bits are 0s
  return as_type<float>((uint32_t)x << 16);
}

struct _MLX_BFloat16;

template <typename T>
static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;

/////////////////////////////////////////////////////////////////////////////
// Bfloat struct
/////////////////////////////////////////////////////////////////////////////

struct _MLX_BFloat16 {
  /////////////////////////////////////////////////////////////////////////////
  // Constructors
  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  struct bits_to_bfloat_struct {};
  static constexpr METAL_FUNC bits_to_bfloat_struct bits_to_bfloat() {
    return bits_to_bfloat_struct();
  }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t bits, bits_to_bfloat_struct)
      : bits_(bits) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions to bfloat

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  /////////////////////////////////////////////////////////////////////////////
  // Conversions from bfloat

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const constant {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
};

/////////////////////////////////////////////////////////////////////////////
// Bfloat operators
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Unary ops
constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 x) {
  return -static_cast<float>(x);
}

/////////////////////////////////////////////////////////////////////////////
// Binary operators
#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype) \
  constexpr METAL_FUNC otype __operator__(atype lhs, btype rhs) {           \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);          \
  }

#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype)    \
  constexpr METAL_FUNC otype __operator__(_MLX_BFloat16 lhs, itype rhs) { \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }                                                                       \
  constexpr METAL_FUNC otype __operator__(itype lhs, _MLX_BFloat16 rhs) { \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);        \
  }

/////////////////////////////////////////////////////////////////////////////
// Arithmetic Operators
#define bfloat_binop(_op_, _operator_)                                       \
  bfloat_binop_base(                                                         \
      _op_, _operator_, _MLX_BFloat16, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(_op_, _operator_, float, float, float);                \
  bfloat_binop_helper(_op_, _operator_, float, half, float);                 \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);     \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);      \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);

bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);

/////////////////////////////////////////////////////////////////////////////
// Comparison ops
#define bfloat_compop(__op__, __operator__)                             \
  bfloat_binop_base(                                                    \
      __op__, __operator__, bool, _MLX_BFloat16, _MLX_BFloat16, float); \
  bfloat_binop_helper(__op__, __operator__, bool, float, float);        \
  bfloat_binop_helper(__op__, __operator__, bool, half, float);         \
  bfloat_binop_helper(__op__, __operator__, bool, int32_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint32_t, float);     \
  bfloat_binop_helper(__op__, __operator__, bool, int64_t, float);      \
  bfloat_binop_helper(__op__, __operator__, bool, uint64_t, float);

bfloat_compop(>, operator>);
bfloat_compop(<, operator<);
bfloat_compop(>=, operator>=);
bfloat_compop(<=, operator<=);
bfloat_compop(==, operator==);
bfloat_compop(!=, operator!=);

#undef bfloat_compop
#undef bfloat_binop_base
#undef bfloat_binop_helper
#undef bfloat_binop

/////////////////////////////////////////////////////////////////////////////
// Inplace Operators
#define bfloat_inplace_op_helper(__op__, __operator__, itype, addr_space) \
  constexpr METAL_FUNC addr_space _MLX_BFloat16& __operator__(            \
      addr_space _MLX_BFloat16& lhs, itype rhs) {                         \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);         \
    return lhs;                                                           \
  }                                                                       \
  constexpr METAL_FUNC addr_space itype& __operator__(                    \
      addr_space itype& lhs, _MLX_BFloat16 rhs) {                         \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);         \
    return lhs;                                                           \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__, itype) \
  bfloat_inplace_op_helper(__op__, __operator__, itype, device);         \
  bfloat_inplace_op_helper(__op__, __operator__, itype, thread);         \
  bfloat_inplace_op_helper(__op__, __operator__, itype, threadgroup);

#define bfloat_inplace_op(itype)                             \
  bfloat_inplace_op_addr_space_helper(+, operator+=, itype); \
  bfloat_inplace_op_addr_space_helper(-, operator-=, itype); \
  bfloat_inplace_op_addr_space_helper(*, operator*=, itype); \
  bfloat_inplace_op_addr_space_helper(/, operator/=, itype);

bfloat_inplace_op(float);
bfloat_inplace_op(half);
bfloat_inplace_op(int16_t);
bfloat_inplace_op(int32_t);
bfloat_inplace_op(int64_t);
bfloat_inplace_op(uint16_t);
bfloat_inplace_op(uint32_t);
bfloat_inplace_op(uint64_t);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper
#undef bfloat_inplace_op

#define bfloat_inplace_op_helper(__op__, __operator__, addr_space) \
  constexpr METAL_FUNC addr_space _MLX_BFloat16& __operator__(     \
      addr_space _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) {          \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);  \
    return lhs;                                                    \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__) \
  bfloat_inplace_op_helper(__op__, __operator__, device);         \
  bfloat_inplace_op_helper(__op__, __operator__, thread);         \
  bfloat_inplace_op_helper(__op__, __operator__, threadgroup);

bfloat_inplace_op_addr_space_helper(+, operator+=);
bfloat_inplace_op_addr_space_helper(-, operator-=);
bfloat_inplace_op_addr_space_helper(*, operator*=);
bfloat_inplace_op_addr_space_helper(/, operator/=);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper

/////////////////////////////////////////////////////////////////////////////
// Bfloat typedef
/////////////////////////////////////////////////////////////////////////////

typedef struct _MLX_BFloat16 bfloat16_t;

#endif

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

typedef matrix<bfloat16_t, 4, 4> bfloat4x4;

// QK = number of values after dequantization
// QK_K = super-block size

#define QK_K 256
#define K_SCALE_SIZE 12

#define QK4_0 32
typedef struct {
    half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(half) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    union {
        struct {
            half d; // delta
            half m; // min
        };
        half2 dm;
    };
    uint8_t qs[QK4_1 / 2]; // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(half) + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
typedef struct {
    half d;           // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(half) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
typedef struct {
    union {
        struct {
            half d; // delta
            half m; // min
        };
        half2 dm;
    };
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_1 / 2]; // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(half) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
typedef struct {
    half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(half) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
typedef struct {
    union {
        struct {
            half d; // delta
            half s; // d * sum(qs[i])
        };
        half2 ds;
    };
    int8_t qs[QK8_1]; // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(half) + QK8_1, "wrong q8_1 block size/padding");

typedef struct {
    half d[4];        // deltas for 4 q4_0 blocks
    uint8_t qs[QK4_0 * 2]; // nibbles / quants for 4 q4_0 blocks
} block_q4_0x4;
static_assert(sizeof(block_q4_0x4) == 4 * sizeof(half) + QK4_0 * 2, "wrong q4_0x4 block size/padding");

typedef struct {
    half d[8];        // deltas for 8 q4_0 blocks
    uint8_t qs[QK4_0 * 4]; // nibbles / quants for 8 q4_0 blocks
} block_q4_0x8;
static_assert(sizeof(block_q4_0x8) == 8 * sizeof(half) + QK4_0 * 4, "wrong q4_0x8 block size/padding");

typedef struct {
    half d[4];        // deltas for 4 q8_0 blocks
    int8_t qs[QK8_0 * 4];  // quants for 4 q8_0 blocks
} block_q8_0x4;
static_assert(sizeof(block_q8_0x4) == 4 * sizeof(half) + QK8_0 * 4, "wrong q8_0x4 block size/padding");

typedef struct {
    half d[8];        // deltas for 8 q8_0 blocks
    int8_t qs[QK8_0 * 8];  // quants for 8 q8_0 blocks
} block_q8_0x8;
static_assert(sizeof(block_q8_0x8) == 8 * sizeof(half) + QK8_0 * 8, "wrong q8_0x8 block size/padding");

//
// Ternary quantization
//

// 1.6875 bpw
typedef struct {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5]; // 5 elements per byte (3^5 = 243 < 256)
    uint8_t qh[QK_K/64]; // 4 elements per byte
    half d;
} block_tq1_0;
static_assert(sizeof(block_tq1_0) == sizeof(half) + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5, "wrong tq1_0 block size/padding");

// 2.0625 bpw
typedef struct {
    uint8_t qs[QK_K/4]; // 2 bits per element
    half d;
} block_tq2_0;
static_assert(sizeof(block_tq2_0) == sizeof(half) + QK_K / 4, "wrong tq2_0 block size/padding");

//
// Super-block quantization structures
//

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    union {
        struct {
            half d;    // super-block scale for quantized scales
            half dmin; // super-block scale for quantized mins
        };
        half2 dm;
    };
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(half) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

// 3-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 3.4375 bits per weight
typedef struct {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    half d;           // super-block scale
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(half) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    union {
        struct {
            half d;    // super-block scale for quantized scales
            half dmin; // super-block scale for quantized mins
        };
        half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(half) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");

// 5-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight
typedef struct {
    union {
        struct {
            half d;    // super-block scale for quantized scales
            half dmin; // super-block scale for quantized mins
        };
        half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(half) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    half d;             // super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(half) + QK_K / 16 + 3*QK_K/4, "wrong q6_K block size/padding");

// This is only used for intermediate quantization and dot products
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");

// (Almost) "true" 2-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 2.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct {
    half d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == sizeof(half) + QK_K/8*sizeof(uint16_t), "wrong iq2_xxs block size/padding");

// 2.3125 bpw quants
typedef struct {
    half d;
    uint16_t qs[QK_K/8];
    uint8_t  scales[QK_K/32];
} block_iq2_xs;
static_assert(sizeof(block_iq2_xs) == sizeof(half) + QK_K/8*sizeof(uint16_t) + QK_K/32, "wrong iq2_xs block size/padding");

// 2.5625 bpw quants
typedef struct {
    half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t scales[QK_K/32];
} block_iq2_s;
static_assert(sizeof(block_iq2_s) == sizeof(half) + QK_K/4 + QK_K/16, "wrong iq2_s block size/padding");

// (Almost) "true" 3-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 3.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct {
    half d;
    uint8_t qs[3*QK_K/8];
} block_iq3_xxs;
static_assert(sizeof(block_iq3_xxs) == sizeof(half) + 3*(QK_K/8), "wrong iq3_xxs block size/padding");

// 3.4375 bpw
#define IQ3S_N_SCALE QK_K/64
typedef struct {
    half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;
static_assert(sizeof(block_iq3_s) == sizeof(half) + 13*(QK_K/32) + IQ3S_N_SCALE, "wrong iq3_s block size/padding");

// 1.5625 bpw
typedef struct {
    half d;
    uint8_t  qs[QK_K/8];
    uint16_t qh[QK_K/32];
} block_iq1_s;
static_assert(sizeof(block_iq1_s) == sizeof(half) + QK_K/8 + QK_K/16, "wrong iq1_s block size/padding");

// 1.75 bpw
typedef struct {
    uint8_t  qs[QK_K/8];      // grid index, low 8 bits
    uint8_t  qh[QK_K/16];     // grid index, high 3 bits + grid shift bit (for two groups of 8)
    uint8_t  scales[QK_K/32]; // 3-bit block scales (4-bit if QK_K == 64)
} block_iq1_m;
static_assert(sizeof(block_iq1_m) == QK_K/8 + QK_K/16 + QK_K/32, "wrong iq1_m block size/padding");

// Used by IQ1_M quants
typedef union {
    half f16;
    uint16_t  u16;
} iq1m_scale_t;

// Non-linear quants
#define QK4_NL 32
typedef struct {
    half d;
    uint8_t qs[QK4_NL/2];
} block_iq4_nl;
static_assert(sizeof(block_iq4_nl) == sizeof(half) + QK4_NL/2, "wrong iq4_nl block size/padding");

typedef struct {
    half d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/2];
} block_iq4_xs;
static_assert(sizeof(block_iq4_xs) == sizeof(half) + sizeof(uint16_t) + QK_K/64 + QK_K/2, "wrong iq4_xs block size/padding");

#define GGML_TABLE_BEGIN(type, name, size) static const constant type name[size] = {
#define GGML_TABLE_END() };

GGML_TABLE_BEGIN(uint8_t, kmask_iq2xs, 8)
    1, 2, 4, 8, 16, 32, 64, 128
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint8_t, ksigns_iq2xs, 128)
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
GGML_TABLE_END()

//#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
GGML_TABLE_BEGIN(uint64_t, ksigns64, 128)
    0x0000000000000000, 0xff000000000000ff, 0xff0000000000ff00, 0x000000000000ffff,
    0xff00000000ff0000, 0x0000000000ff00ff, 0x0000000000ffff00, 0xff00000000ffffff,
    0xff000000ff000000, 0x00000000ff0000ff, 0x00000000ff00ff00, 0xff000000ff00ffff,
    0x00000000ffff0000, 0xff000000ffff00ff, 0xff000000ffffff00, 0x00000000ffffffff,
    0xff0000ff00000000, 0x000000ff000000ff, 0x000000ff0000ff00, 0xff0000ff0000ffff,
    0x000000ff00ff0000, 0xff0000ff00ff00ff, 0xff0000ff00ffff00, 0x000000ff00ffffff,
    0x000000ffff000000, 0xff0000ffff0000ff, 0xff0000ffff00ff00, 0x000000ffff00ffff,
    0xff0000ffffff0000, 0x000000ffffff00ff, 0x000000ffffffff00, 0xff0000ffffffffff,
    0xff00ff0000000000, 0x0000ff00000000ff, 0x0000ff000000ff00, 0xff00ff000000ffff,
    0x0000ff0000ff0000, 0xff00ff0000ff00ff, 0xff00ff0000ffff00, 0x0000ff0000ffffff,
    0x0000ff00ff000000, 0xff00ff00ff0000ff, 0xff00ff00ff00ff00, 0x0000ff00ff00ffff,
    0xff00ff00ffff0000, 0x0000ff00ffff00ff, 0x0000ff00ffffff00, 0xff00ff00ffffffff,
    0x0000ffff00000000, 0xff00ffff000000ff, 0xff00ffff0000ff00, 0x0000ffff0000ffff,
    0xff00ffff00ff0000, 0x0000ffff00ff00ff, 0x0000ffff00ffff00, 0xff00ffff00ffffff,
    0xff00ffffff000000, 0x0000ffffff0000ff, 0x0000ffffff00ff00, 0xff00ffffff00ffff,
    0x0000ffffffff0000, 0xff00ffffffff00ff, 0xff00ffffffffff00, 0x0000ffffffffffff,
    0xffff000000000000, 0x00ff0000000000ff, 0x00ff00000000ff00, 0xffff00000000ffff,
    0x00ff000000ff0000, 0xffff000000ff00ff, 0xffff000000ffff00, 0x00ff000000ffffff,
    0x00ff0000ff000000, 0xffff0000ff0000ff, 0xffff0000ff00ff00, 0x00ff0000ff00ffff,
    0xffff0000ffff0000, 0x00ff0000ffff00ff, 0x00ff0000ffffff00, 0xffff0000ffffffff,
    0x00ff00ff00000000, 0xffff00ff000000ff, 0xffff00ff0000ff00, 0x00ff00ff0000ffff,
    0xffff00ff00ff0000, 0x00ff00ff00ff00ff, 0x00ff00ff00ffff00, 0xffff00ff00ffffff,
    0xffff00ffff000000, 0x00ff00ffff0000ff, 0x00ff00ffff00ff00, 0xffff00ffff00ffff,
    0x00ff00ffffff0000, 0xffff00ffffff00ff, 0xffff00ffffffff00, 0x00ff00ffffffffff,
    0x00ffff0000000000, 0xffffff00000000ff, 0xffffff000000ff00, 0x00ffff000000ffff,
    0xffffff0000ff0000, 0x00ffff0000ff00ff, 0x00ffff0000ffff00, 0xffffff0000ffffff,
    0xffffff00ff000000, 0x00ffff00ff0000ff, 0x00ffff00ff00ff00, 0xffffff00ff00ffff,
    0x00ffff00ffff0000, 0xffffff00ffff00ff, 0xffffff00ffffff00, 0x00ffff00ffffffff,
    0xffffffff00000000, 0x00ffffff000000ff, 0x00ffffff0000ff00, 0xffffffff0000ffff,
    0x00ffffff00ff0000, 0xffffffff00ff00ff, 0xffffffff00ffff00, 0x00ffffff00ffffff,
    0x00ffffffff000000, 0xffffffffff0000ff, 0xffffffffff00ff00, 0x00ffffffff00ffff,
    0xffffffffffff0000, 0x00ffffffffff00ff, 0x00ffffffffffff00, 0xffffffffffffffff,
GGML_TABLE_END()
//#endif


GGML_TABLE_BEGIN(uint64_t, iq2xxs_grid, 256)
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint64_t, iq2xs_grid, 512)
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint64_t, iq2s_grid, 1024)
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint32_t, iq3xxs_grid, 256)
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
GGML_TABLE_END()

GGML_TABLE_BEGIN(uint32_t, iq3s_grid, 512)
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
GGML_TABLE_END()

#define NGRID_IQ1S 2048
#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f
GGML_TABLE_BEGIN(uint32_t, iq1s_grid_gpu, NGRID_IQ1S)
    0x00000000, 0x00000002, 0x00000101, 0x00000200, 0x00000202, 0x00010001, 0x00010101, 0x00020000,
    0x00020002, 0x00020200, 0x00020202, 0x01000101, 0x01010001, 0x01010100, 0x01010102, 0x01020101,
    0x02000000, 0x02000002, 0x02000200, 0x02000202, 0x02010101, 0x02020000, 0x02020002, 0x02020200,
    0x02020202, 0x00000110, 0x00000111, 0x00010011, 0x00010110, 0x00010112, 0x00010211, 0x00010212,
    0x00020111, 0x01000011, 0x01000112, 0x01000211, 0x01010012, 0x01010111, 0x01010212, 0x01020011,
    0x01020110, 0x01020112, 0x01020210, 0x02000111, 0x02010011, 0x02010110, 0x02010112, 0x02020111,
    0x00000020, 0x00000022, 0x00000220, 0x00000222, 0x00010121, 0x00020020, 0x00020022, 0x00020220,
    0x00020222, 0x01000121, 0x01010021, 0x01010221, 0x01020120, 0x01020221, 0x02000020, 0x02000022,
    0x02000220, 0x02000222, 0x02010021, 0x02010121, 0x02010221, 0x02020020, 0x02020022, 0x02020220,
    0x02020222, 0x00011001, 0x00011100, 0x00011102, 0x00021101, 0x01001001, 0x01001201, 0x01011101,
    0x01011202, 0x01021100, 0x01021101, 0x02011001, 0x02011201, 0x02021101, 0x00001011, 0x00001110,
    0x00001111, 0x00001112, 0x00011111, 0x00011210, 0x00011212, 0x00021211, 0x01001010, 0x01001111,
    0x01001212, 0x01011010, 0x01011011, 0x01011110, 0x01011111, 0x01011112, 0x01011211, 0x01021010,
    0x01021012, 0x01021111, 0x01021210, 0x01021212, 0x02001011, 0x02011011, 0x02011111, 0x02011210,
    0x02011212, 0x02021011, 0x02021110, 0x02021111, 0x02021112, 0x02021211, 0x00011120, 0x00011221,
    0x01001021, 0x01001120, 0x01011020, 0x01011022, 0x01011121, 0x01011220, 0x01021020, 0x01021021,
    0x01021122, 0x01021221, 0x02001121, 0x02011021, 0x02011120, 0x02011221, 0x00002000, 0x00002002,
    0x00002200, 0x00002202, 0x00012101, 0x00022000, 0x00022002, 0x00022200, 0x00022202, 0x01002101,
    0x01012001, 0x01012102, 0x01022101, 0x02002000, 0x02002002, 0x02002200, 0x02002202, 0x02012101,
    0x02022000, 0x02022002, 0x02022200, 0x02022202, 0x00002111, 0x00012011, 0x00012110, 0x00012211,
    0x00022110, 0x00022111, 0x01002011, 0x01012010, 0x01012011, 0x01012111, 0x01022011, 0x01022110,
    0x01022211, 0x02012011, 0x02012110, 0x02012112, 0x02012211, 0x02022111, 0x00002020, 0x00002022,
    0x00002220, 0x00002222, 0x00012121, 0x00022020, 0x00022022, 0x00022220, 0x00022222, 0x01002121,
    0x01012021, 0x01012221, 0x01022021, 0x01022121, 0x02002020, 0x02002022, 0x02002121, 0x02002220,
    0x02002222, 0x02012121, 0x02022020, 0x02022022, 0x02022220, 0x02022222, 0x00110000, 0x00110001,
    0x00110100, 0x00110201, 0x00120100, 0x00120101, 0x01100001, 0x01100100, 0x01110000, 0x01110101,
    0x01110200, 0x01120001, 0x01120100, 0x01120101, 0x01120201, 0x02110001, 0x02110100, 0x02110102,
    0x02120001, 0x02120101, 0x00100011, 0x00100110, 0x00100112, 0x00100211, 0x00110010, 0x00110012,
    0x00110111, 0x00110210, 0x00120011, 0x00120110, 0x00120211, 0x01100111, 0x01100212, 0x01110010,
    0x01110011, 0x01110012, 0x01110110, 0x01110111, 0x01110112, 0x01110211, 0x01120010, 0x01120111,
    0x02100110, 0x02110012, 0x02110111, 0x02120011, 0x02120110, 0x00110021, 0x00110120, 0x00110122,
    0x00120121, 0x01100020, 0x01100122, 0x01100221, 0x01110022, 0x01110121, 0x01110220, 0x01110222,
    0x01120120, 0x01120122, 0x02100121, 0x02110021, 0x02110120, 0x02110122, 0x02120121, 0x00101001,
    0x00101102, 0x00101201, 0x00111100, 0x00111101, 0x00111200, 0x00111201, 0x00121001, 0x00121102,
    0x01101001, 0x01101101, 0x01101102, 0x01101200, 0x01101202, 0x01111001, 0x01111100, 0x01111101,
    0x01111102, 0x01111201, 0x01121002, 0x01121101, 0x01121200, 0x02101100, 0x02101201, 0x02111000,
    0x02111100, 0x02111101, 0x02111200, 0x02111201, 0x02111202, 0x02121001, 0x02121100, 0x02121101,
    0x02121201, 0x00101012, 0x00101111, 0x00101212, 0x00111011, 0x00111110, 0x00111111, 0x00111112,
    0x00111211, 0x00121010, 0x00121012, 0x00121111, 0x00121210, 0x00121212, 0x01101011, 0x01101110,
    0x01101111, 0x01101112, 0x01111011, 0x01111012, 0x01111110, 0x01111111, 0x01111112, 0x01111211,
    0x01111212, 0x01121011, 0x01121110, 0x01121111, 0x01121112, 0x01121211, 0x02101010, 0x02101012,
    0x02101110, 0x02101111, 0x02101210, 0x02101212, 0x02111010, 0x02111011, 0x02111110, 0x02111111,
    0x02111112, 0x02111211, 0x02111212, 0x02121010, 0x02121012, 0x02121111, 0x00101021, 0x00101120,
    0x00101121, 0x00101122, 0x00111121, 0x00111122, 0x00111220, 0x00111222, 0x00121021, 0x00121122,
    0x01101020, 0x01101022, 0x01101120, 0x01101121, 0x01101220, 0x01101222, 0x01111021, 0x01111121,
    0x01111122, 0x01111220, 0x01111221, 0x01121021, 0x01121120, 0x01121121, 0x01121220, 0x01121221,
    0x01121222, 0x02101122, 0x02101222, 0x02111022, 0x02111121, 0x02121120, 0x02121221, 0x00112001,
    0x00112102, 0x00122101, 0x01102001, 0x01102100, 0x01102102, 0x01102201, 0x01112000, 0x01112101,
    0x01112200, 0x01112202, 0x01122000, 0x01122001, 0x01122100, 0x01122102, 0x01122201, 0x02102101,
    0x02112001, 0x02112100, 0x02122101, 0x00112010, 0x00112012, 0x00112111, 0x00112212, 0x00122011,
    0x00122111, 0x01102012, 0x01102110, 0x01102111, 0x01102210, 0x01112011, 0x01112110, 0x01112111,
    0x01112112, 0x01112211, 0x01112212, 0x01122010, 0x01122111, 0x01122212, 0x02102211, 0x02112011,
    0x02112012, 0x02112111, 0x02112210, 0x02122011, 0x02122112, 0x02122211, 0x00102221, 0x00112122,
    0x00122120, 0x00122122, 0x01102120, 0x01102122, 0x01102221, 0x01112020, 0x01112022, 0x01112121,
    0x01112220, 0x01122021, 0x01122122, 0x01122221, 0x02102121, 0x02112021, 0x02112122, 0x02112222,
    0x00200000, 0x00200002, 0x00200200, 0x00200202, 0x00210101, 0x00220000, 0x00220002, 0x00220101,
    0x00220200, 0x00220202, 0x01200101, 0x01210001, 0x01210201, 0x01220001, 0x01220101, 0x02200000,
    0x02200002, 0x02200200, 0x02200202, 0x02210101, 0x02220000, 0x02220002, 0x02220101, 0x02220200,
    0x02220202, 0x00200111, 0x00210011, 0x00210110, 0x00210211, 0x00220111, 0x01200012, 0x01200110,
    0x01200211, 0x01210111, 0x01210210, 0x01210212, 0x01220011, 0x01220110, 0x01220111, 0x01220112,
    0x02200111, 0x02210010, 0x02210112, 0x02210211, 0x02220111, 0x00200021, 0x00200220, 0x00200222,
    0x00210021, 0x00210121, 0x00220020, 0x00220022, 0x00220220, 0x00220222, 0x01200121, 0x01210021,
    0x01210122, 0x01210221, 0x01220121, 0x02200021, 0x02200220, 0x02200222, 0x02210021, 0x02210121,
    0x02220020, 0x02220022, 0x02220220, 0x02220222, 0x00201101, 0x00211100, 0x00211102, 0x00211201,
    0x00221101, 0x01201100, 0x01201101, 0x01201102, 0x01201201, 0x01211002, 0x01211101, 0x01211200,
    0x01211202, 0x01221102, 0x02201101, 0x02211001, 0x02211100, 0x02211201, 0x02221001, 0x02221101,
    0x00201211, 0x00211111, 0x00221011, 0x00221211, 0x01201010, 0x01201111, 0x01201210, 0x01211011,
    0x01211110, 0x01211111, 0x01211211, 0x01221012, 0x01221111, 0x01221210, 0x02201211, 0x02211010,
    0x02211110, 0x02211111, 0x02211210, 0x02211212, 0x02221011, 0x02221110, 0x02221112, 0x02221211,
    0x00201121, 0x00211020, 0x00211022, 0x00211221, 0x00221121, 0x01201021, 0x01201221, 0x01211121,
    0x01221020, 0x01221021, 0x01221221, 0x02201120, 0x02201122, 0x02211020, 0x02211222, 0x00202000,
    0x00202002, 0x00202200, 0x00202202, 0x00212101, 0x00222000, 0x00222002, 0x00222200, 0x00222202,
    0x01202101, 0x01212001, 0x01212100, 0x01222101, 0x02202000, 0x02202002, 0x02202200, 0x02202202,
    0x02222000, 0x02222002, 0x02222200, 0x02222202, 0x00202211, 0x00212011, 0x00212110, 0x00212211,
    0x00222111, 0x01202112, 0x01202211, 0x01212012, 0x01212111, 0x01222011, 0x01222110, 0x01222112,
    0x01222211, 0x02202111, 0x02212010, 0x02212112, 0x02212211, 0x02222110, 0x02222111, 0x00202020,
    0x00202022, 0x00202220, 0x00202222, 0x00222020, 0x00222022, 0x00222220, 0x00222222, 0x01202121,
    0x01212021, 0x01212122, 0x01212221, 0x01222121, 0x02202020, 0x02202022, 0x02202220, 0x02202222,
    0x02212121, 0x02222020, 0x02222022, 0x02222220, 0x02222222, 0x10000101, 0x10010001, 0x10010102,
    0x10020101, 0x11000201, 0x11010002, 0x11010101, 0x11010200, 0x11010202, 0x11020001, 0x11020100,
    0x11020102, 0x12010100, 0x12010201, 0x12020001, 0x12020102, 0x10000010, 0x10000011, 0x10000110,
    0x10000112, 0x10000211, 0x10010012, 0x10010111, 0x10010112, 0x10010210, 0x10010212, 0x10020011,
    0x10020112, 0x10020211, 0x11000111, 0x11000210, 0x11000212, 0x11010011, 0x11010110, 0x11010111,
    0x11010112, 0x11010211, 0x11010212, 0x11020111, 0x11020210, 0x11020212, 0x12000011, 0x12000110,
    0x12000112, 0x12010010, 0x12010012, 0x12010111, 0x12020010, 0x12020011, 0x12020012, 0x10000121,
    0x10010021, 0x10010120, 0x10010122, 0x10020121, 0x11000021, 0x11010022, 0x11010121, 0x11010222,
    0x11020120, 0x11020221, 0x12000221, 0x12010120, 0x12020121, 0x10001001, 0x10011101, 0x10011201,
    0x10021201, 0x11001101, 0x11001200, 0x11001202, 0x11011001, 0x11011100, 0x11011101, 0x11011102,
    0x11021001, 0x11021002, 0x11021101, 0x11021200, 0x11021202, 0x12001001, 0x12001102, 0x12001201,
    0x12011000, 0x12011002, 0x12011101, 0x12021000, 0x12021001, 0x12021201, 0x10001011, 0x10001012,
    0x10001111, 0x10001212, 0x10011011, 0x10011110, 0x10011111, 0x10011112, 0x10011211, 0x10021010,
    0x10021111, 0x10021212, 0x11001011, 0x11001110, 0x11001111, 0x11001112, 0x11001211, 0x11011010,
    0x11011011, 0x11011110, 0x11011111, 0x11011112, 0x11011210, 0x11011211, 0x11021011, 0x11021110,
    0x11021111, 0x11021112, 0x11021211, 0x12001012, 0x12001110, 0x12001111, 0x12001210, 0x12011011,
    0x12011110, 0x12011111, 0x12011112, 0x12011211, 0x12011212, 0x12021111, 0x12021210, 0x12021212,
    0x10001021, 0x10001121, 0x10001221, 0x10011120, 0x10011121, 0x10011220, 0x10011222, 0x10021021,
    0x10021120, 0x10021221, 0x11001020, 0x11001022, 0x11001121, 0x11001220, 0x11011020, 0x11011021,
    0x11011022, 0x11011121, 0x11011122, 0x11011221, 0x11021022, 0x11021121, 0x11021220, 0x12001021,
    0x12001121, 0x12001222, 0x12011120, 0x12011121, 0x12021021, 0x12021120, 0x12021122, 0x10002101,
    0x10012001, 0x10012101, 0x10012202, 0x10022101, 0x11002002, 0x11002201, 0x11012000, 0x11012101,
    0x11012200, 0x11022001, 0x11022100, 0x11022102, 0x11022201, 0x12002101, 0x12012001, 0x12012100,
    0x12012102, 0x12012201, 0x12022101, 0x10002011, 0x10002111, 0x10002112, 0x10002212, 0x10012010,
    0x10012110, 0x10012111, 0x10012210, 0x10022011, 0x10022110, 0x10022112, 0x11002010, 0x11002111,
    0x11002212, 0x11012011, 0x11012012, 0x11012110, 0x11012111, 0x11012112, 0x11012211, 0x11022010,
    0x11022012, 0x11022111, 0x11022112, 0x11022212, 0x12002112, 0x12002211, 0x12012012, 0x12012111,
    0x12012112, 0x12012210, 0x12022011, 0x12022110, 0x12022112, 0x12022211, 0x10012122, 0x11002120,
    0x11002122, 0x11002221, 0x11012121, 0x11012220, 0x11012222, 0x11022120, 0x11022221, 0x12012120,
    0x12022121, 0x10100001, 0x10100100, 0x10100101, 0x10100102, 0x10100201, 0x10110002, 0x10110101,
    0x10110202, 0x10120001, 0x10120100, 0x10120201, 0x11100000, 0x11100101, 0x11100200, 0x11110001,
    0x11110100, 0x11110101, 0x11110102, 0x11110201, 0x11120101, 0x11120200, 0x12100102, 0x12100201,
    0x12110101, 0x12110200, 0x12120000, 0x12120001, 0x12120102, 0x12120201, 0x10100111, 0x10100210,
    0x10100211, 0x10100212, 0x10110011, 0x10110110, 0x10110111, 0x10110112, 0x10110210, 0x10110211,
    0x10120010, 0x10120111, 0x10120112, 0x10120210, 0x10120212, 0x11100011, 0x11100110, 0x11100111,
    0x11100112, 0x11100211, 0x11110010, 0x11110011, 0x11110012, 0x11110110, 0x11110111, 0x11110112,
    0x11110210, 0x11110211, 0x11110212, 0x11120011, 0x11120110, 0x11120111, 0x11120112, 0x11120211,
    0x12100012, 0x12100111, 0x12110011, 0x12110110, 0x12110111, 0x12110112, 0x12110211, 0x12120010,
    0x12120111, 0x12120212, 0x10100021, 0x10100122, 0x10110022, 0x10110121, 0x10110222, 0x10120021,
    0x10120120, 0x11100022, 0x11100121, 0x11100222, 0x11110021, 0x11110120, 0x11110121, 0x11110122,
    0x11110221, 0x11120022, 0x11120121, 0x12100121, 0x12110020, 0x12110022, 0x12110121, 0x12110221,
    0x12110222, 0x12120120, 0x10101100, 0x10101101, 0x10111001, 0x10111100, 0x10111101, 0x10111102,
    0x10111200, 0x10111201, 0x10121001, 0x10121101, 0x10121200, 0x10121202, 0x11101001, 0x11101100,
    0x11101101, 0x11101102, 0x11101201, 0x11101202, 0x11111000, 0x11111001, 0x11111100, 0x11111101,
    0x11111102, 0x11111200, 0x11111201, 0x11111202, 0x11121001, 0x11121002, 0x11121100, 0x11121101,
    0x11121102, 0x11121201, 0x12101000, 0x12101200, 0x12101202, 0x12111001, 0x12111100, 0x12111101,
    0x12111102, 0x12111201, 0x12121001, 0x12121100, 0x12121101, 0x12121202, 0x10101011, 0x10101012,
    0x10101110, 0x10101111, 0x10101112, 0x10101211, 0x10111010, 0x10111011, 0x10111012, 0x10111110,
    0x10111111, 0x10111112, 0x10111211, 0x10111212, 0x10121011, 0x10121110, 0x10121111, 0x10121112,
    0x10121211, 0x11101010, 0x11101011, 0x11101012, 0x11101110, 0x11101111, 0x11101112, 0x11101210,
    0x11101211, 0x11111010, 0x11111011, 0x11111012, 0x11111110, 0x11111111, 0x11111112, 0x11111210,
    0x11111211, 0x11111212, 0x11121010, 0x11121011, 0x11121110, 0x11121111, 0x11121112, 0x11121210,
    0x11121211, 0x11121212, 0x12101011, 0x12101110, 0x12101111, 0x12101211, 0x12101212, 0x12111010,
    0x12111011, 0x12111110, 0x12111111, 0x12111112, 0x12111210, 0x12111211, 0x12121011, 0x12121110,
    0x12121111, 0x12121112, 0x12121211, 0x10101020, 0x10101021, 0x10101022, 0x10101120, 0x10101122,
    0x10101220, 0x10101221, 0x10111021, 0x10111120, 0x10111121, 0x10111220, 0x10111221, 0x10121020,
    0x10121021, 0x10121022, 0x10121120, 0x10121121, 0x10121122, 0x10121220, 0x10121221, 0x11101021,
    0x11101121, 0x11101122, 0x11101220, 0x11101221, 0x11101222, 0x11111020, 0x11111021, 0x11111022,
    0x11111120, 0x11111121, 0x11111122, 0x11111220, 0x11111221, 0x11111222, 0x11121021, 0x11121120,
    0x11121121, 0x11121221, 0x12101022, 0x12101121, 0x12101122, 0x12101220, 0x12101221, 0x12101222,
    0x12111021, 0x12111121, 0x12111222, 0x12121022, 0x12121121, 0x12121122, 0x12121220, 0x12121221,
    0x10102100, 0x10102101, 0x10102102, 0x10102201, 0x10112000, 0x10112101, 0x10112200, 0x10122001,
    0x10122202, 0x11102101, 0x11102200, 0x11102202, 0x11112001, 0x11112100, 0x11112101, 0x11112102,
    0x11112200, 0x11112201, 0x11122000, 0x11122002, 0x11122100, 0x11122101, 0x12102002, 0x12102201,
    0x12112000, 0x12112002, 0x12112101, 0x12112200, 0x12122001, 0x12122201, 0x10102011, 0x10102012,
    0x10102111, 0x10102212, 0x10112011, 0x10112110, 0x10112111, 0x10112112, 0x10112211, 0x10122111,
    0x11102011, 0x11102110, 0x11102111, 0x11102112, 0x11102211, 0x11112010, 0x11112011, 0x11112012,
    0x11112110, 0x11112111, 0x11112112, 0x11112210, 0x11112211, 0x11112212, 0x11122011, 0x11122110,
    0x11122111, 0x11122112, 0x11122211, 0x12102011, 0x12102111, 0x12102211, 0x12112011, 0x12112110,
    0x12112111, 0x12112112, 0x12112210, 0x12112211, 0x12122111, 0x10102120, 0x10102220, 0x10112121,
    0x10112222, 0x10122020, 0x10122121, 0x10122122, 0x10122221, 0x11102121, 0x11102220, 0x11102221,
    0x11112021, 0x11112121, 0x11112122, 0x11112220, 0x11112221, 0x11122022, 0x11122121, 0x11122220,
    0x11122222, 0x12102021, 0x12102222, 0x12112022, 0x12112121, 0x12112122, 0x12112220, 0x12112222,
    0x12122021, 0x10200101, 0x10210100, 0x10210102, 0x10210201, 0x10220101, 0x11200100, 0x11210000,
    0x11210101, 0x11210102, 0x11210200, 0x11210202, 0x11220001, 0x11220100, 0x11220102, 0x11220201,
    0x12200001, 0x12210102, 0x12220101, 0x10200011, 0x10200110, 0x10200112, 0x10200211, 0x10210012,
    0x10210111, 0x10220011, 0x10220012, 0x10220112, 0x10220211, 0x11200111, 0x11200211, 0x11210011,
    0x11210111, 0x11210112, 0x11210211, 0x11220111, 0x11220112, 0x11220212, 0x12200110, 0x12200212,
    0x12210012, 0x12210111, 0x12220011, 0x12220112, 0x12220211, 0x10210021, 0x10210122, 0x10210221,
    0x11200020, 0x11200021, 0x11200122, 0x11210121, 0x11210122, 0x11210220, 0x11220020, 0x12200121,
    0x12210021, 0x12210122, 0x12220121, 0x10211001, 0x10211002, 0x10211101, 0x10211102, 0x10211202,
    0x10221001, 0x10221102, 0x10221201, 0x11201000, 0x11201002, 0x11201101, 0x11201200, 0x11201202,
    0x11211001, 0x11211100, 0x11211101, 0x11211102, 0x11211201, 0x11211202, 0x11221000, 0x11221002,
    0x11221101, 0x12201100, 0x12201101, 0x12201201, 0x12211000, 0x12211002, 0x12211100, 0x12211101,
    0x12211102, 0x12211200, 0x12211202, 0x12221001, 0x12221100, 0x12221201, 0x10201111, 0x10201210,
    0x10201212, 0x10211011, 0x10211111, 0x10211112, 0x10211211, 0x11201110, 0x11201111, 0x11201112,
    0x11201211, 0x11211010, 0x11211011, 0x11211110, 0x11211111, 0x11211112, 0x11211211, 0x11221011,
    0x11221110, 0x11221111, 0x11221112, 0x11221211, 0x12201112, 0x12201211, 0x12201212, 0x12211011,
    0x12211111, 0x12211112, 0x12211211, 0x12211212, 0x12221012, 0x12221111, 0x12221112, 0x12221210,
    0x10201022, 0x10201221, 0x10211121, 0x10221020, 0x10221122, 0x10221220, 0x10221221, 0x11201020,
    0x11201121, 0x11201220, 0x11201222, 0x11211021, 0x11211120, 0x11211121, 0x11211122, 0x11211220,
    0x11211222, 0x11221020, 0x11221121, 0x11221220, 0x12201020, 0x12201022, 0x12201121, 0x12201222,
    0x12211120, 0x12211122, 0x12211220, 0x12211221, 0x12221020, 0x12221120, 0x12221122, 0x12221222,
    0x10212102, 0x10212201, 0x10222101, 0x11202001, 0x11212002, 0x11212101, 0x11212202, 0x11222001,
    0x11222201, 0x12202101, 0x12212001, 0x12212200, 0x12222102, 0x10202011, 0x10202110, 0x10212010,
    0x10212111, 0x10222011, 0x10222110, 0x10222112, 0x10222211, 0x11202010, 0x11202011, 0x11202111,
    0x11202112, 0x11202210, 0x11212011, 0x11212110, 0x11212111, 0x11212112, 0x11212211, 0x11222010,
    0x11222111, 0x11222212, 0x12202012, 0x12202110, 0x12202212, 0x12212111, 0x12222011, 0x12222110,
    0x12222111, 0x12222211, 0x10212021, 0x10212122, 0x10212220, 0x11202021, 0x11202120, 0x11202221,
    0x11212020, 0x11212121, 0x11212220, 0x11212222, 0x11222120, 0x11222121, 0x11222221, 0x12202122,
    0x12212120, 0x12212220, 0x12212222, 0x12222122, 0x20000000, 0x20000002, 0x20000200, 0x20000202,
    0x20020000, 0x20020002, 0x20020200, 0x20020202, 0x21000101, 0x21010000, 0x21010001, 0x21010100,
    0x21010102, 0x21010201, 0x21020101, 0x22000000, 0x22000002, 0x22000200, 0x22000202, 0x22010101,
    0x22020000, 0x22020002, 0x22020200, 0x22020202, 0x20000111, 0x20010011, 0x20010110, 0x20010112,
    0x20010211, 0x20020111, 0x21000011, 0x21000110, 0x21000211, 0x21010010, 0x21010012, 0x21010111,
    0x21010112, 0x21010210, 0x21010211, 0x21020110, 0x21020112, 0x21020211, 0x22000111, 0x22000211,
    0x22010110, 0x22010112, 0x22010211, 0x22020111, 0x20000020, 0x20000022, 0x20000220, 0x20000222,
    0x20010121, 0x20020020, 0x20020022, 0x20020220, 0x20020222, 0x21010021, 0x21010120, 0x21010221,
    0x21020121, 0x22000020, 0x22000022, 0x22000220, 0x22000222, 0x22010121, 0x22020020, 0x22020022,
    0x22020220, 0x22020222, 0x20011100, 0x20011201, 0x21001001, 0x21001100, 0x21011001, 0x21011101,
    0x21011202, 0x21021001, 0x21021100, 0x21021201, 0x22011100, 0x22011201, 0x20001011, 0x20001211,
    0x20011012, 0x20011111, 0x20011212, 0x20021112, 0x20021211, 0x21001010, 0x21001011, 0x21001111,
    0x21001210, 0x21011011, 0x21011110, 0x21011111, 0x21011112, 0x21011211, 0x21011212, 0x21021111,
    0x21021112, 0x21021210, 0x21021212, 0x22001011, 0x22001110, 0x22001112, 0x22001211, 0x22011010,
    0x22011012, 0x22011111, 0x22011210, 0x22021112, 0x20011021, 0x20011122, 0x20011221, 0x20021121,
    0x21001021, 0x21001120, 0x21001221, 0x21001222, 0x21011020, 0x21011121, 0x21011221, 0x21011222,
    0x21021021, 0x21021122, 0x21021222, 0x22001121, 0x22011021, 0x22011222, 0x22021120, 0x20002000,
    0x20002002, 0x20002200, 0x20002202, 0x20012101, 0x20022000, 0x20022002, 0x20022200, 0x20022202,
    0x21002001, 0x21002101, 0x21012001, 0x21012100, 0x21012201, 0x21022101, 0x21022201, 0x22002000,
    0x22002002, 0x22002200, 0x22002202, 0x22012101, 0x22022000, 0x22022002, 0x22022200, 0x22022202,
    0x20002111, 0x20002112, 0x20012011, 0x20012110, 0x20012112, 0x20022111, 0x21002011, 0x21002110,
    0x21002112, 0x21002211, 0x21012010, 0x21012012, 0x21012111, 0x21012212, 0x21022011, 0x21022110,
    0x22002111, 0x22012112, 0x22012211, 0x22022111, 0x20002020, 0x20002022, 0x20002220, 0x20002222,
    0x20012121, 0x20022020, 0x20022022, 0x20022220, 0x20022222, 0x21002121, 0x21012021, 0x21012120,
    0x21012122, 0x22002020, 0x22002022, 0x22002220, 0x22002222, 0x22012121, 0x22022020, 0x22022022,
    0x22022220, 0x22022222, 0x20100101, 0x20110001, 0x20110102, 0x20110200, 0x20110201, 0x20120101,
    0x21100001, 0x21100102, 0x21100201, 0x21110101, 0x21110200, 0x21110202, 0x21120201, 0x21120202,
    0x22100101, 0x22110001, 0x22110100, 0x22110102, 0x22110201, 0x22120101, 0x20100011, 0x20100110,
    0x20100112, 0x20100211, 0x20110010, 0x20110111, 0x20110210, 0x20110212, 0x20120011, 0x20120110,
    0x20120112, 0x20120211, 0x21100010, 0x21100111, 0x21110010, 0x21110011, 0x21110110, 0x21110111,
    0x21110112, 0x21110211, 0x21120012, 0x21120111, 0x22100110, 0x22100112, 0x22110012, 0x22110111,
    0x22110210, 0x22120011, 0x22120110, 0x22120112, 0x22120211, 0x20100121, 0x20110021, 0x20110120,
    0x20110221, 0x20120121, 0x21100120, 0x21100122, 0x21100221, 0x21110020, 0x21110022, 0x21110121,
    0x21110220, 0x21120122, 0x21120221, 0x22100121, 0x22110120, 0x22110122, 0x22120221, 0x20101001,
    0x20101100, 0x20101102, 0x20111000, 0x20111101, 0x20111200, 0x20121102, 0x21101000, 0x21101202,
    0x21111001, 0x21111100, 0x21111101, 0x21111102, 0x21111200, 0x21111201, 0x21121000, 0x21121001,
    0x21121002, 0x21121101, 0x22101100, 0x22101102, 0x22111002, 0x22111100, 0x22111101, 0x22111200,
    0x22121001, 0x22121201, 0x20101010, 0x20101111, 0x20101210, 0x20101212, 0x20111010, 0x20111011,
    0x20111110, 0x20111111, 0x20111112, 0x20111211, 0x20121011, 0x20121111, 0x20121211, 0x20121212,
    0x21101011, 0x21101110, 0x21101111, 0x21101112, 0x21101211, 0x21111010, 0x21111011, 0x21111012,
    0x21111110, 0x21111111, 0x21111112, 0x21111210, 0x21111211, 0x21111212, 0x21121011, 0x21121110,
    0x21121111, 0x21121112, 0x21121211, 0x22101011, 0x22101111, 0x22101210, 0x22111011, 0x22111012,
    0x22111110, 0x22111111, 0x22111112, 0x22111211, 0x22111212, 0x22121010, 0x22121012, 0x22121111,
    0x22121210, 0x22121212, 0x20101021, 0x20101120, 0x20111020, 0x20111121, 0x20111221, 0x20121020,
    0x20121122, 0x20121221, 0x21101121, 0x21101220, 0x21101221, 0x21111021, 0x21111022, 0x21111121,
    0x21111122, 0x21111221, 0x21121121, 0x21121220, 0x22101022, 0x22101120, 0x22101221, 0x22101222,
    0x22111022, 0x22111120, 0x22111121, 0x22121120, 0x22121122, 0x22121221, 0x20102101, 0x20112102,
    0x20112201, 0x20122101, 0x21102001, 0x21102102, 0x21112000, 0x21112002, 0x21112101, 0x21112102,
    0x21112202, 0x21122100, 0x21122101, 0x22102101, 0x22112001, 0x22112102, 0x22112201, 0x22122101,
    0x20102110, 0x20102112, 0x20102211, 0x20112010, 0x20112012, 0x20112111, 0x20112210, 0x20112212,
    0x20122010, 0x20122011, 0x20122110, 0x20122112, 0x21102010, 0x21102012, 0x21102111, 0x21102210,
    0x21102212, 0x21112011, 0x21112110, 0x21112111, 0x21112112, 0x21112211, 0x21122012, 0x21122111,
    0x21122112, 0x21122212, 0x22102011, 0x22102110, 0x22112010, 0x22112012, 0x22112111, 0x22112212,
    0x22122011, 0x22122112, 0x20102121, 0x20112121, 0x20122121, 0x21102120, 0x21102122, 0x21102221,
    0x21112020, 0x21112121, 0x21112220, 0x21122021, 0x22102121, 0x22112021, 0x22112120, 0x22112121,
    0x22112122, 0x20200000, 0x20200002, 0x20200200, 0x20200202, 0x20210101, 0x20220000, 0x20220002,
    0x20220200, 0x20220202, 0x21200101, 0x21210001, 0x21210100, 0x21210102, 0x21210201, 0x22200000,
    0x22200002, 0x22200200, 0x22200202, 0x22210101, 0x22220000, 0x22220002, 0x22220200, 0x22220202,
    0x20200111, 0x20200211, 0x20210011, 0x20210110, 0x20210112, 0x20210211, 0x20210212, 0x21200112,
    0x21200211, 0x21210011, 0x21210111, 0x21210210, 0x21210212, 0x21220011, 0x21220110, 0x22200111,
    0x22210010, 0x22210012, 0x22210112, 0x22210211, 0x20200022, 0x20200220, 0x20200222, 0x20210020,
    0x20210221, 0x20220022, 0x20220220, 0x20220222, 0x21200121, 0x21210021, 0x21210122, 0x21210221,
    0x21220121, 0x22200020, 0x22200022, 0x22200220, 0x22200222, 0x22210121, 0x22220020, 0x22220022,
    0x22220220, 0x22220222, 0x20211201, 0x20221101, 0x21201001, 0x21201100, 0x21211000, 0x21211100,
    0x21211101, 0x21211200, 0x21211202, 0x21221001, 0x21221101, 0x21221102, 0x21221200, 0x21221201,
    0x22201101, 0x20201112, 0x20201211, 0x20211010, 0x20211012, 0x20211111, 0x20211210, 0x20221112,
    0x20221211, 0x21201012, 0x21201111, 0x21211011, 0x21211110, 0x21211111, 0x21211112, 0x21211211,
    0x21221111, 0x21221212, 0x22201011, 0x22201110, 0x22201111, 0x22201112, 0x22201211, 0x22211012,
    0x22211111, 0x22211210, 0x20201121, 0x20211021, 0x20211122, 0x20211222, 0x20221021, 0x20221121,
    0x21201120, 0x21201122, 0x21201222, 0x21211022, 0x21211121, 0x21211122, 0x21211220, 0x21221020,
    0x21221022, 0x22201122, 0x22211020, 0x22211121, 0x22211122, 0x22211221, 0x22221021, 0x22221120,
    0x22221122, 0x20202000, 0x20202002, 0x20202200, 0x20202202, 0x20222000, 0x20222002, 0x20222200,
    0x20222202, 0x21212001, 0x21212100, 0x21212102, 0x21212201, 0x22202000, 0x22202002, 0x22202200,
    0x22202202, 0x22212101, 0x22222000, 0x22222002, 0x22222200, 0x22222202, 0x20202111, 0x20212110,
    0x20212211, 0x20222011, 0x20222111, 0x21202011, 0x21212010, 0x21212111, 0x21212212, 0x21222011,
    0x21222112, 0x21222211, 0x22212010, 0x22212112, 0x20202020, 0x20202022, 0x20202220, 0x20202222,
    0x20222020, 0x20222022, 0x20222220, 0x20222222, 0x21212021, 0x21212120, 0x21212122, 0x22202020,
    0x22202022, 0x22202220, 0x22202222, 0x22212121, 0x22222020, 0x22222022, 0x22222220, 0x22222222,
GGML_TABLE_END()


enum ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

// general-purpose kernel for addition, subtraction, multiplication and division of two tensors
// pros: works for non-contiguous tensors, supports broadcast across all dims
// cons: not very efficient
kernel void kernel_add(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        constant  int64_t & offs,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01 + offs;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1  + offs;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i10 = i0 % ne10;
        *((device float *)(dst_ptr + i0*nb0)) = *((device float *)(src0_ptr + i0*nb00)) + *((device float *)(src1_ptr + i10*nb10));
    }
}

kernel void kernel_sub(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        constant  int64_t & offs,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01 + offs;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1  + offs;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i10 = i0 % ne10;
        *((device float *)(dst_ptr + i0*nb0)) = *((device float *)(src0_ptr + i0*nb00)) - *((device float *)(src1_ptr + i10*nb10));
    }
}

kernel void kernel_mul(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i10 = i0 % ne10;
        *((device float *)(dst_ptr + i0*nb0)) = *((device float *)(src0_ptr + i0*nb00)) * *((device float *)(src1_ptr + i10*nb10));
    }
}

kernel void kernel_div(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i10 = i0 % ne10;
        *((device float *)(dst_ptr + i0*nb0)) = *((device float *)(src0_ptr + i0*nb00)) / *((device float *)(src1_ptr + i10*nb10));
    }
}

template<typename T>
kernel void kernel_repeat(
        device const char * src0,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3 % ne03;
    const int64_t i02 = i2 % ne02;
    const int64_t i01 = i1 % ne01;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    device       char * dst_ptr  = dst  +  i3*nb3  +  i2*nb2  +  i1*nb1 ;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i00 = i0 % ne00;
        *((device T *)(dst_ptr + i0*nb0)) = *((device T *)(src0_ptr + i00*nb00));
    }
}

typedef decltype(kernel_repeat<float>) kernel_repeat_t;

template [[host_name("kernel_repeat_f32")]] kernel kernel_repeat_t kernel_repeat<float>;
template [[host_name("kernel_repeat_f16")]] kernel kernel_repeat_t kernel_repeat<half>;
template [[host_name("kernel_repeat_i32")]] kernel kernel_repeat_t kernel_repeat<int>;
template [[host_name("kernel_repeat_i16")]] kernel kernel_repeat_t kernel_repeat<short>;

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_add_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant   uint64_t & nb [[buffer(28)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] + src1[tpig % nb];
}

kernel void kernel_sub_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant   uint64_t & nb [[buffer(28)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] - src1[tpig % nb];
}

kernel void kernel_mul_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant   uint64_t & nb  [[buffer(28)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig % nb];
}

kernel void kernel_div_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant   uint64_t & nb  [[buffer(28)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] / src1[tpig % nb];
}

kernel void kernel_scale(
        device const float * src0,
        device       float * dst,
        constant     float & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_scale_4(
        device const float4 * src0,
        device       float4 * dst,
        constant     float  & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_clamp(
        device const float * src0,
        device       float * dst,
        constant     float & min,
        constant     float & max,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] < min ? min : (src0[tpig] > max ? max : src0[tpig]);
}

kernel void kernel_relu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = max(0.0f, src0[tpig]);
}

kernel void kernel_sigmoid(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = 1.0f / (1.0f + exp(-src0[tpig]));
}

kernel void kernel_tanh(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];
    dst[tpig] = precise::tanh(x);
}

constant float GELU_COEF_A     = 0.044715f;
constant float GELU_QUICK_COEF = -1.702f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

kernel void kernel_gelu(
    device const float * src0,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];

    dst[tpig] = 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_gelu_4(
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    // BEWARE !!!
    // Simply using "tanh" instead of "precise::tanh" will sometimes results in NaNs!
    // This was observed with Falcon 7B and 40B models
    //
    dst[tpig] = 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_gelu_quick(
    device const float * src0,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];

    dst[tpig] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
}

kernel void kernel_gelu_quick_4(
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    dst[tpig] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
}

kernel void kernel_silu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_silu_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_sqr(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src0[tpig];
}

kernel void kernel_sqrt(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = sqrt(src0[tpig]);
}

kernel void kernel_sin(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = sin(src0[tpig]);
}

kernel void kernel_cos(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = cos(src0[tpig]);
}

kernel void kernel_sum_rows(
        device const float * src0,
        device       float * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tpig[[thread_position_in_grid]]) {
    int64_t i3 = tpig.z;
    int64_t i2 = tpig.y;
    int64_t i1 = tpig.x;

    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    device const float * src_row = (device const float *) ((device const char *) src0 + i1*nb01 + i2*nb02 + i3*nb03);
    device       float * dst_row = (device       float *) ((device       char *) dst  + i1*nb1  + i2*nb2  + i3*nb3);

    float row_sum = 0;

    for (int64_t i0 = 0; i0 < ne00; i0++) {
        row_sum += src_row[i0];
    }

    dst_row[0] = row_sum;
}

template<typename T>
kernel void kernel_soft_max(
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint32_t & n_head_log2,
        threadgroup  float * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float * psrc0 = (device const float *) src0 + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    device const     T * pmask = src1 != src0 ? (device const    T *) src1         + i01*ne00 : nullptr;
    device       float * pdst  = (device       float *) dst  + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        const int64_t h = i02;

        const float base = h < n_head_log2 ? m0 : m1;
        const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = -INFINITY;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        lmax = MAX(lmax, psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        const float exp_psrc0 = exp((psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        pdst[i00] *= inv_sum;
    }
}

template<typename T>
kernel void kernel_soft_max_4(
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint32_t & n_head_log2,
        threadgroup  float * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float4 * psrc4 = (device const float4 *) src0 + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00)/4;
    device const      T * pmask = src1 != src0 ? (device const     T *) src1         + i01*ne00/4 : nullptr;
    device       float4 * pdst4 = (device       float4 *) dst  + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00)/4;

    float slope = 1.0f;

    if (max_bias > 0.0f) {
        const int64_t h = i02;

        const float base = h < n_head_log2 ? m0 : m1;
        const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = -INFINITY;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        const float4 exp_psrc4 = exp((psrc4[i00]*scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        pdst4[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>)    kernel_soft_max_t;
typedef decltype(kernel_soft_max_4<float4>) kernel_soft_max_4_t;

template [[host_name("kernel_soft_max_f16")]]   kernel kernel_soft_max_t   kernel_soft_max<half>;
template [[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t   kernel_soft_max<float>;
template [[host_name("kernel_soft_max_f16_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<half4>;
template [[host_name("kernel_soft_max_f32_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<float4>;

kernel void kernel_diag_mask_inf(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant       int & n_past,
        uint3 tpig[[thread_position_in_grid]]) {
    const int64_t i02 = tpig[2];
    const int64_t i01 = tpig[1];
    const int64_t i00 = tpig[0];

    if (i00 > n_past + i01) {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = -INFINITY;
    } else {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = src0[i02*ne01*ne00 + i01*ne00 + i00];
    }
}

kernel void kernel_diag_mask_inf_8(
        device const float4 * src0,
        device       float4 * dst,
        constant    int64_t & ne00,
        constant    int64_t & ne01,
        constant        int & n_past,
        uint3 tpig[[thread_position_in_grid]]) {

    const int64_t i = 2*tpig[0];

    dst[i+0] = src0[i+0];
    dst[i+1] = src0[i+1];
    int64_t i4 = 4*i;
    const int64_t i02 = i4/(ne00*ne01); i4 -= i02*ne00*ne01;
    const int64_t i01 = i4/(ne00);      i4 -= i01*ne00;
    const int64_t i00 = i4;
    for (int k = 3; k >= 0; --k) {
        if (i00 + 4 + k <= n_past + i01) {
            break;
        }
        dst[i+1][k] = -INFINITY;
        if (i00 + k > n_past + i01) {
            dst[i][k] = -INFINITY;
        }
    }
}

// ref: ggml.c:ggml_compute_forward_ssm_conv_f32
// TODO: optimize
kernel void kernel_ssm_conv_f32(
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = tgpig.x;
    const int64_t i2 = tgpig.y;
    const int64_t i3 = tgpig.z;

    const int64_t nc  = ne10;
    const int64_t ncs = ne00;
    const int64_t nr  = ne01;
    const int64_t n_t = ne1;
    const int64_t n_s = ne2;

    device const float * s = (device const float *) ((device const char *) src0 + ir*nb01 + i2*nb00 + i3*nb02);
    device const float * c = (device const float *) ((device const char *) src1 + ir*nb11);
    device       float * x = (device       float *) ((device       char *) dst  + ir*nb0  + i2*nb1  + i3*nb2);

    float sumf = 0.0f;

    for (int64_t i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    x[0] = sumf;
}

// ref: ggml.c:ggml_compute_forward_ssm_scan_f32
// TODO: optimize
kernel void kernel_ssm_scan_f32(
        device const void * src0,
        device const void * src1,
        device const void * src2,
        device const void * src3,
        device const void * src4,
        device const void * src5,
        device      float * dst,
        constant  int64_t & d_state,
        constant  int64_t & d_inner,
        constant  int64_t & n_seq_tokens,
        constant  int64_t & n_seqs,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant uint64_t & nb20,
        constant uint64_t & nb21,
        constant uint64_t & nb22,
        constant uint64_t & nb30,
        constant uint64_t & nb31,
        constant uint64_t & nb40,
        constant uint64_t & nb41,
        constant uint64_t & nb42,
        constant uint64_t & nb50,
        constant uint64_t & nb51,
        constant uint64_t & nb52,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = tgpig.x;
    const int64_t i3 = tgpig.y;

    const int64_t nc  = d_state;
    const int64_t nr  = d_inner;
    const int64_t n_t = n_seq_tokens;
    const int64_t n_s = n_seqs;

    for (int64_t i2 = 0; i2 < n_t; ++i2) {
        device const float * s0 = (device const float *) ((device const char *) src0 + ir*nb01 + i3*nb02);
        device const float * x  = (device const float *) ((device const char *) src1 + ir*nb10 + i2*nb11 + i3*nb12);
        device const float * dt = (device const float *) ((device const char *) src2 + ir*nb20 + i2*nb21 + i3*nb22);
        device const float * A  = (device const float *) ((device const char *) src3 + ir*nb31);
        device const float * B  = (device const float *) ((device const char *) src4 + i2*nb41 + i3*nb42);
        device const float * C  = (device const float *) ((device const char *) src5 + i2*nb51 + i3*nb52);
        device       float * y  = (device       float *) ((device       char *) dst  + ir*nb10 + i2*nb11 + i3*nb12); // TODO: do not use src1 strides
        device       float * s  = (device       float *) ((device       char *) dst  + ir*nb01 + i3*nb02 +    nb13);

        if (i2 > 0) {
            s0 = s;
        }

        // i1 == 0
        float dt_soft_plus = dt[0] <= 20.0f ? log(1.0f + exp(dt[0])) : dt[0];
        float x_dt = x[0] * dt_soft_plus;
        float sumf = 0.0f;

        for (int64_t i0 = 0; i0 < nc; ++i0) {
            int64_t i = i0;
            float state = (s0[i] * exp(dt_soft_plus * A[i])) + (B[i0] * x_dt);
            sumf += state * C[i0];
            s[i] = state;
        }

        y[0] = sumf;
    }
}

kernel void kernel_norm(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        threadgroup float  * sum [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * x = (device const float *) ((device const char *) src0 + tgpig*nb01);
    // MEAN
    // parallel sum
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        sum[tpitg] += x[i00];
    }
    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg/2; i > 0; i /= 2) {
        if (tpitg < i) {
            sum[tpitg] += sum[tpitg + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean  = sum[0] / ne00;

    // recenter and VARIANCE
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float * y = dst + tgpig*ne00;
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = x[i00] - mean;
        sum[tpitg] += y[i00] * y[i00];
    }

    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg/2; i > 0; i /= 2) {
        if (tpitg < i) {
            sum[tpitg] += sum[tpitg + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float variance = sum[0] / ne00;

    const float scale = 1.0f/sqrt(variance + eps);
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = y[i00] * scale;
    }
}

kernel void kernel_rms_norm(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float4 * x = (device const float4 *) ((device const char *) src0 + tgpig*nb01);

    float4 sumf = 0;
    float all_sum = 0;

    // parallel sum
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        sumf += x[i00] * x[i00];
    }
    all_sum = sumf[0] + sumf[1] + sumf[2] + sumf[3];
    all_sum = simd_sum(all_sum);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = all_sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        all_sum = buf[tiisg];
        all_sum = simd_sum(all_sum);
    }

    const float mean  = all_sum/ne00;
    const float scale = 1.0f/sqrt(mean + eps);

    device float4 * y = (device float4 *) (dst + tgpig*ne00);
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        y[i00] = x[i00] * scale;
    }
}

kernel void kernel_group_norm(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int32_t & n_groups,
        constant     float & eps,
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    const int64_t ne = ne00*ne01*ne02;
    const int64_t gs = ne00*ne01*((ne02 + n_groups - 1) / n_groups);

    int start = tgpig * gs;
    int end   = start + gs;

    start += tpitg;

    if (end >= ne) {
        end = ne;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += ntg) {
        tmp += src0[j];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float mean = tmp / gs;
    tmp = 0.0f;

    for (int j = start; j < end; j += ntg) {
        float xi = src0[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float variance = tmp / gs;
    const float scale = 1.0f/sqrt(variance + eps);
    for (int j = start; j < end; j += ntg) {
        dst[j] *= scale;
    }
}

// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float2 acc = 0.f;

    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 1 + il/2);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.f + acc[0] + acc[1]);
}

// function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float2 acc = 0.f;

    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 2 + il/2);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (acc[0] + acc[1]) + sumy * m;
}

// function for calculate inner product between half a q5_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float2 acc = 0.f;

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 3 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010))
                + yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[1] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100))
                + yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }
    return d * (sumy * -16.f + acc[0] + acc[1]);
}

// function for calculate inner product between half a q5_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_1/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float2 acc = 0.f;

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 4 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010))
                + yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[1] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100))
                + yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }
    return d * (acc[0] + acc[1]) + sumy * m;
}

// putting them in the kernel cause a significant performance penalty
#define N_DST 4        // each SIMD group works on 4 rows
#define N_SIMDGROUP 2  // number of SIMD groups in a thread group
//Note: This is a template, but strictly speaking it only applies to
//      quantizations where the block size is 32. It also does not
//      guard against the number of rows not being divisible by
//      N_DST, so this is another explicit assumption of the implementation.
template<typename block_q_type, int nr, int nsg, int nw>
void mul_vec_q_n_f32_impl(
        device const void  * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3 tgpig, uint tiisg, uint sgitg) {
    const int nb = ne00/QK4_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * nsg + sgitg) * nr;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q_type * x = (device const block_q_type *) src0 + offset0;
    device const float        * y = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16]; // src1 vector cache
    float sumf[nr] = {0.f};

    const int ix = (tiisg/2);
    const int il = (tiisg%2)*8;

    device const float * yb = y + ix * QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    for (int ib = ix; ib < nb; ib += nw/2) {
        float sumy = 0;
        for (int i = 0; i < 8; i += 2) {
            sumy += yb[i] + yb[i+1];
            yl[i+0] = yb[i+ 0];
            yl[i+1] = yb[i+ 1]/256.f;

            sumy += yb[i+16] + yb[i+17];
            yl[i+8] = yb[i+16]/16.f;
            yl[i+9] = yb[i+17]/4096.f;
        }

        for (int row = 0; row < nr; row++) {
            sumf[row] += block_q_n_dot_y(x+ib+row*nb, sumy, yl, il);
        }

        yb += QK4_0 * 16;
    }

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < ne01) {
            dst[im*ne0*ne1 + r1*ne0 + first_row + row] = tot;
        }
    }
}

kernel void kernel_mul_mv_q4_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mv_q4_1_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {
     mul_vec_q_n_f32_impl<block_q4_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mv_q5_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mv_q5_1_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}


#define NB_Q8_0 8

void kernel_mul_mv_q8_0_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {
    const int nr  = N_DST;
    const int nsg = N_SIMDGROUP;
    const int nw  = N_SIMDWIDTH;

    const int nb = ne00/QK8_0;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * nsg + sgitg) * nr;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q8_0 * x = (device const block_q8_0 *) src0 + offset0;
    device const float      * y = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[NB_Q8_0];
    float sumf[nr]={0.f};

    const int ix = tiisg/4;
    const int il = tiisg%4;

    device const float * yb = y + ix * QK8_0 + NB_Q8_0*il;

    // each thread in a SIMD group deals with NB_Q8_0 quants at a time
    for (int ib = ix; ib < nb; ib += nw/4) {
        for (int i = 0; i < NB_Q8_0; ++i) {
            yl[i] = yb[i];
        }

        for (int row = 0; row < nr; row++) {
            device const int8_t * qs = x[ib+row*nb].qs + NB_Q8_0*il;
            float sumq = 0.f;
            for (int iq = 0; iq < NB_Q8_0; ++iq) {
                sumq += qs[iq] * yl[iq];
            }
            sumf[row] += sumq*x[ib+row*nb].d;
        }

        yb += NB_Q8_0 * nw;
    }

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}

#define N_MV_T_T 4

template<typename T0, typename T04, typename T1, typename T14>
void kernel_mul_mv_impl(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb00,
                  uint64_t   nb01,
                  uint64_t   nb02,
                   int64_t   ne10,
                   int64_t   ne11,
                   int64_t   ne12,
                  uint64_t   nb10,
                  uint64_t   nb11,
                  uint64_t   nb12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
                   uint3     tgpig,
                   uint      tiisg) {
    const int64_t r0 = tgpig.x;
    const int64_t rb = tgpig.y*N_MV_T_T;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb02*ne02;

    device const T0 * x = (device const T0 *) (src0 + offset0);

    if (ne00 < 128) {
        for (int row = 0; row < N_MV_T_T; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            device const T1 * y = (device const T1 *) (src1 + r1*nb11 + im*nb12);

            float sumf = 0;
            for (int i = tiisg; i < ne00; i += 32) {
                sumf += (T0) x[i] * (T1) y[i];
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    } else {
        device const T04 * x4 = (device const T04 *) x;
        for (int row = 0; row < N_MV_T_T; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            device const T1  * y  = (device const T1  *) (src1 + r1*nb11 + im*nb12);
            device const T14 * y4 = (device const T14 *) y;

            float sumf = 0;
            for (int i = tiisg; i < ne00/4; i += 32) {
                for (int k = 0; k < 4; ++k) sumf += (float) (x4[i][k] * y4[i][k]);
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                for (int i = 4*(ne00/4); i < ne00; ++i) all_sum += (float) (x[i] * y[i]);
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    }
}

template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_impl<T0, T04, T1, T14>(
        src0,
        src1,
        dst,
        ne00,
        ne01,
        ne02,
        nb00,
        nb01,
        nb02,
        ne10,
        ne11,
        ne12,
        nb10,
        nb11,
        nb12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg);
}

typedef decltype(kernel_mul_mv<half, half4, half, half4>) mul_mv_t;

template [[host_name("kernel_mul_mv_f32_f32")]]   kernel mul_mv_t kernel_mul_mv<float,  float4,  float,  float4>;
template [[host_name("kernel_mul_mv_f16_f32")]]   kernel mul_mv_t kernel_mul_mv<half,   half4,   float,  float4>;
template [[host_name("kernel_mul_mv_f16_f16")]]   kernel mul_mv_t kernel_mul_mv<half,   half4,   half,   half4>;

template<typename T, typename T4>
kernel void kernel_mul_mv_1row(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]]) {

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb02*ne02;

    device const T     * x = (device const T     *) (src0 + offset0);
    device const float * y = (device const float *) (src1 + r1*nb11 + im*nb12);

    float sumf = 0;
    if (ne00 < 128) {
        for (int i = tiisg; i < ne00; i += 32) {
            sumf += (float) x[i] * (float) y[i];
        }
        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    } else {
        device const T4     * x4 = (device const T4     *) x;
        device const float4 * y4 = (device const float4 *) y;

        for (int i = tiisg; i < ne00/4; i += 32) {
            for (int k = 0; k < 4; ++k) sumf += (float) (x4[i][k] * y4[i][k]);
        }

        float all_sum = simd_sum(sumf);

        if (tiisg == 0) {
            for (int i = 4*(ne00/4); i < ne00; ++i) all_sum += (float) (x[i] * y[i]);
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}

typedef decltype(kernel_mul_mv_1row<half, half4>) mul_mv_1row_t;

template [[host_name("kernel_mul_mv_f16_f32_1row")]]  kernel mul_mv_1row_t kernel_mul_mv_1row<half,   half4>;

// Assumes row size (ne00) is a multiple of 4
template<typename T, typename T4>
kernel void kernel_mul_mv_l4(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]]) {

    const int nrows = ne11;
    const int64_t r0 = tgpig.x;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb02*ne02;

    device const T4 * x4 = (device const T4 *) (src0 + offset0);

    for (int r1 = 0; r1 < nrows; ++r1) {
        device const float4 * y4 = (device const float4 *) (src1 + r1*nb11 + im*nb12);

        float sumf = 0;
        for (int i = tiisg; i < ne00/4; i += 32) {
            for (int k = 0; k < 4; ++k) sumf += (float) (x4[i][k] * y4[i][k]);
        }

        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}

typedef decltype(kernel_mul_mv_l4<half, half4>) mul_mv_l4_t;

template [[host_name("kernel_mul_mv_f16_f32_l4")]]  kernel mul_mv_l4_t kernel_mul_mv_l4<half, half4>;

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    dims[0] = max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

template<typename T>
kernel void kernel_rope_norm(
        device const    void * src0,
        device const int32_t * src1,
        device const   float * src2,
        device         float * dst,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant     int64_t & ne03,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant    uint64_t & nb03,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant     int64_t & ne2,
        constant     int64_t & ne3,
        constant    uint64_t & nb0,
        constant    uint64_t & nb1,
        constant    uint64_t & nb2,
        constant    uint64_t & nb3,
        constant         int & n_past,
        constant         int & n_dims,
        constant         int & n_ctx_orig,
        constant       float & freq_base,
        constant       float & freq_scale,
        constant       float & ext_factor,
        constant       float & attn_factor,
        constant       float & beta_fast,
        constant       float & beta_slow,
        uint  tiitg[[thread_index_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]],
        uint3 tgpig[[threadgroup_position_in_grid]]) {
    const int64_t i3 = tgpig[2];
    const int64_t i2 = tgpig[1];
    const int64_t i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    device const int32_t * pos = src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/n_dims;

    float cos_theta;
    float sin_theta;

    for (int64_t i0 = 2*tiitg; i0 < ne0; i0 += 2*tptg.x) {
        if (i0 < n_dims) {
            const int64_t ic = i0/2;

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

template<typename T>
kernel void kernel_rope_neox(
        device const    void * src0,
        device const int32_t * src1,
        device const   float * src2,
        device         float * dst,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant     int64_t & ne03,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant    uint64_t & nb03,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant     int64_t & ne2,
        constant     int64_t & ne3,
        constant    uint64_t & nb0,
        constant    uint64_t & nb1,
        constant    uint64_t & nb2,
        constant    uint64_t & nb3,
        constant         int & n_past,
        constant         int & n_dims,
        constant         int & n_ctx_orig,
        constant       float & freq_base,
        constant       float & freq_scale,
        constant       float & ext_factor,
        constant       float & attn_factor,
        constant       float & beta_fast,
        constant       float & beta_slow,
        uint  tiitg[[thread_index_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]],
        uint3 tgpig[[threadgroup_position_in_grid]]) {
    const int64_t i3 = tgpig[2];
    const int64_t i2 = tgpig[1];
    const int64_t i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    device const int32_t * pos = src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/n_dims;

    float cos_theta;
    float sin_theta;

    for (int64_t i0 = 2*tiitg; i0 < ne0; i0 += 2*tptg.x) {
        if (i0 < n_dims) {
            const int64_t ic = i0/2;

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
            device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

            const float x0 = src[0];
            const float x1 = src[n_dims/2];

            dst_data[0]        = x0*cos_theta - x1*sin_theta;
            dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

typedef decltype(kernel_rope_norm<float>) kernel_rope_norm_t;
typedef decltype(kernel_rope_neox<float>) kernel_rope_neox_t;

template [[host_name("kernel_rope_norm_f32")]] kernel kernel_rope_norm_t kernel_rope_norm<float>;
template [[host_name("kernel_rope_norm_f16")]] kernel kernel_rope_norm_t kernel_rope_norm<half>;

template [[host_name("kernel_rope_neox_f32")]] kernel kernel_rope_neox_t kernel_rope_neox<float>;
template [[host_name("kernel_rope_neox_f16")]] kernel kernel_rope_neox_t kernel_rope_neox<half>;

typedef void (im2col_t)(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t iiw = tgpig[2] * s0 + tpitg[2] * d0 - p0;
    const int32_t iih = tgpig[1] * s1 + tpitg[1] * d1 - p1;

    const int32_t offset_dst =
        (tpitg[0] * tgpg[1] * tgpg[2] + tgpig[1] * tgpg[2] + tgpig[2]) * CHW +
        (tgpig[0] * (ntg[1] * ntg[2]) + tpitg[1] * ntg[2] + tpitg[2]);

    device T * pdst = (device T *) (dst);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        pdst[offset_dst] = 0.0f;
    } else {
        const int32_t offset_src = tpitg[0] * ofs0 + tgpig[0] * ofs1;
        pdst[offset_dst] = x[offset_src + iih * IW + iiw];
    }
}

template [[host_name("kernel_im2col_f32")]] kernel im2col_t kernel_im2col<float>;
template [[host_name("kernel_im2col_f16")]] kernel im2col_t kernel_im2col<half>;

typedef void (im2col_ext_t)(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        constant   int32_t & N,
        constant   int32_t & KH,
        constant   int32_t & KW,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col_ext(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        constant   int32_t & N,
        constant   int32_t & KH,
        constant   int32_t & KW,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],      // tgpg[0] = D x IC x KH x KW, CHW = IC x KH x KW
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {  // [M, 1, 1]
    const int32_t KHW = KH * KW;             // KHW == ntg[1] * ntg[2], KW == ntg[2]

    const int32_t d = tgpig[0] / CHW;
    const int32_t chw = tgpig[0] % CHW;
    const int32_t tgpig_0 = chw / KHW;  // 0 ~ (IC - 1)
    const int32_t HW = tgpig[0] % KHW;

    const int32_t tpitg_0 = (d * ntg[0]) + tpitg[0];
    if (tpitg_0 >= N) {
        return;
    }

    const int32_t tpitg_1 = HW / KW;
    const int32_t tpitg_2 = HW % KW;

    const int32_t iiw = tgpig[2] * s0 + tpitg_2 * d0 - p0;
    const int32_t iih = tgpig[1] * s1 + tpitg_1 * d1 - p1;

    const int32_t offset_dst =
        (tpitg_0 * tgpg[1] * tgpg[2] + tgpig[1] * tgpg[2] + tgpig[2]) * CHW +
        (tgpig_0 * KHW + tpitg_1 * KW + tpitg_2);

    device T * pdst = (device T *) (dst);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        pdst[offset_dst] = 0.0f;
    } else {
        const int32_t offset_src = tpitg_0 * ofs0 + tgpig_0 * ofs1;
        pdst[offset_dst] = x[offset_src + iih * IW + iiw];
    }
}

template [[host_name("kernel_im2col_ext_f32")]] kernel im2col_ext_t kernel_im2col_ext<float>;
template [[host_name("kernel_im2col_ext_f16")]] kernel im2col_ext_t kernel_im2col_ext<half>;

kernel void kernel_upscale_f32(
    device  const char * src0,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne0,
    constant   int64_t & ne1,
    constant   int64_t & ne2,
    constant   int64_t & ne3,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    constant     float & sf0,
    constant     float & sf1,
    constant     float & sf2,
    constant     float & sf3,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3/sf3;
    const int64_t i02 = i2/sf2;
    const int64_t i01 = i1/sf1;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int64_t i00 = i0/sf0;

        device const float * src0_ptr = (device const float *) (src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        device       float * dst_ptr  = (device       float *) (dst  +  i3*nb3  +  i2*nb2  +  i1*nb1  +  i0*nb0);

        dst_ptr[0] = src0_ptr[0];
    }
}

kernel void kernel_pad_f32(
    device  const char * src0,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne0,
    constant   int64_t & ne1,
    constant   int64_t & ne2,
    constant   int64_t & ne3,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*nb3  +  i2*nb2  +  i1*nb1);

    if (i1 < ne01 && i2 < ne02 && i3 < ne03) {
        for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
            if (i0 < ne00) {
                dst_ptr[i0] = src0_ptr[i0];
            } else {
                dst_ptr[i0] = 0.0f;
            }
        }

        return;
    }

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        dst_ptr[i0] = 0.0f;
    }
}

kernel void kernel_arange_f32(
    device        char * dst,
    constant   int64_t & ne0,
    constant   float   & start,
    constant   float   & step,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        dst_ptr[i0] = start + step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    device  const char * src0,
    device        char * dst,
    constant  uint64_t & nb1,
    constant  int      & dim,
    constant  int      & max_period,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst +  i*nb1);

    int half_ = dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (dim % 2 != 0 && tpitg.x == 0) {
        embed_data[dim] = 0.f;
    }
}

// bitonic sort implementation following the CUDA kernels as reference
typedef void (argsort_t)(
        device const float  * x,
        device     int32_t  * dst,
        constant   int64_t  & ncols,
        constant   int64_t  & ncols_pad,
        threadgroup int32_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]);

template<ggml_sort_order order>
kernel void kernel_argsort_f32_i32(
        device const float   * x,
        device       int32_t * dst,
        constant     int64_t & ncols,
        constant     int64_t & ncols_pad,
        threadgroup int32_t  * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {
    // bitonic sort
    int col = tpitg[0];
    int row = tgpig[1];

    if (col >= ncols_pad) return;

    device const float   * x_row   = x + row * ncols;
    threadgroup int32_t  * dst_row = shared_values;

    // initialize indices
    dst_row[col] = col;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

template [[host_name("kernel_argsort_f32_i32_asc")]]  kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_ASC>;
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_DESC>;

kernel void kernel_leaky_relu_f32(
        device const float * src0,
        device       float * dst,
        constant     float & slope,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] > 0.0f ? src0[tpig] : src0[tpig] * slope;
}

typedef void (flash_attn_ext_f16_t)(
        device const  char * q,
        device const  char * k,
        device const  char * v,
        device const  char * mask,
        device       float * dst,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant   int64_t & ne13,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant  uint64_t & nb21,
        constant  uint64_t & nb22,
        constant  uint64_t & nb23,
        constant  uint64_t & nb31,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint32_t & n_head_log2,
        constant     float & logit_softcap,
        threadgroup   half * shared,
        uint3  tgpig[[threadgroup_position_in_grid]],
        uint3  tpitg[[thread_position_in_threadgroup]],
        uint3    ntg[[threads_per_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]);

// ref: https://arxiv.org/pdf/2307.08691.pdf
template<int64_t D, int64_t Q = 8, int64_t C = 32> // head size, queries per threadgroup, cache items per threadgroup
kernel void kernel_flash_attn_ext_f16(
        device const  char * q,
        device const  char * k,
        device const  char * v,
        device const  char * mask,
        device       float * dst,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant   int64_t & ne13,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant  uint64_t & nb21,
        constant  uint64_t & nb22,
        constant  uint64_t & nb23,
        constant  uint64_t & nb31,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint32_t & n_head_log2,
        constant     float & logit_softcap,
        threadgroup   half * shared [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        uint3  tpitg[[thread_position_in_threadgroup]],
        uint3    ntg[[threads_per_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short nsg = ntg.y; // number of simdgroups

    const short iq3 = tgpig[2];
    const short iq2 = tgpig[1];
    const short iq1 = tgpig[0]*Q;

    const short D4 = D/4;
    const short D8 = D/8;
  //const short Q8 = Q/8;
    const short NW = N_SIMDWIDTH;
    const short SH = (C + Q); // shared memory per simdgroup in (half)

    const short T  = D + 2*nsg*SH; // shared memory size per query in (half)
    const short TF = T/2;        // shared memory size per query in (float)
    const short T4 = T/4;        // shared memory size per query in (half4)

    threadgroup half  * sq  = (threadgroup half  *) (shared +              0*D); // holds the query data
    threadgroup half4 * sq4 = (threadgroup half4 *) (shared +              0*D); // same as above but in half4
    threadgroup float * ss  = (threadgroup float *) (shared + 2*sgitg*SH + 1*D); // scratch buffer for attention and diagonal matrix

    // store the result for all queries in local memory in 8x8 matrices (the O matrix from the paper)
    simdgroup_half8x8 lo[D8];

    // load heads from Q to shared memory
    for (short j = sgitg; j < Q; j += nsg) {
        device const float4 * q4 = (device const float4 *) ((device const char *) q + ((iq1 + j)*nb01 + iq2*nb02 + iq3*nb03));

        for (short i = tiisg; i < D4; i += NW) {
            if (iq1 + j < ne01) {
                sq4[j*T4 + i] = (half4) q4[i];
            } else {
                sq4[j*T4 + i] = 0.0h;
            }
        }
    }

    // zero out lo
    for (short i = 0; i < D8; ++i) {
        lo[i] = make_filled_simdgroup_matrix<half, 8>(0.0h);
    }

    // zero out shared memory SH
    for (short j = 0; j < Q; ++j) {
        for (short i = tiisg; i < SH; i += NW) {
            ss[j*TF + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S[Q] = { [0 ... Q-1] = 0.0h };
        float M[Q] = { [0 ... Q-1] = -FLT_MAX/2 };

        // assume K and V are same shape
        const short ne22 = ne12;
        const short ne23 = ne13;

        // broadcast
        const short rk2 = ne02/ne12;
        const short rk3 = ne03/ne13;

        const short rv2 = ne02/ne22;
        const short rv3 = ne03/ne23;

        // k indices
        const short ik2 = iq2/rk2;
        const short ik3 = iq3/rk3;

        // v indices
        const short iv2 = iq2/rv2;
        const short iv3 = iq3/rv3;

        // load the queries from shared memory into local memory
        simdgroup_half8x8 mq[D8];

        for (short i = 0; i < D8; ++i) {
            simdgroup_load(mq[i], sq + i*8, T);
        }

        // pointer to the mask
        device const half * mp = (device const half *) (mask + iq1*nb31);

        float slope = 1.0f;

        // ALiBi
        if (max_bias > 0.0f) {
            const uint32_t h = iq2;

            const float base = h < n_head_log2 ? m0 : m1;
            const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = 0; ic0 < ne11; ic0 += C*nsg) {
            const int ic = ic0 + C*sgitg;
            if (ic >= ne11) {
                break;
            }

            // Q*K^T
            {
                for (short cc = 0; cc < C/8; ++cc) {
                    simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.h);

                    device const half * pk = (device const half *) ((device const char *) k + ((ic + 8*cc)*nb11 + ik2*nb12 + ik3*nb13));

                    for (short i = 0; i < D8; ++i) {
                        simdgroup_half8x8 mk;
                        simdgroup_load(mk, pk + i*8, nb11/sizeof(half), 0, true); // transpose

                        simdgroup_multiply_accumulate(mqk, mq[i], mk, mqk);
                    }

                    simdgroup_store(mqk, ss + 8*cc, TF, 0, false);
                }
            }

            // used to detect blocks full of -INF
            float smax = -INFINITY;

            // online softmax
            {
                float ms[Q];

                for (short j = 0; j < Q; ++j) {
                    const float m = M[j];

                    // scale and apply the logitcap / mask
                    float s = ss[j*TF + tiisg]*scale;

                    if (logit_softcap != 0.0f) {
                        s = logit_softcap*precise::tanh(s);
                    }

                    if (mask != q) {
                        // mqk = mqk + mask*slope
                        s += slope*mp[ic + j*nb31/sizeof(half) + tiisg];
                    }

                    smax = simd_max(max(smax, s));
                    M[j] = simd_max(max(M[j], s));

                                ms[j] = exp(m - M[j]);
                    const float vs    = exp(s - M[j]);

                    S[j] = S[j]*ms[j] + simd_sum(vs);

                    // the P matrix from the paper (Q rows, C columns)
                    ss[j*TF + tiisg] = vs;
                }

                // create a QxQ diagonal matrix for rescaling the output
                if (tiisg < Q) {
                    ss[tiisg*TF + C + tiisg] = ms[tiisg];
                }
            }

            // skip -INF blocks
            if (smax == -INFINITY) {
                continue;
            }

            // O = diag(ms)*O
            {
                simdgroup_float8x8 mm;
                simdgroup_load(mm, ss + C, TF, 0, false);

                for (short i = 0; i < D8; ++i) {
                    simdgroup_multiply(lo[i], mm, lo[i]);
                }
            }

            // O = O + (Q*K^T)*V
            {
                for (short cc = 0; cc < C/8; ++cc) {
                    device const half * pv = (device const half *) ((device const char *) v + ((ic + 8*cc)*nb21 + iv2*nb22 + iv3*nb23));

                    for (short i = 0; i < D8; ++i) {
                        simdgroup_half8x8 mk;
                        simdgroup_load(mk, pv + i*8, nb21/sizeof(half), 0, false);

                        simdgroup_float8x8 mv;
                        simdgroup_load(mv, ss + 8*cc, TF, 0, false);

                        simdgroup_multiply_accumulate(lo[i], mv, mk, lo[i]);
                    }
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        for (short j = 0; j < Q; ++j) {
            if (tiisg == 0) {
                ss[j*TF + 0] = S[j];
                ss[j*TF + 1] = M[j];
            }
        }
    }

    // reduce the warps sequentially
    for (short sg = 1; sg < nsg; ++sg) {
        float S = { 0.0h };
        float M = { -FLT_MAX/2 };

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // each simdgroup stores its output to shared memory, reusing sq
        if (sgitg == sg) {
            for (short i = 0; i < D8; ++i) {
                simdgroup_store(lo[i], sq + i*8, T, 0, false);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // the first simdgroup accumulates the results from the other simdgroups
        if (sgitg == 0) {
            for (short j = 0; j < Q; ++j) {
                const float S0 = ss[j*TF +         0];
                const float S1 = ss[j*TF + sg*SH + 0];

                const float M0 = ss[j*TF +         1];
                const float M1 = ss[j*TF + sg*SH + 1];

                M = max(M0, M1);

                const float ms0 = exp(M0 - M);
                const float ms1 = exp(M1 - M);

                S = S0*ms0 + S1*ms1;

                if (tiisg == 0) {
                    ss[j*TF + 0] = S;
                    ss[j*TF + 1] = M;

                    ss[j*TF + C + j        ] = ms0;
                    ss[j*TF + C + j + sg*SH] = ms1;
                }
            }

            // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
            {
                simdgroup_half8x8 t;
                simdgroup_float8x8 ms0;
                simdgroup_float8x8 ms1;

                simdgroup_load(ms0, ss + C,         TF, 0, false);
                simdgroup_load(ms1, ss + C + sg*SH, TF, 0, false);

                for (short i = 0; i < D8; ++i) {
                    simdgroup_load    (t, sq + i*8, T, 0, false);
                    simdgroup_multiply(t, ms1, t);

                    simdgroup_multiply_accumulate(lo[i], ms0, lo[i], t);
                }
            }
        }
    }

    // store result to shared memory (reuse sq)
    if (sgitg == 0) {
        for (short i = 0; i < D8; ++i) {
            simdgroup_store(lo[i], sq + i*8, T, 0, false);
        }
    }

    device float4 * dst4 = (device float4 *) dst;

    // final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        for (short j = 0; j < Q && iq1 + j < ne01; ++j) {
            const float S = ss[j*TF + 0];

            for (short i = tiisg; i < D4; i += NW) {
                dst4[(iq3*ne2*ne1 + iq2 + (iq1 + j)*ne1)*D4 + i] = (float4) sq4[j*T4 + i]/S;
            }
        }
    }
}

template [[host_name("kernel_flash_attn_ext_f16_h64" )]] kernel flash_attn_ext_f16_t kernel_flash_attn_ext_f16<64>;
template [[host_name("kernel_flash_attn_ext_f16_h80" )]] kernel flash_attn_ext_f16_t kernel_flash_attn_ext_f16<80>;
template [[host_name("kernel_flash_attn_ext_f16_h96" )]] kernel flash_attn_ext_f16_t kernel_flash_attn_ext_f16<96>;
template [[host_name("kernel_flash_attn_ext_f16_h112")]] kernel flash_attn_ext_f16_t kernel_flash_attn_ext_f16<112>;
template [[host_name("kernel_flash_attn_ext_f16_h128")]] kernel flash_attn_ext_f16_t kernel_flash_attn_ext_f16<128>;
//template [[host_name("kernel_flash_attn_ext_f16_h256")]] kernel flash_attn_ext_f16_t kernel_flash_attn_ext_f16<256>;

template<int64_t D, int64_t Q = 1, int64_t C = 32> // head size, queries per threadgroup, cache items per threadgroup
kernel void kernel_flash_attn_ext_vec_f16(
        device const  char * q,
        device const  char * k,
        device const  char * v,
        device const  char * mask,
        device       float * dst,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant   int64_t & ne13,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant  uint64_t & nb13,
        constant  uint64_t & nb21,
        constant  uint64_t & nb22,
        constant  uint64_t & nb23,
        constant  uint64_t & nb31,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint32_t & n_head_log2,
        constant     float & logit_softcap,
        threadgroup   half * shared [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        uint3  tpitg[[thread_position_in_threadgroup]],
        uint3    ntg[[threads_per_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short nsg = ntg.y; // number of simdgroups

    const short iq3 = tgpig[2];
    const short iq2 = tgpig[1];
    const short iq1 = tgpig[0];

    const short D4 = D/4;
    const short NW = N_SIMDWIDTH;
    const short SH = (C + Q); // shared memory per simdgroup in (half)

    const short T  = D + 2*nsg*SH; // shared memory size per query in (half)

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        const uint32_t h = iq2;

        const float base = h < n_head_log2 ? m0 : m1;
        const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

  //threadgroup half   * sq  = (threadgroup half   *) (shared +              0*D); // holds the query data
    threadgroup half4  * sq4 = (threadgroup half4  *) (shared +              0*D); // same as above but in half4
    threadgroup float  * ss  = (threadgroup float  *) (shared + 2*sgitg*SH + 1*D); // scratch buffer for attention and diagonal matrix
    threadgroup float4 * ss4 = (threadgroup float4 *) (shared + 2*sgitg*SH + 1*D); // same as above but in half4
    threadgroup half4  * sr4 = (threadgroup half4  *) (shared +   sgitg*D  + 1*T); // scratch buffer for the results

    // store the result for all queries in local memory in 8x8 matrices (the O matrix from the paper)
    half4 lo[D4/NW];

    // load heads from Q to shared memory
    device const float4 * q4 = (device const float4 *) ((device const char *) q + (iq1*nb01 + iq2*nb02 + iq3*nb03));

    for (short i = tiisg; i < D4; i += NW) {
        if (iq1 < ne01) {
            sq4[i] = (half4) q4[i];
        } else {
            sq4[i] = 0.0h;
        }
    }

    // zero out lo
    for (short i = tiisg; i < D4; i += NW) {
        lo[i/NW] = 0.0h;
    }

    // zero out shared memory SH
    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = 0.0h;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S = { 0.0h };
        float M = { -FLT_MAX/2 };

        // assume K and V are same shape
        const short ne22 = ne12;
        const short ne23 = ne13;

        // broadcast
        const short rk2 = ne02/ne12;
        const short rk3 = ne03/ne13;

        const short rv2 = ne02/ne22;
        const short rv3 = ne03/ne23;

        // k indices
        const short ik2 = iq2 / rk2;
        const short ik3 = iq3 / rk3;

        // v indices
        const short iv2 = iq2 / rv2;
        const short iv3 = iq3 / rv3;

        // load the queries from shared memory into local memory
        float4 mq[D4];

        for (short ii = 0; ii < D4; ii += NW) {
            short i = ii + tiisg;
            mq[i] = (float4) sq4[i];
        }

        // pointer to the mask
        device const half4 * mp4 = (device const half4 *) (mask + iq1*nb31);

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = 0; ic0 < ne11; ic0 += C*nsg) {
            const int ic = ic0 + C*sgitg;
            if (ic >= ne11) {
                break;
            }

            // Q*K^T
            {
#pragma unroll
                for (short cc = 0; cc < C/4; ++cc) {
                    float4 mqk = { 0.0h };

                    device const half4 * pk4 = (device const half4 *) ((device const char *) k + ((ic + 4*cc)*nb11 + ik2*nb12 + ik3*nb13));

#pragma unroll
                    for (short ii = 0; ii < D4; ii += NW) {
                        const short i = ii + tiisg;

                        float4x4 mk;
                        mk[0] = (float4) pk4[i + 0*(nb11/8)];
                        mk[1] = (float4) pk4[i + 1*(nb11/8)];
                        mk[2] = (float4) pk4[i + 2*(nb11/8)];
                        mk[3] = (float4) pk4[i + 3*(nb11/8)];

                        mqk += (float4) (mq[i] * mk);
                    }

                    // reduce the results from the threads in the simdgroup
                    mqk += simd_shuffle_down(mqk, 16);
                    mqk += simd_shuffle_down(mqk,  8);
                    mqk += simd_shuffle_down(mqk,  4);
                    mqk += simd_shuffle_down(mqk,  2);
                    mqk += simd_shuffle_down(mqk,  1);

                    // mqk = mqk*scale + mask*slope
                    if (tiisg == 0) {
                        mqk *= scale;

                        if (logit_softcap != 0.0f) {
                            mqk = logit_softcap*precise::tanh(mqk);
                        }

                        mqk += (mask != q) ? ((float4) mp4[ic/4 + cc])*slope : (float4) 0.0f;

                        ss4[cc] = mqk;
                    }
                }
            }

            // online softmax
            {
                const short p = tiisg;

                const float m = M;
                const float s = ss[p];

                M = simd_max(max(M, s));

                const float ms = exp(m - M);
                const float vs = exp(s - M);

                S = S*ms + simd_sum(vs);

                // the P matrix from the paper (Q rows, C columns)
                ss[p] = vs;

                // O = diag(ms)*O
#pragma unroll
                for (short ii = 0; ii < D4; ii += NW) {
                    const short i = ii + tiisg;
                    lo[i/NW] *= ms;
                }
            }

            // O = O + (Q*K^T)*V
            {
#pragma unroll
                for (short cc = 0; cc < C/4; ++cc) {
                    device const half4 * pv4 = (device const half4 *) ((device const char *) v + ((ic + 4*cc)*nb21 + iv2*nb22 + iv3*nb23));

#pragma unroll
                    for (short ii = 0; ii < D4; ii += NW) {
                        const short i = ii + tiisg;

                        lo[i/NW] += pv4[i + 0*(nb21/8)] * ss[4*cc + 0];
                        lo[i/NW] += pv4[i + 1*(nb21/8)] * ss[4*cc + 1];
                        lo[i/NW] += pv4[i + 2*(nb21/8)] * ss[4*cc + 2];
                        lo[i/NW] += pv4[i + 3*(nb21/8)] * ss[4*cc + 3];
                    }
                }
            }

        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        if (tiisg == 0) {
            ss[0] = S;
            ss[1] = M;
        }
    }

    // store results to shared memory
    for (short ii = 0; ii < D4; ii += NW) {
        short i = ii + tiisg;
        sr4[i] = lo[ii/NW];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduce
    for (short r = nsg/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const float S0 = ss[       0];
            const float S1 = ss[r*SH + 0];

            const float M0 = ss[       1];
            const float M1 = ss[r*SH + 1];

            const float M = max(M0, M1);

            const float ms0 = exp(M0 - M);
            const float ms1 = exp(M1 - M);

            const float S = S0*ms0 + S1*ms1;

            if (tiisg == 0) {
                ss[0] = S;
                ss[1] = M;
            }

            // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
            for (short ii = 0; ii < D4; ii += NW) {
                short i = ii + tiisg;
                sr4[i] = sr4[i]*ms0 + sr4[i + r*D4]*ms1;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float4 * dst4 = (device float4 *) dst;

    // final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        const float S = ss[0];

        for (short ii = 0; ii < D4; ii += NW) {
            short i = ii + tiisg;
            dst4[(iq3*ne2*ne1 + iq2 + (iq1)*ne1)*D4 + i] = (float4) sr4[i]/S;
        }
    }
}

template [[host_name("kernel_flash_attn_ext_vec_f16_h128")]] kernel flash_attn_ext_f16_t kernel_flash_attn_ext_vec_f16<128>;
//template [[host_name("kernel_flash_attn_ext_vec_f16_h256")]] kernel flash_attn_ext_f16_t kernel_flash_attn_ext_vec_f16<256>;

template<typename T0, typename T1>
kernel void kernel_cpy(
        device  const void * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device T1 * dst_data = (device T1 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const T0 * src = (device T0 *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = (T1) src[0];
    }
}

typedef decltype(kernel_cpy<float, float>) kernel_cpy_t;

template [[host_name("kernel_cpy_f32_f32")]]  kernel kernel_cpy_t kernel_cpy<float,  float>;
template [[host_name("kernel_cpy_f32_f16")]]  kernel kernel_cpy_t kernel_cpy<float,  half>;
template [[host_name("kernel_cpy_f16_f16")]]  kernel kernel_cpy_t kernel_cpy<half,   half>;
template [[host_name("kernel_cpy_f16_f32")]]  kernel kernel_cpy_t kernel_cpy<half,   float>;

kernel void kernel_cpy_f32_q8_0(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK8_0;

    device block_q8_0 * dst_data = (device block_q8_0 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK8_0; i00 < ne00; i00 += ntg.x*QK8_0) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = src[j];
            amax = MAX(amax, fabs(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK8_0].d = d;

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = src[j]*id;

            dst_data[i00/QK8_0].qs[j] = round(x0);
        }
    }
}

kernel void kernel_cpy_f32_q4_0(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK4_0;

    device block_q4_0 * dst_data = (device block_q4_0 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK4_0; i00 < ne00; i00 += ntg.x*QK4_0) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < QK4_0; j++) {
            const float v = src[j];
            if (amax < fabs(v)) {
                amax = fabs(v);
                max  = v;
            }
        }

        const float d = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK4_0].d = d;

        for (int j = 0; j < QK4_0/2; ++j) {
            const float x0 = src[0       + j]*id;
            const float x1 = src[QK4_0/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            dst_data[i00/QK4_0].qs[j]  = xi0;
            dst_data[i00/QK4_0].qs[j] |= xi1 << 4;
        }
    }
}

kernel void kernel_cpy_f32_q4_1(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK4_1;

    device block_q4_1 * dst_data = (device block_q4_1 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK4_1; i00 < ne00; i00 += ntg.x*QK4_1) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < QK4_1; j++) {
            const float v = src[j];
            if (min > v) min = v;
            if (max < v) max = v;
        }

        const float d = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK4_1].d = d;
        dst_data[i00/QK4_1].m = min;

        for (int j = 0; j < QK4_1/2; ++j) {
            const float x0 = (src[0       + j] - min)*id;
            const float x1 = (src[QK4_1/2 + j] - min)*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

            dst_data[i00/QK4_1].qs[j]  = xi0;
            dst_data[i00/QK4_1].qs[j] |= xi1 << 4;
        }
    }
}

kernel void kernel_cpy_f32_q5_0(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK5_0;

    device block_q5_0 * dst_data = (device block_q5_0 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK5_0; i00 < ne00; i00 += ntg.x*QK5_0) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < QK5_0; j++) {
            const float v = src[j];
            if (amax < fabs(v)) {
                amax = fabs(v);
                max  = v;
            }
        }

        const float d = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK5_0].d = d;

        uint32_t qh = 0;
        for (int j = 0; j < QK5_0/2; ++j) {
            const float x0 = src[0       + j]*id;
            const float x1 = src[QK5_0/2 + j]*id;

            const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

            dst_data[i00/QK5_0].qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
        }
        thread const uint8_t * qh8 = (thread const uint8_t *)&qh;
        for (int j = 0; j < 4; ++j) {
            dst_data[i00/QK5_0].qh[j] = qh8[j];
        }
    }
}

kernel void kernel_cpy_f32_q5_1(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK5_1;

    device block_q5_1 * dst_data = (device block_q5_1 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK5_1; i00 < ne00; i00 += ntg.x*QK5_1) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float max = src[0];
        float min = src[0];

        for (int j = 1; j < QK5_1; j++) {
            const float v = src[j];
            min = v < min ? v : min;
            max = v > max ? v : max;
        }

        const float d = (max - min) / 31;
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK5_1].d = d;
        dst_data[i00/QK5_1].m = min;

        uint32_t qh = 0;
        for (int j = 0; j < QK5_1/2; ++j) {
            const float x0 = (src[0       + j] - min)*id;
            const float x1 = (src[QK5_1/2 + j] - min)*id;

            const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
            const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

            dst_data[i00/QK5_1].qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
        }
        thread const uint8_t * qh8 = (thread const uint8_t *)&qh;
        for (int j = 0; j < 4; ++j) {
            dst_data[i00/QK5_1].qh[j] = qh8[j];
        }
    }
}

static inline int best_index_int8(int n, constant float * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

kernel void kernel_cpy_f32_iq4_nl(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK4_NL;

    device block_iq4_nl * dst_data = (device block_iq4_nl *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK4_NL; i00 < ne00; i00 += ntg.x*QK4_NL) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < QK4_0; j++) {
            const float v = src[j];
            if (amax < fabs(v)) {
                amax = fabs(v);
                max  = v;
            }
        }

        const float d = max / kvalues_iq4nl_f[0];
        const float id = d ? 1.0f/d : 0.0f;

        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            const float x0 = src[0        + j]*id;
            const float x1 = src[QK4_NL/2 + j]*id;

            const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
            const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

            dst_data[i00/QK4_NL].qs[j] = xi0 | (xi1 << 4);

            const float v0 = kvalues_iq4nl_f[xi0];
            const float v1 = kvalues_iq4nl_f[xi1];
            const float w0 = src[0        + j]*src[0        + j];
            const float w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
            sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
            sumq2 += w0*v0*v0 + w1*v1*v1;

        }

        dst_data[i00/QK4_NL].d = sumq2 > 0 ? sumqx/sumq2 : d;

    }
}

kernel void kernel_concat(
    device  const char * src0,
    device  const char * src1,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne10,
    constant   int64_t & ne11,
    constant   int64_t & ne12,
    constant   int64_t & ne13,
    constant  uint64_t & nb10,
    constant  uint64_t & nb11,
    constant  uint64_t & nb12,
    constant  uint64_t & nb13,
    constant   int64_t & ne0,
    constant   int64_t & ne1,
    constant   int64_t & ne2,
    constant   int64_t & ne3,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    constant   int32_t & dim,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

    device const float * x;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (device const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            x = (device const float *)(src1 + (i3 - o[3])*nb13 + (i2 - o[2])*nb12 + (i1 - o[1])*nb11 + (i0 - o[0])*nb10);
        }

        device float * y = (device float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

void kernel_mul_mv_q2_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q2_K * x = (device const block_q2_K *) src0 + ib_row + offset0;
    device const float      * y = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int step = sizeof(block_q2_K) * nb;

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3
    const int is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += step/2;
            sc += step;
            dh += step/2;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_q2_K_f32")]]
kernel void kernel_mul_mv_q2_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_q3_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q3_K * x = (device const block_q3_K *) src0 + first_row*nb + offset0;
    device const float     * yy = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];

    //const uint16_t kmask1 = 0x3030;
    //const uint16_t kmask2 = 0x0f0f;

    const int tid = tiisg/4;
    const int ix  = tiisg%4;
    const int ip  = tid/4;          // 0 or 1
    const int il  = 2*((tid%4)/2);  // 0 or 2
    const int ir  = tid%2;
    const int n   = 8;
    const int l0  = n*ir;

    // One would think that the Metal compiler would figure out that ip and il can only have
    // 4 possible states, and optimize accordingly. Well, no. It needs help, and we do it
    // with these two tales.
    //
    // Possible masks for the high bit
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},  // ip = 0, il = 0
                           {0x0004, 0x0400, 0x0008, 0x0800},  // ip = 0, il = 2
                           {0x0010, 0x1000, 0x0020, 0x2000},  // ip = 1, il = 0
                           {0x0040, 0x4000, 0x0080, 0x8000}}; // ip = 1, il = 2

    // Possible masks for the low 2 bits
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};

    const ushort4 hm = mm[2*ip + il/2];

    const int shift = 2*il;
    const float    v1 = il == 0 ? 4.f : 64.f;
    const float    v2 = 4.f * v1;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;

    const int q_offset = 32*ip + l0;
    const int y_offset = 128*ip + 32*il + l0;

    const int step = sizeof(block_q3_K) * nb / 2;

    device const float * y1 = yy + ix*QK_K + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[2] = {0.f};
    float sumf2[2] = {0.f};
    for (int i = ix; i < nb; i += 4) {

        for (int l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;

        for (int row = 0; row < 2; ++row) {

            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (int l = 0; l < n; l += 2) {
                const int32_t qs = q[l/2];
                s1 += yl[l+0] * (qs & qm[il/2][0]);
                s2 += yl[l+1] * (qs & qm[il/2][1]);
                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm[il/2][2]);
                s5 += yl[l+17] * (qs & qm[il/2][3]);
                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (int l = 0; l < n; l += 2) {
                const int32_t qs = q[l/2+8];
                s1 += yl[l+8] * (qs & qm[il/2][0]);
                s2 += yl[l+9] * (qs & qm[il/2][1]);
                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qs & qm[il/2][2]);
                s5 += yl[l+25] * (qs & qm[il/2][3]);
                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);

            q  += step;
            h  += step;
            a  += step;
            dh += step;

        }

        y1 += 4 * QK_K;

    }

    for (int row = 0; row < 2; ++row) {
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sumf);
    }
    if (tiisg == 0) {
        for (int row = 0; row < 2; ++row) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = sumf1[row];
        }
    }
}

[[host_name("kernel_mul_mv_q3_K_f32")]]
kernel void kernel_mul_mv_q3_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_q4_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    //const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int first_row = r0 * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q4_K * x = (device const block_q4_K *) src0 + ib_row + offset0;
    device const float      * y = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16];
    float yh[16];
    float sumf[N_DST]={0.f}, all_sum;

    const int step = sizeof(block_q4_K) * nb / 2;

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+0] * (q1[i/2] & 0x000F);
                acc1[1] += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1[2] += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1[3] += yl[i+9] * (q1[i/2] & 0xF000);
                acc2[0] += yh[i+0] * (q2[i/2] & 0x000F);
                acc2[1] += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2[2] += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2[3] += yh[i+9] * (q2[i/2] & 0xF000);
            }

            float dall = dh[0];
            float dmin = dh[1];
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                 (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                 (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += step;
            sc += step;
            dh += step;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_q5_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q5_K * x = (device const block_q5_K *) src0 + first_row*nb + offset0;
    device const float     * yy = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float sumf[2]={0.f};

    const int step = sizeof(block_q5_K) * nb;

    float yl[16], yh[16];

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = tiisg/4;
    const int ix  = tiisg%4;
    const int iq  = tid/4;
    const int ir  = tid%4;
    const int n   = 8;

    const int l0 = n*ir;
    const int q_offset = 32*iq + l0;
    const int y_offset = 64*iq + l0;

    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {

        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        for (int row = 0; row < 2; ++row) {

            device const uint8_t * q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f};
            float4 acc2 = {0.f};
            for (int l = 0; l < n; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }
            const float dall = dh[0];
            const float dmin = dh[1];
            sumf[row] += dall * (sc8[0] * (acc1[0] +  16.f*acc2[0]) +
                                 sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                 sc8[4] * (acc1[2] +  16.f*acc2[2]) +
                                 sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += step;
            qh += step;
            dh += step/2;
            a  += step/2;

        }

        y1 += 4 * QK_K;

    }

    for (int row = 0; row < 2; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q5_K_f32")]]
kernel void kernel_mul_mv_q5_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_q6_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const uint8_t kmask1 = 0x03;
    const uint8_t kmask2 = 0x0C;
    const uint8_t kmask3 = 0x30;
    const uint8_t kmask4 = 0xC0;

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int     im = tgpig.z;

    const int row = 2 * r0 + sgitg;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q6_K * x = (device const block_q6_K *) src0 + row * nb + offset0;
    device const float     * yy = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float sumf = 0;

    const int tid  = tiisg/2;
    const int ix   = tiisg%2;
    const int ip   = tid/8;         // 0 or 1
    const int il   = tid%8;
    const int n    = 4;
    const int l0   = n*il;
    const int is   = 8*ip + l0/16;

    const int y_offset = 128*ip + l0;
    const int q_offset_l = 64*ip + l0;
    const int q_offset_h = 32*ip + l0;

    for (int i = ix; i < nb; i += 2) {

        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;

        device const float * y = yy + i * QK_K + y_offset;

        const float dall = x[i].d;

        float4 sums = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < n; ++l) {
            sums[0] += y[l+ 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
            sums[1] += y[l+32] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
            sums[2] += y[l+64] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
            sums[3] += y[l+96] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
        }

        sumf += dall * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

    }

    const float tot = simd_sum(sumf);
    if (tiisg == 0) {
        dst[r1*ne0 + im*ne0*ne1 + row] = tot;
    }
}

[[host_name("kernel_mul_mv_q6_K_f32")]]
kernel void kernel_mul_mv_q6_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

// ======================= "True" 2-bit

void kernel_mul_mv_iq2_xxs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) src0 + ib_row + offset0;
    device const float         * y = (device const float         *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * values = (threadgroup uint64_t *)shared_values;
    threadgroup uint8_t  * shared_signs = (threadgroup uint8_t *)(values + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) shared_signs[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (int l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(values + aux8[l]);
                const uint8_t signs = shared_signs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += nb*sizeof(block_iq2_xxs)/2;
            q2 += nb*sizeof(block_iq2_xxs)/2;
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xxs_f32")]]
kernel void kernel_mul_mv_iq2_xxs_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xxs_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq2_xs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq2_xs * x = (device const block_iq2_xs *) src0 + ib_row + offset0;
    device const float        * y = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * values = (threadgroup uint64_t *)shared_values;
    threadgroup uint8_t  * shared_signs = (threadgroup uint8_t *)(values + 512);
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq2xs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) shared_signs[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const uint8_t  * sc = xr->scales + ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const uint8_t ls1 = sc[0] & 0xf;
            const uint8_t ls2 = sc[0] >>  4;
            const float d1 = db * (0.5f + ls1);
            const float d2 = db * (0.5f + ls2);

            float sum1 = 0, sum2 = 0;
            for (int l = 0; l < 2; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(values + (q2[l] & 511));
                const uint8_t signs = shared_signs[(q2[l] >> 9)];
                for (int j = 0; j < 8; ++j) {
                    sum1 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (int l = 2; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(values + (q2[l] & 511));
                const uint8_t signs = shared_signs[(q2[l] >> 9)];
                for (int j = 0; j < 8; ++j) {
                    sum2 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;

            dh += nb*sizeof(block_iq2_xs)/2;
            q2 += nb*sizeof(block_iq2_xs)/2;
            sc += nb*sizeof(block_iq2_xs);
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xs_f32")]]
kernel void kernel_mul_mv_iq2_xs_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq3_xxs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq3_xxs * x = (device const block_iq3_xxs *) src0 + ib_row + offset0;
    device const float         * y = (device const float         *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * values = (threadgroup uint32_t *)shared_values;
    threadgroup uint8_t  * shared_signs = (threadgroup uint8_t *)(values + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq3xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) shared_signs[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t  * q3 = xr->qs + 8 * ib;
        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float2 sum = {0};
            for (int l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(values + q3[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(values + q3[2*l+1]);
                const uint8_t signs = shared_signs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += nb*sizeof(block_iq3_xxs)/2;
            q3  += nb*sizeof(block_iq3_xxs);
            gas += nb*sizeof(block_iq3_xxs)/2;
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.5f;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_xxs_f32")]]
kernel void kernel_mul_mv_iq3_xxs_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq3_s_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq3_s * x = (device const block_iq3_s *) src0 + ib_row + offset0;
    device const float       * y = (device const float       *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * values = (threadgroup uint32_t *)shared_values;
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq3s_grid[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 8 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + (ib/2);
        device const uint8_t * signs = xr->signs + 4 * ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));

            float2 sum = {0};
            for (int l = 0; l < 4; ++l) {
                const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? values + 256 : values;
                const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? values + 256 : values;
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
                for (int j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += nb*sizeof(block_iq3_s)/2;
            qs  += nb*sizeof(block_iq3_s);
            qh  += nb*sizeof(block_iq3_s);
            sc  += nb*sizeof(block_iq3_s);
            signs += nb*sizeof(block_iq3_s);
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_s_f32")]]
kernel void kernel_mul_mv_iq3_s_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_s_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq2_s_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq2_s * x = (device const block_iq2_s *) src0 + ib_row + offset0;
    device const float       * y = (device const float       *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    //threadgroup uint64_t * values = (threadgroup uint64_t *)shared_values;
    //{
    //    int nval = 32;
    //    int pos  = (32*sgitg + tiisg)*nval;
    //    for (int i = 0; i < nval; ++i) values[pos + i] = iq2s_grid[pos + i];
    //    threadgroup_barrier(mem_flags::mem_threadgroup);
    //}

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 4 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + ib;
        device const uint8_t * signs = qs + QK_K/8;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const float d1 = db * (0.5f + (sc[0] & 0xf));
            const float d2 = db * (0.5f + (sc[0] >>  4));

            float2 sum = {0};
            for (int l = 0; l < 2; ++l) {
                //const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(values + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                //const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(values + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sum[0] += yl[8*l + j +  0] * grid1[j] * select(1, -1, signs[l+0] & kmask_iq2xs[j]);
                    sum[1] += yl[8*l + j + 16] * grid2[j] * select(1, -1, signs[l+2] & kmask_iq2xs[j]);
                }
            }
            sumf[row] += d1 * sum[0] + d2 * sum[1];

            dh  += nb*sizeof(block_iq2_s)/2;
            qs  += nb*sizeof(block_iq2_s);
            qh  += nb*sizeof(block_iq2_s);
            sc  += nb*sizeof(block_iq2_s);
            signs += nb*sizeof(block_iq2_s);
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_s_f32")]]
kernel void kernel_mul_mv_iq2_s_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_s_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq1_s_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_value,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    device const block_iq1_s * x = (device const block_iq1_s *) src0 + ib_row + offset0;
    device const float       * y = (device const float       *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        float sumy = 0;
        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
            sumy += yl[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_s * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint16_t * qh = xr->qh + ib;
        device const half     * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 5) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[0] << 2) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[0] >> 1) & 0x700)));

            float sum = 0;
            for (int j = 0; j < 4; ++j) {
                sum += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                     + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4)
                     + yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                     + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            sumf[row] += (float)dh[0] * (sum + sumy * (qh[0] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA)) * (2*((qh[0] >> 12) & 7) + 1);

            dh += nb*sizeof(block_iq1_s)/2;
            qs += nb*sizeof(block_iq1_s);
            qh += nb*sizeof(block_iq1_s)/2;
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

void kernel_mul_mv_iq1_m_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_value,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    device const block_iq1_m * x = (device const block_iq1_m *) src0 + ib_row + offset0;
    device const float       * y = (device const float       *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    iq1m_scale_t scale;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        float4 sumy = {0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+ 8]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+16]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+24]; sumy[3] += yl[i+24];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_m * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint8_t  * qh = xr->qh + 2 * ib;
        device const uint16_t * sc = (device const uint16_t *)xr->scales;

        for (int row = 0; row < N_DST; row++) {
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));

            float2 sum = {0.f};
            for (int j = 0; j < 4; ++j) {
                sum[0] += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                        + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4);
                sum[1] += yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                        + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);

            sumf[row] += (float)scale.f16 * ((sum[0] + delta1) * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1) +
                                             (sum[1] + delta2) * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1));

            sc += nb*sizeof(block_iq1_m)/2;
            qs += nb*sizeof(block_iq1_m);
            qh += nb*sizeof(block_iq1_m);
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

void kernel_mul_mv_iq4_nl_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values_i8,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    threadgroup float * shared_values = (threadgroup float *)shared_values_i8;
    const int nb = ne00/QK4_NL;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    device const block_iq4_nl * x = (device const block_iq4_nl *) src0 + ib_row + offset0;
    device const float        * y = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;

    const int ix = tiisg/2;  // 0...15
    const int it = tiisg%2;  // 0 or 1

    shared_values[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2]={0.f}, all_sum;

    device const float * yb = y + ix * QK4_NL + it * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ib = ix; ib < nb; ib += 16) {

        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[1]; yl[3] = y4[5];

        for (int row = 0; row < 2 && first_row + row < ne01; ++row) {

            device const block_iq4_nl & xb = x[row*nb + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shared_values[q8[0]], shared_values[q8[1]], shared_values[q8[2]], shared_values[q8[3]]};
            qf2 = {shared_values[q8[4]], shared_values[q8[5]], shared_values[q8[6]], shared_values[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[2] | (q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shared_values[q8[0]], shared_values[q8[1]], shared_values[q8[2]], shared_values[q8[3]]};
            qf2 = {shared_values[q8[4]], shared_values[q8[5]], shared_values[q8[6]], shared_values[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);

        }

        yb += 16 * QK4_NL;
    }

    for (int row = 0; row < 2 && first_row + row < ne01; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

void kernel_mul_mv_iq4_xs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values_i8,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    threadgroup float * shared_values = (threadgroup float *)shared_values_i8;
    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    device const block_iq4_xs * x = (device const block_iq4_xs *) src0 + ib_row + offset0;
    device const float        * y = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;

    const int ix = tiisg/16;  // 0 or 1
    const int it = tiisg%16;  // 0...15
    const int ib = it/2;
    const int il = it%2;

    shared_values[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2]={0.f}, all_sum;

    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb; ibl += 2) {

        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[1]; yl[3] = y4[5];

        for (int row = 0; row < 2; ++row) {

            device const block_iq4_xs & xb = x[row*nb + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {shared_values[q8[0]], shared_values[q8[1]], shared_values[q8[2]], shared_values[q8[3]]};
            qf2 = {shared_values[q8[4]], shared_values[q8[5]], shared_values[q8[6]], shared_values[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[1] & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shared_values[q8[0]], shared_values[q8[1]], shared_values[q8[2]], shared_values[q8[3]]};
            qf2 = {shared_values[q8[4]], shared_values[q8[5]], shared_values[q8[6]], shared_values[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);

        }

        yb += 2 * QK_K;
    }

    for (int row = 0; row < 2; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_s_f32")]]
kernel void kernel_mul_mv_iq1_s_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq1_m_f32")]]
kernel void kernel_mul_mv_iq1_m_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32")]]
kernel void kernel_mul_mv_iq4_nl_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_xs_f32")]]
kernel void kernel_mul_mv_iq4_xs_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_xs_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

//============================= templates and their specializations =============================

// NOTE: this is not dequantizing - we are simply fitting the template
template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    float4x4 temp = *(((device float4x4 *)src));
    for (int i = 0; i < 16; i++){
        reg[i/4][i%4] = temp[i/4][i%4];
    }
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    half4x4 temp = *(((device half4x4 *)src));
    for (int i = 0; i < 16; i++){
        reg[i/4][i%4] = temp[i/4][i%4];
    }
}

template <typename type4x4>
void dequantize_bf16(device const bfloat4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i=0;i<8;i++) {
        reg[i/2][2*(i%2)+0] = d1 * (qs[i] & mask0) + md;
        reg[i/2][2*(i%2)+1] = d2 * (qs[i] & mask1) + md;
    }
}

template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i=0;i<8;i++) {
        reg[i/2][2*(i%2)+0] = ((qs[i] & mask0) * d1) + m;
        reg[i/2][2*(i%2)+1] = ((qs[i] & mask1) * d2) + m;
    }
}

template <typename type4x4>
void dequantize_q5_0(device const block_q5_0 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[i/2][2*(i%2)+0] = d * x0 + md;
        reg[i/2][2*(i%2)+1] = d * x1 + md;
    }
}

template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[i/2][2*(i%2)+0] = d * x0 + m;
        reg[i/2][2*(i%2)+1] = d * x1 + m;
    }
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const half d = xb->d;

    for (int i = 0; i < 16; i++) {
        reg[i/4][i%4] = (qs[i + 16*il] * d);
    }
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                 ((il/4)>0 ? 12  : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;

    il = (il/2) & 3;
    const half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;

    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
    }
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il<2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin; 
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * ql = (device const uint8_t *)xb->ql;
    device const uint8_t * qh = (device const uint8_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 64*(il/8) + 32*((il/2)&1) + 16*(il&1);
    qh = qh + 32*(il/8) + 16*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint16_t  kmask1 = il>1 ? (il>2 ? 192 : 48) : (il>0 ? 12 : 3);
    const uint16_t  kmask2 = il>1 ? 0xF0              : 0x0F;
    const float       coef = il>1 ? 1.f/16.f          : 1.f;
    const float ml = d_all * sc * 32.f;
    const float dl = d_all * sc * coef;
    for (int i = 0; i < 16; ++i) {
        const half q = il&1 ? ((ql[i] & kmask2) | ((qh[i] & kmask1) << 2))
                            : ((ql[i] & kmask2) | ((qh[i] & kmask1) << 4));
        reg[i/4][i%4] = dl * q - ml;
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    // each block of 32 needs 2 uint32_t's for the quants & scale, so 4 uint16_t's.
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq2_xs(device const block_iq2_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+0] & 511));
    uint8_t signs = ksigns_iq2xs[q2[2*il+0] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+1] & 511));
    signs = ksigns_iq2xs[q2[2*il+1] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_xxs(device const block_iq3_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * q3 = xb->qs + 8*ib32;
    device const uint16_t * gas = (device const uint16_t *)(xb->qs + QK_K/4) + 2*ib32;
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+0]);
    constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+1]);
    uint8_t signs = ksigns_iq2xs[(aux32 >> 14*il) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[1][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
    grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+2]);
    grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+3]);
    signs = ksigns_iq2xs[(aux32 >> (14*il+7)) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[3][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_s(device const block_iq3_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 8*ib32;
    device const uint8_t * signs = xb->signs + 4*ib32 + 2*il;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (1 + 2*((xb->scales[ib32/2] >> 4*(ib32%2)) & 0xf));
    constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+0] | ((qh << 8) & 256)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+1] | ((qh << 7) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i+0]);
        reg[1][i] = dl * grid2[i] * select(1, -1, signs[0] & kmask_iq2xs[i+4]);
    }
    grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+2] | ((qh << 6) & 256)));
    grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+3] | ((qh << 5) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * select(1, -1, signs[1] & kmask_iq2xs[i+0]);
        reg[3][i] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i+4]);
    }
}

template <typename type4x4>
void dequantize_iq2_s(device const block_iq2_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * signs = qs + QK_K/8;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[0] | ((qh << 8) & 0x300)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[1] | ((qh << 6) & 0x300)));
    for (int i = 0; i < 8; ++i) {
        reg[i/4+0][i%4] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i]);
        reg[i/4+2][i%4] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i]);
    }
}

template <typename type4x4>
void dequantize_iq1_s(device const block_iq1_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    const float d = xb->d;
    device const uint8_t  * qs = xb->qs + 4*ib32 + 2*il;
    device const uint16_t * qh = xb->qh;
    const float dl = d * (2*((qh[ib32] >> 12) & 7) + 1);
    const float ml = dl * (qh[ib32] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA);
    const uint16_t h = qh[ib32] >> 6*il;
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((h << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((h << 5) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml;
        reg[1][i] = dl * (grid1[i] >>  4) + ml;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml;
        reg[3][i] = dl * (grid2[i] >>  4) + ml;
    }
}

template <typename type4x4>
void dequantize_iq1_m(device const block_iq1_m * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * sc = (device const uint16_t *)xb->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = scale.f16;

    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * qh = xb->qh + 2*ib32 + il;

    const float dl  = d * (2*((sc[ib32/2] >> (6*(ib32%2)+3*il)) & 7) + 1);
    const float ml1 = dl * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    const float ml2 = dl * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml1;
        reg[1][i] = dl * (grid1[i] >>  4) + ml1;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml2;
        reg[3][i] = dl * (grid2[i] >>  4) + ml2;
    }
}

template <typename type4x4>
void dequantize_iq4_nl(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

template <typename type4x4>
void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4*ib32;
    const int ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((xb->scales_h >> 2*ib32) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_get_rows_q(
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int64_t ind = tiitg; ind < ne00/16; ind += tptg.x) {
        float4x4 temp;
        dequantize_func(((device const block_q *) ((const device char *) src0 + r*nb01 + i02*nb02)) + ind/nl, ind%nl, temp);
        *(((device float4x4 *) ((device char *) dst + i11*nb2 + i10*nb1)) + ind) = temp;
    }
}

template<typename T>
kernel void kernel_get_rows_f(
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int ind = tiitg; ind < ne00; ind += tptg.x) {
        ((      device float *) ((      device char *)  dst + i11*nb2  + i10*nb1))[ind] =
        ((const device T     *) ((const device char *) src0 + i02*nb02 +  r*nb01))[ind];
    }
}

kernel void kernel_get_rows_i32(
        device const  void * src0,
        device const  void * src1,
        device     int32_t * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int ind = tiitg; ind < ne00; ind += tptg.x) {
        ((      device int32_t *) ((      device char *) dst  + i11*nb2 + i10*nb1))[ind] =
        ((const device int32_t *) ((const device char *) src0 + i02*nb02 + r*nb01))[ind];
    }
}


#define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A
#define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B
#define THREAD_PER_BLOCK 128
#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

// each block_q contains 16*nl weights
template<typename T, typename T4x4, typename simdgroup_T8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
kernel void kernel_mul_mm(device const  uchar * src0,
                          device const  uchar * src1,
                          device        float * dst,
                          constant    int64_t & ne00,
                          constant    int64_t & ne02,
                          constant   uint64_t & nb01,
                          constant   uint64_t & nb02,
                          constant   uint64_t & nb03,
                          constant    int64_t & ne12,
                          constant   uint64_t & nb10,
                          constant   uint64_t & nb11,
                          constant   uint64_t & nb12,
                          constant   uint64_t & nb13,
                          constant    int64_t & ne0,
                          constant    int64_t & ne1,
                          constant       uint & r2,
                          constant       uint & r3,
                          threadgroup   uchar * shared_memory [[threadgroup(0)]],
                          uint3                 tgpig[[threadgroup_position_in_grid]],
                          uint                  tiitg[[thread_index_in_threadgroup]],
                          uint                  sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup T     * sa = (threadgroup T     *)(shared_memory);
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;
    const uint im = tgpig.z;

    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0*BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0*BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1*BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1*BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_T8x8     ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    uint   offset0 = (i12/r2)*nb02 + (i13/r3)*nb03;
    ushort offset1 = il/nl;

    device const block_q * x = (device const block_q *)(src0 + (r0*BLOCK_SIZE_M + thread_row)*nb01 + offset0) + offset1;
    device const float   * y = (device const float   *)(src1
        + nb13 * i13
        + nb12 * i12
        + nb11 * (r1 * BLOCK_SIZE_N + thread_col)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        T4x4 temp_a;
        dequantize_func(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (short i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg/THREAD_PER_ROW/8) \
            +                     (tiitg%THREAD_PER_ROW)*16 + (i/8)*8) \
            +                     (tiitg/THREAD_PER_ROW)%8  + (i&7)*8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL)*8*32 + 8*(tiitg/THREAD_PER_COL)) = *((device float2x4 *) y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup T     * lsma = (sa + THREAD_MAT_M*SG_MAT_SIZE*(sgitg%2));
        threadgroup float * lsmb = (sb + THREAD_MAT_N*SG_MAT_SIZE*(sgitg/2));

        #pragma unroll(4)
        for (short ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            #pragma unroll(4)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll(2)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M/SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N/SG_MAT_ROW * SG_MAT_SIZE;

            #pragma unroll(8)
            for (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
        }
    }

    if ((r0 + 1) * BLOCK_SIZE_M <= ne0 && (r1 + 1) * BLOCK_SIZE_N <= ne1) {
        device float * C = dst + (BLOCK_SIZE_M * r0 + 32 * (sgitg &  1)) \
                               + (BLOCK_SIZE_N * r1 + 16 * (sgitg >> 1)) * ne0 + im*ne1*ne0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i%4) + 8 * ne0 * (i/4), ne0);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1))*BLOCK_SIZE_M;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*BLOCK_SIZE_M*(i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                device float  * D  = dst + (r0*BLOCK_SIZE_M) + (r1*BLOCK_SIZE_N + j)*ne0 + im*ne1*ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*BLOCK_SIZE_M);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < n_rows/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < n_rows; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

// same as kernel_mul_mm_impl, but src1 and dst are accessed via indices stored in rowids
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
void kernel_mul_mm_id_impl(
        device const  uchar * src0,
        device const  uchar * src1,
        threadgroup ushort2 * rowids,
        device        float * dst,
        constant    int64_t & ne00,
        constant    int64_t & ne02,
        constant   uint64_t & nb01,
        constant   uint64_t & nb02,
        constant    int64_t & ne11,
        constant    int64_t & ne12,
        constant   uint64_t & nb10,
        constant   uint64_t & nb11,
        constant   uint64_t & nb12,
        constant    int64_t & ne0,
                    int64_t   ne1,
                    int64_t   ne0ne1,
        threadgroup   uchar * shared_memory,
        uint3                 tgpig[[threadgroup_position_in_grid]],
        uint                  tiitg[[thread_index_in_threadgroup]],
        uint                  sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half  * sa = (threadgroup half  *)(shared_memory);
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;

    if (r1 * BLOCK_SIZE_N >= ne1) return;

    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 c_res[8];
    for (int i = 0; i < 8; i++){
        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    short il = (tiitg % THREAD_PER_ROW);

    ushort offset1 = il/nl;

    threadgroup const auto & id = rowids[r1 * BLOCK_SIZE_N + thread_col];

    device const block_q * x = (device const block_q *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01) + offset1;
    device const float   * y = (device const float   *)(src1
        + nb12 * id[1]
        + nb11 * (id[0] % ne11)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        half4x4 temp_a;
        dequantize_func(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8) \
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = *((device float2x4 *)y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            for (int i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (int i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

            for (int i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
            }
        }
    }

    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float * C = dst + (BLOCK_SIZE_M * r0);
        if (sgitg == 0) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                threadgroup const auto & jid = rowids[r1 * BLOCK_SIZE_N + j];
                int joff =  jid[0] * ne0 + jid[1] * ne0ne1;
                for (int i = 0; i < n_rows; i++) {
                    *(C + i + joff) = *(temp_str + i + j * BLOCK_SIZE_M);
                }
            }
        }
    }
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void kernel_mul_mm_id(
        device const   uchar * src0s,
        device const   uchar * src1,
        device         float * dst,
        device const   uchar * ids,
        constant     int64_t & nei0,
        constant     int64_t & nei1,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne02,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        threadgroup    uchar * shared_memory [[threadgroup(0)]],
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {

    const int32_t i02 = tgpig.z;
    tgpig.z = 0;

    device const uchar * src0 = src0s + i02*nb02;

    // row indices
    threadgroup ushort2 * rowids = (threadgroup ushort2 *)(shared_memory + 8192);

    // TODO: parallelize this loop
    int64_t _ne1 = 0;
    for (ushort ii1 = 0; ii1 < nei1; ii1++) {
        for (ushort ii0 = 0; ii0 < nei0; ii0++) {
            int32_t id = ((device int32_t *) (ids + ii1*nbi1))[ii0];
            if (id == i02) {
                //if (tiitg == 0) {
                    rowids[_ne1] = ushort2(ii0, ii1);
                //}
                _ne1++;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    kernel_mul_mm_id_impl<block_q, nl, dequantize_func>(
        src0,
        src1,
        rowids,
        dst,
        ne00,
        ne02,
        nb01,
        nb02,
        ne11,
        ne12,
        nb10,
        nb11,
        nb12,
        ne0,
        _ne1,
        ne0*ne1,
        shared_memory,
        tgpig,
        tiitg,
        sgitg);
}

#define QK_NL 16

//
// get rows
//

typedef decltype(kernel_get_rows_f<float>) get_rows_f_t;

template [[host_name("kernel_get_rows_f32")]]  kernel get_rows_f_t kernel_get_rows_f<float>;
template [[host_name("kernel_get_rows_f16")]]  kernel get_rows_f_t kernel_get_rows_f<half>;
template [[host_name("kernel_get_rows_bf16")]] kernel get_rows_f_t kernel_get_rows_f<bfloat16_t>;

typedef decltype(kernel_get_rows_q<block_q4_0, 2, dequantize_q4_0>) get_rows_q_t;

template [[host_name("kernel_get_rows_q4_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_0,    2, dequantize_q4_0>;
template [[host_name("kernel_get_rows_q4_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_1,    2, dequantize_q4_1>;
template [[host_name("kernel_get_rows_q5_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_0,    2, dequantize_q5_0>;
template [[host_name("kernel_get_rows_q5_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_1,    2, dequantize_q5_1>;
template [[host_name("kernel_get_rows_q8_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q8_0,    2, dequantize_q8_0>;
template [[host_name("kernel_get_rows_q2_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_get_rows_q3_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_get_rows_q4_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_get_rows_q5_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_get_rows_q6_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_get_rows_iq2_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_get_rows_iq2_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_get_rows_iq3_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_get_rows_iq3_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_get_rows_iq2_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_get_rows_iq1_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_get_rows_iq1_m")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_get_rows_iq4_nl")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_get_rows_iq4_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_xs,  QK_NL, dequantize_iq4_xs>;

//
// matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, float4x4, 1, dequantize_f32>) mat_mm_t;

template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16>;
template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mat_mm_t kernel_mul_mm<bfloat16_t, bfloat4x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16>;
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs>;

//
// indirect matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_id<float4x4, 1, dequantize_f32>) mat_mm_id_t;

template [[host_name("kernel_mul_mm_id_f32_f32")]]     kernel mat_mm_id_t kernel_mul_mm_id<float4x4,      1,     dequantize_f32>;
template [[host_name("kernel_mul_mm_id_f16_f32")]]     kernel mat_mm_id_t kernel_mul_mm_id<half4x4,       1,     dequantize_f16>;
template [[host_name("kernel_mul_mm_id_q4_0_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q4_0,    2,     dequantize_q4_0>;
template [[host_name("kernel_mul_mm_id_q4_1_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q4_1,    2,     dequantize_q4_1>;
template [[host_name("kernel_mul_mm_id_q5_0_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q5_0,    2,     dequantize_q5_0>;
template [[host_name("kernel_mul_mm_id_q5_1_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q5_1,    2,     dequantize_q5_1>;
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q8_0,    2,     dequantize_q8_0>;
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_mul_mm_id_q3_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<block_iq4_xs,  QK_NL, dequantize_iq4_xs>;

//
// matrix-vector multiplication
//

typedef void (kernel_mul_mv_impl_t)(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb00,
                  uint64_t   nb01,
                  uint64_t   nb02,
                   int64_t   ne10,
                   int64_t   ne11,
                   int64_t   ne12,
                  uint64_t   nb10,
                  uint64_t   nb11,
                  uint64_t   nb12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
                   uint3     tgpig,
                   uint      tiisg);

typedef void (kernel_mul_mv2_impl_t)(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg);

template<kernel_mul_mv_impl_t impl_fn>
void mmv_fn(
        device const    char * src0,
        device const    char * src1,
        device         float * dst,
                     int64_t   ne00,
                     int64_t   ne01,
                     int64_t   ne02,
                    uint64_t   nb00,
                    uint64_t   nb01,
                    uint64_t   nb02,
                     int64_t   ne10,
                     int64_t   ne11,
                     int64_t   ne12,
                     int64_t   ne13,
                    uint64_t   nb10,
                    uint64_t   nb11,
                    uint64_t   nb12,
                     int64_t   ne0,
                     int64_t   ne1,
                    uint64_t   nb1,
                        uint   r2,
                        uint   r3,
        threadgroup int8_t   * shared_values,
        uint3                  tgpig,
        uint                   tiitg,
        uint                   tiisg,
        uint                   sgitg) {
    impl_fn(src0,src1,dst,ne00,ne01,ne02,nb00,nb01,nb02,ne10,ne11,ne12,nb10,nb11,nb12,ne0,ne1,r2,r3,tgpig,tiisg);
}

template<kernel_mul_mv2_impl_t impl_fn>
void mmv_fn(
        device const    char * src0,
        device const    char * src1,
        device         float * dst,
                     int64_t   ne00,
                     int64_t   ne01,
                     int64_t   ne02,
                    uint64_t   nb00,
                    uint64_t   nb01,
                    uint64_t   nb02,
                     int64_t   ne10,
                     int64_t   ne11,
                     int64_t   ne12,
                     int64_t   ne13,
                    uint64_t   nb10,
                    uint64_t   nb11,
                    uint64_t   nb12,
                     int64_t   ne0,
                     int64_t   ne1,
                    uint64_t   nb1,
                        uint   r2,
                        uint   r3,
        threadgroup int8_t   * shared_values,
        uint3                  tgpig,
        uint                   tiitg,
        uint                   tiisg,
        uint                   sgitg) {
    impl_fn(src0,(const device float *)src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,shared_values,tgpig,tiisg,sgitg);
}

typedef decltype(mmv_fn<kernel_mul_mv_impl<half, half4, half, half4>>) mul_mv_impl_fn_t;

template<mul_mv_impl_fn_t impl_fn>
kernel void kernel_mul_mv_id(
        device const    char * src0s,
        device const    char * src1,
        device         float * dst,
        device const    char * ids,
        constant     int64_t & nei0,
        constant     int64_t & nei1,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        threadgroup int8_t   * shared_values [[threadgroup(0)]],
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/nei0;
    const int idx = tgpig.z%nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*nbi1))[idx];

    const int64_t i11 = idx % ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device const char * src0_cur = src0s + i02*nb02;
    device const char * src1_cur = src1 + i11*nb11 + i12*nb12;
    device      float * dst_cur  = dst + i1*ne0 + i2*ne1*ne0;

    impl_fn(
        /* src0 */ src0_cur,
        /* src1 */ src1_cur,
        /* dst  */ dst_cur,
        /* ne00 */ ne00,
        /* ne01 */ ne01,
        /* ne02 */ 1,//ne02,
        /* nb00 */ nb00,
        /* nb01 */ nb01,
        /* nb02 */ nb02,
        /* ne10 */ ne10,
        /* ne11 */ 1,//ne11,
        /* ne12 */ 1,//ne12,
        /* ne13 */ 1,//ne13,
        /* nb10 */ nb10,
        /* nb11 */ nb11,
        /* nb12 */ nb12,
        /* ne0  */ ne0,
        /* ne1  */ 1,//ne1,
        /* nb1  */ nb1,
        /* r2   */ 1,
        /* r3   */ 1,
        shared_values,
        tgpig,
        tiitg,
        tiisg,
        sgitg);
}

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_impl<float, float4, float, float4>>>) kernel_mul_mv_id_t;

template [[host_name("kernel_mul_mv_id_f32_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_impl<float, float4, float, float4>>>;
template [[host_name("kernel_mul_mv_id_f16_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_impl<half, half4, float, float4>>>;
template [[host_name("kernel_mul_mv_id_q8_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q4_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q4_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q5_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q5_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q2_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q3_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q3_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q5_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q6_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq1_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_s_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq1_m_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_m_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xs_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq3_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_xxs_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq3_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_s_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_s_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq4_nl_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_nl_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq4_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_xs_f32_impl>>;

kernel void kernel_pool_2d_max_f32(
        device  const float * src0,
        device        float * dst,
        constant    int32_t & k0,
        constant    int32_t & k1,
        constant    int32_t & s0,
        constant    int32_t & s1,
        constant    int32_t & p0,
        constant    int32_t & p1,
        constant    int64_t & IH,
        constant    int64_t & IW,
        constant    int64_t & OH,
        constant    int64_t & OW,
        constant    int64_t & parallel_elements,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= parallel_elements) {
        return;
    }

    const int idx = gid;
    const int I_HW = IH * IW;
    const int O_HW = OH * OW;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / OW;
    const int cur_ow = idx % O_HW % OW;

    device const float * i_ptr = src0 + nc * I_HW;
    device       float * o_ptr = dst  + nc * O_HW;

    const int start_h = cur_oh * s1 - p1;
    const int bh = MAX(0,  start_h);
    const int eh = MIN(IH, start_h + k1);
    const int start_w = cur_ow * s0 - p0;
    const int bw = MAX(0,  start_w);
    const int ew = MIN(IW, start_w + k0);

    float res = -INFINITY;

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
            res = MAX(res, i_ptr[i * IW + j]);
        }
    }

    o_ptr[cur_oh * OW + cur_ow] = res;
}

kernel void kernel_pool_2d_avg_f32(
        device  const float * src0,
        device        float * dst,
        constant    int32_t & k0,
        constant    int32_t & k1,
        constant    int32_t & s0,
        constant    int32_t & s1,
        constant    int32_t & p0,
        constant    int32_t & p1,
        constant    int64_t & IH,
        constant    int64_t & IW,
        constant    int64_t & OH,
        constant    int64_t & OW,
        constant    int64_t & parallel_elements,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= parallel_elements) {
        return;
    }

    const int idx = gid;
    const int I_HW = IH * IW;
    const int O_HW = OH * OW;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / OW;
    const int cur_ow = idx % O_HW % OW;

    device const float * i_ptr = src0 + nc * I_HW;
    device       float * o_ptr = dst  + nc * O_HW;

    const int start_h = cur_oh * s1 - p1;
    const int bh = MAX(0,  start_h);
    const int eh = MIN(IH, start_h + k1);
    const int start_w = cur_ow * s0 - p0;
    const int bw = MAX(0,  start_w);
    const int ew = MIN(IW, start_w + k0);
    // const float scale = 1. / ((eh - bh) * (ew - bw));
    const float scale = 1. / (k0 * k1);

    float res = 0;

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
            float cur = i_ptr[i * IW + j];
            res += cur * scale;
        }
    }

    o_ptr[cur_oh * OW + cur_ow] = res;
}
