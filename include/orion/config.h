/* include/orion/config.h.  Generated from config.h.in by configure.  */
/* include/sedna/config.h.in.  Generated from configure.ac by autoheader.  */

#pragma once

/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* Assumed size of a cache line. */
#define CACHE_LINE_SIZE 64

/* Define to enable debugging assertions. */
#define ENABLE_ASSERTIONS 1

/* Define to 1 if you have the `clock_gettime' function. */
#define HAVE_CLOCK_GETTIME 1

/* Define to 1 if you have the declaration of `clock_gettime', and to 0 if you
   don't. */
#define HAVE_DECL_CLOCK_GETTIME 1

/* Define to 1 if you have the declaration of `getline', and to 0 if you
   don't. */
#define HAVE_DECL_GETLINE 1

/* Define to 1 if you have the <execinfo.h> header file. */
#define HAVE_EXECINFO_H 1

/* Define if int64_t and long are the same type. */
#define HAVE_INT64_T_IS_LONG 1

/* Define if int64_t and long long are the same type. */
/* #undef HAVE_INT64_T_IS_LONG_LONG */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define if you have libnuma. */
#define HAVE_LIBNUMA 1

/* Define to 1 if the system has the type `long long'. */
#define HAVE_LONG_LONG 1

/* Define if MADV_HUGEPAGE is supported. */
#define HAVE_MADV_HUGEPAGE 1

/* Define if MAP_HUGETLB is supported. */
#define HAVE_MAP_HUGETLB 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the <numa.h> header file. */
#define HAVE_NUMA_H 1

/* Define if off_t and long are the same type. */
#define HAVE_OFF_T_IS_LONG 1

/* Define if off_t and long long are the same type. */
/* #undef HAVE_OFF_T_IS_LONG_LONG */

/* Define if size_t and unsigned are the same type. */
/* #undef HAVE_SIZE_T_IS_UNSIGNED */

/* Define if size_t and unsigned long are the same type. */
#define HAVE_SIZE_T_IS_UNSIGNED_LONG 1

/* Define if size_t and unsigned long long are the same type. */
/* #undef HAVE_SIZE_T_IS_UNSIGNED_LONG_LONG */

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define if superpage support is enabled. */
#define HAVE_SUPERPAGE 1

/* Define to 1 if you have the <sys/epoll.h> header file. */
#define HAVE_SYS_EPOLL_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <time.h> header file. */
#define HAVE_TIME_H 1

/* Define to 1 if you have the <type_traits> header file. */
/* #undef HAVE_TYPE_TRAITS */

/* Define if unaligned accesses are OK. */
#define HAVE_UNALIGNED_ACCESS 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define if you have the __builtin_clz builtin. */
#define HAVE___BUILTIN_CLZ 1

/* Define if you have the __builtin_clzl builtin. */
#define HAVE___BUILTIN_CLZL 1

/* Define if you have the __builtin_clzll builtin. */
#define HAVE___BUILTIN_CLZLL 1

/* Define if you have the __builtin_ctz builtin. */
#define HAVE___BUILTIN_CTZ 1

/* Define if you have the __builtin_ctzl builtin. */
#define HAVE___BUILTIN_CTZL 1

/* Define if you have the __builtin_ctzll builtin. */
#define HAVE___BUILTIN_CTZLL 1

/* Define if you have the __has_trivial_copy compiler intrinsic. */
/* #undef HAVE___HAS_TRIVIAL_COPY */

/* Define if you have the __sync_add_and_fetch builtin. */
#define HAVE___SYNC_ADD_AND_FETCH 1

/* Define if you have the __sync_add_and_fetch_8 builtin. */
#define HAVE___SYNC_ADD_AND_FETCH_8 1

/* Define if you have the __sync_bool_compare_and_swap builtin. */
#define HAVE___SYNC_BOOL_COMPARE_AND_SWAP 1

/* Define if you have the __sync_bool_compare_and_swap_8 builtin. */
#define HAVE___SYNC_BOOL_COMPARE_AND_SWAP_8 1

/* Define if you have the __sync_fetch_and_add builtin. */
#define HAVE___SYNC_FETCH_AND_ADD 1

/* Define if you have the __sync_fetch_and_add_8 builtin. */
#define HAVE___SYNC_FETCH_AND_ADD_8 1

/* Define if you have the __sync_fetch_and_or builtin. */
#define HAVE___SYNC_FETCH_AND_OR 1

/* Define if you have the __sync_fetch_and_or_8 builtin. */
#define HAVE___SYNC_FETCH_AND_OR_8 1

/* Define if you have the __sync_lock_release_set builtin. */
#define HAVE___SYNC_LOCK_RELEASE_SET 1

/* Define if you have the __sync_lock_test_and_set builtin. */
#define HAVE___SYNC_LOCK_TEST_AND_SET 1

/* Define if you have the __sync_lock_test_and_set_val builtin. */
#define HAVE___SYNC_LOCK_TEST_AND_SET_VAL 1

/* Define if you have the __sync_or_and_fetch builtin. */
#define HAVE___SYNC_OR_AND_FETCH 1

/* Define if you have the __sync_or_and_fetch_8 builtin. */
#define HAVE___SYNC_OR_AND_FETCH_8 1

/* Define if you have the __sync_synchronize builtin. */
#define HAVE___SYNC_SYNCHRONIZE 1

/* Define if you have the __sync_val_compare_and_swap builtin. */
#define HAVE___SYNC_VAL_COMPARE_AND_SWAP 1

/* Define if you have the __sync_val_compare_and_swap_8 builtin. */
#define HAVE___SYNC_VAL_COMPARE_AND_SWAP_8 1

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT ""

/* Define to the full name of this package. */
#define PACKAGE_NAME "sedna"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "sedna 0.1"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "sedna"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "0.1"

/* The size of `int', as computed by sizeof. */
#define SIZEOF_INT 4

/* The size of `long', as computed by sizeof. */
#define SIZEOF_LONG 8

/* The size of `long long', as computed by sizeof. */
#define SIZEOF_LONG_LONG 8

/* The size of `short', as computed by sizeof. */
#define SIZEOF_SHORT 2

/* The size of `void *', as computed by sizeof. */
#define SIZEOF_VOID_P 8

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif

/* Define if WORDS_BIGENDIAN has been set. */
#define WORDS_BIGENDIAN_SET 1
