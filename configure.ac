dnl Process this file with autoconf to produce a configure script.

AC_INIT([sedna], [0.1])
AC_PREREQ(2.60)
AC_CONFIG_HEADERS([include/orion/config.h])
AC_CONFIG_FILES([GNUmakefile])
AC_SUBST(ac_configure_args)

ac_user_cxx=${CXX+y}

AC_PROG_CC
AC_PROG_CXX
AC_LANG_CPLUSPLUS

AC_DEFINE([WORDS_BIGENDIAN_SET], [1], [Define if WORDS_BIGENDIAN has been set.])
AC_C_BIGENDIAN()

AC_CHECK_HEADERS([sys/epoll.h numa.h])

AC_SEARCH_LIBS([numa_available], [numa], [AC_DEFINE([HAVE_LIBNUMA], [1], [Define if you have libnuma.])])

AC_ARG_ENABLE([debug],
[  --enable-debug    Turn on debugging],
[case "${enableval}" in
  yes) debug=true ;;
  no)  debug=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-debug]) ;;
esac],[debug=false])
AC_SUBST(DEBUG_ON, [$debug])

AC_ARG_ENABLE([sanitizer],
[  --enable-sanitizer    Turn on google sanitizer],
[case "${enableval}" in
  address) sanitizer=address ;;
  thread) sanitizer=thread;;
  none)  sanitizer=none ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-sanitizer]) ;;
esac],[sanitizer=none])
AC_SUBST(SANITIZER_ON, [$sanitizer])

AC_ARG_ENABLE([perfcount],
[  --enable-perfcount    Turn on perf count],
[case "${enableval}" in
  yes) perfcount=true ;;
  no) perfcount=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-perfcount]) ;;
esac],[perfcount=false])
AC_SUBST(PERF_COUNT_ON, [$perfcount])

AC_ARG_ENABLE([gprof],
[  --enable-gprof    Turn on GNU Profiling],
[case "${enableval}" in
  yes) gprof=true ;;
  no) gprof=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-gprof]) ;;
esac],[gprof=false])
AC_SUBST(GPROF_ON, [$gprof])

AC_ARG_ENABLE([googleprof],
[  --enable-googleprof    Turn on Google prof],
[case "${enableval}" in
  yes) googleprof=true ;;
  no) googleprof=false ;;
  *) AC_MSG_ERROR([bad value ${enableval} for --enable-googleprof]) ;;
esac],[googleprof=false])
AC_SUBST(GOOGLE_PROF_ON, [$googleprof])

AC_SUBST(THIRD_PARTY_HOME)
AC_SUBST(HADOOP_HOME)
AC_SUBST(JAVA_HOME)
AC_SUBST(JULIA_HOME)

dnl Builtins

AC_DEFUN([KVDB_CHECK_BUILTIN], [
    AC_CACHE_CHECK([for $1 builtin], [ac_cv_have_$1],
	[AC_LINK_IFELSE([AC_LANG_PROGRAM([$2], [])],
	    [ac_cv_have_$1=yes], [ac_cv_have_$1=no])])
    if test $ac_cv_have_$1 = yes; then
	AC_DEFINE(AS_TR_CPP([HAVE_$1]), [1], [Define if you have the $1 builtin.])
    fi
])

KVDB_CHECK_BUILTIN([__builtin_clz],
    [[unsigned f(unsigned x) { return __builtin_clz(x); }]])

KVDB_CHECK_BUILTIN([__builtin_clzl],
    [[unsigned long f(unsigned long x) { return __builtin_clzl(x); }]])

KVDB_CHECK_BUILTIN([__builtin_clzll],
    [[unsigned long long f(unsigned long long x) { return __builtin_clzll(x); }]])

KVDB_CHECK_BUILTIN([__builtin_ctz],
    [[unsigned f(unsigned x) { return __builtin_ctz(x); }]])

KVDB_CHECK_BUILTIN([__builtin_ctzl],
    [[unsigned long f(unsigned long x) { return __builtin_ctzl(x); }]])

KVDB_CHECK_BUILTIN([__builtin_ctzll],
    [[unsigned long long f(unsigned long long x) { return __builtin_ctzll(x); }]])

KVDB_CHECK_BUILTIN([__sync_synchronize], [[long x = 11;
    void f(long i) { long* y = &x; __sync_synchronize(); *y = i; }]])

KVDB_CHECK_BUILTIN([__sync_fetch_and_add],
    [[long f(long* x) { return __sync_fetch_and_add(x, 2L); }]])

KVDB_CHECK_BUILTIN([__sync_add_and_fetch],
    [[long f(long* x) { return __sync_add_and_fetch(x, 2L); }]])

KVDB_CHECK_BUILTIN([__sync_fetch_and_add_8],
    [[#include <stdint.h>
    int64_t f(int64_t* x) { return __sync_fetch_and_add(x, (int64_t) 2); }]])

KVDB_CHECK_BUILTIN([__sync_add_and_fetch_8],
    [[#include <stdint.h>
    int64_t f(int64_t* x) { return __sync_add_and_fetch(x, (int64_t) 2); }]])

KVDB_CHECK_BUILTIN([__sync_fetch_and_or],
    [[long f(long* x) { return __sync_fetch_and_or(x, 2L); }]])

KVDB_CHECK_BUILTIN([__sync_or_and_fetch],
    [[long f(long* x) { return __sync_or_and_fetch(x, 2L); }]])

KVDB_CHECK_BUILTIN([__sync_fetch_and_or_8],
    [[#include <stdint.h>
    int64_t f(int64_t* x) { return __sync_fetch_and_or(x, (int64_t) 2); }]])

KVDB_CHECK_BUILTIN([__sync_or_and_fetch_8],
    [[#include <stdint.h>
    int64_t f(int64_t* x) { return __sync_or_and_fetch(x, (int64_t) 2); }]])

KVDB_CHECK_BUILTIN([__sync_bool_compare_and_swap],
    [[bool f(long* x, long y, long z) { return __sync_bool_compare_and_swap(x, y, z); }]])

KVDB_CHECK_BUILTIN([__sync_bool_compare_and_swap_8],
    [[#include <stdint.h>
    bool f(int64_t* x, int64_t y, int64_t z) { return __sync_bool_compare_and_swap(x, y, z); }]])

KVDB_CHECK_BUILTIN([__sync_val_compare_and_swap],
    [[long f(long* x, long y, long z) { return __sync_val_compare_and_swap(x, y, z); }]])

KVDB_CHECK_BUILTIN([__sync_val_compare_and_swap_8],
    [[#include <stdint.h>
    int64_t f(int64_t* x, int64_t y, int64_t z) { return __sync_val_compare_and_swap(x, y, z); }]])

KVDB_CHECK_BUILTIN([__sync_lock_test_and_set],
    [[long f(long* x) { return __sync_lock_test_and_set(x, 1); }]])

KVDB_CHECK_BUILTIN([__sync_lock_test_and_set_val],
    [[long f(long* x, long y) { return __sync_lock_test_and_set(x, y); }]])

KVDB_CHECK_BUILTIN([__sync_lock_release_set],
    [[void f(long* x) { __sync_lock_release(x); }]])

dnl Types

AC_DEFUN([KVDB_CHECK_SAME_TYPE], [
    pushdef([KVDB_CST_VAR], [AS_TR_SH([ac_cv_have_same_type_$1_is_$2])])
    AC_CACHE_CHECK([whether $1 and $2 are the same type], KVDB_CST_VAR,
	[AC_COMPILE_IFELSE([AC_LANG_PROGRAM([$3
int f($1) {return 0;} int f($2) {return 0;}], [])],
	    [KVDB_CST_VAR=no], [KVDB_CST_VAR=yes])])
    if test $KVDB_CST_VAR = yes; then
	AC_DEFINE(AS_TR_CPP([HAVE_$1_IS_$2]), [1], [Define if $1 and $2 are the same type.])
    fi
    popdef([KVDB_CST_VAR])
])

KVDB_CHECK_SAME_TYPE([off_t], [long], [#include <stdio.h>])
KVDB_CHECK_SAME_TYPE([off_t], [long long], [#include <stdio.h>])
KVDB_CHECK_SAME_TYPE([int64_t], [long], [#include <stdint.h>])
KVDB_CHECK_SAME_TYPE([int64_t], [long long], [#include <stdint.h>])
KVDB_CHECK_SAME_TYPE([size_t], [unsigned], [#include <stdio.h>])
KVDB_CHECK_SAME_TYPE([size_t], [unsigned long], [#include <stdio.h>])
KVDB_CHECK_SAME_TYPE([size_t], [unsigned long long], [#include <stdio.h>])

AC_CHECK_TYPES([long long])
AC_CHECK_SIZEOF([short])
AC_CHECK_SIZEOF([int])
AC_CHECK_SIZEOF([long])
AC_CHECK_SIZEOF([long long])
AC_CHECK_SIZEOF([void *])

AC_CHECK_DECLS([getline])

AC_CHECK_HEADERS([time.h execinfo.h])
AC_CHECK_DECLS([clock_gettime], [], [], [#if HAVE_TIME_H
# include <time.h>
#endif])
AC_SEARCH_LIBS([clock_gettime], [rt])
AC_CHECK_FUNCS([clock_gettime])


AC_MSG_CHECKING([whether MADV_HUGEPAGE is supported])
AC_PREPROC_IFELSE([AC_LANG_PROGRAM([[#include <sys/mman.h>
#ifndef MADV_HUGEPAGE
#error "no"
#endif]], [])],
                  [have_madv_hugepage=yes], [have_madv_hugepage=no])
AC_MSG_RESULT([$have_madv_hugepage])
if test $have_madv_hugepage = yes; then
    AC_DEFINE([HAVE_MADV_HUGEPAGE], [1], [Define if MADV_HUGEPAGE is supported.])
fi

AC_MSG_CHECKING([whether MAP_HUGETLB is supported])
AC_PREPROC_IFELSE([AC_LANG_PROGRAM([[#include <sys/mman.h>
#ifndef MAP_HUGETLB
#error "no"
#endif]], [])],
                  [have_map_hugetlb=yes], [have_map_hugetlb=no])
AC_MSG_RESULT([$have_map_hugetlb])
if test $have_map_hugetlb = yes; then
    AC_DEFINE([HAVE_MAP_HUGETLB], [1], [Define if MAP_HUGETLB is supported.])
fi

AC_ARG_ENABLE([superpage],
    [AS_HELP_STRING([--disable-superpage],
	    [disable superpage support])],
    [], [enable_superpage=maybe])
if test "$enable_superpage $have_madv_hugepage $have_map_hugetlb" = "yes no no"; then
    AC_MSG_ERROR([
Error: superpages are not supported on this machine.
Try again without --enable-superpage.
])
elif test "$enable_superpage $have_madv_hugepage $have_map_hugetlb" != "maybe no no" -a "$enable_superpage" != no; then
    AC_DEFINE_UNQUOTED([HAVE_SUPERPAGE], [1], [Define if superpage support is enabled.])
fi

AC_ARG_ENABLE([assert],
    [],
    [AC_MSG_WARN([Use --disable-assertions instead of --disable-assert.])])
AC_ARG_ENABLE([assertions],
    [AS_HELP_STRING([--disable-assertions],
	    [disable debugging assertions])])
if test "$enable_assertions" != no -o "(" -z "$enable_assertions" -a "$enable_assert" != no ")"; then
    AC_DEFINE_UNQUOTED([ENABLE_ASSERTIONS], [1], [Define to enable debugging assertions.])
fi

AC_ARG_ENABLE([preconditions],
    [AS_HELP_STRING([--disable-preconditions],
            [disable precondition assertions])])
if test "$enable_preconditions" = no; then
    AC_DEFINE_UNQUOTED([ENABLE_PRECONDITIONS], [0], [Define to enable precondition assertions.])
elif test -n "$enable_preconditions"; then
    AC_DEFINE_UNQUOTED([ENABLE_PRECONDITIONS], [1], [Define to enable precondition assertions.])
fi

AC_ARG_ENABLE([invariants],
    [AS_HELP_STRING([--disable-invariants],
            [disable invariant assertions])])
if test "$enable_invariants" = no; then
    AC_DEFINE_UNQUOTED([ENABLE_INVARIANTS], [0], [Define to enable invariant assertions.])
elif test -n "$enable_preconditions"; then
    AC_DEFINE_UNQUOTED([ENABLE_INVARIANTS], [1], [Define to enable invariant assertions.])
fi

AC_ARG_ENABLE([perf],
    [AS_HELP_STRING([--enable-perf],
            [enable perf measurements])])
if test "$enable_perf" = no; then
    AC_DEFINE_UNQUOTED([ENABLE_PERF], [0], [Define to enable perf measurements.])
elif test -n "$enable_perf"; then
    AC_DEFINE_UNQUOTED([ENABLE_PERF], [1], [Define to enable perf measurements.])
fi

AC_DEFINE_UNQUOTED([CACHE_LINE_SIZE], [64], [Assumed size of a cache line.])

AH_TOP([#pragma once])

AC_DEFINE_UNQUOTED([HAVE_UNALIGNED_ACCESS], [1], [Define if unaligned accesses are OK.])

AC_OUTPUT
