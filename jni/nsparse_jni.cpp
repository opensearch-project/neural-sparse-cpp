/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include <jni.h>
#include <cstring>
#include <memory>
#include <string>

#ifdef __linux__
#include <cerrno>
#include <malloc.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cstdio>
#endif

#include <omp.h>

#include "nsparse/index.h"
#include "nsparse/index_factory.h"
#include "nsparse/io/index_io.h"
#include "nsparse/seismic_index.h"
#include "nsparse/seismic_scalar_quantized_index.h"
#include "nsparse/types.h"

namespace {

void throw_java_exception(JNIEnv* env, const char* msg) {
    jclass cls = env->FindClass("java/lang/RuntimeException");
    if (cls != nullptr) {
        env->ThrowNew(cls, msg);
    }
}

nsparse::Index* to_index(jlong ptr) {
    return reinterpret_cast<nsparse::Index*>(ptr);
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_createIndex(
    JNIEnv* env, jclass, jint dimension, jstring description) {
    try {
        const char* desc = env->GetStringUTFChars(description, nullptr);
        std::string desc_str(desc);
        env->ReleaseStringUTFChars(description, desc);

        nsparse::Index* index = nsparse::index_factory(dimension, desc_str.c_str());
        return reinterpret_cast<jlong>(index);
    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
        return 0;
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_addVectors(
    JNIEnv* env, jclass, jlong indexPtr, jint n, jintArray indptrArr,
    jshortArray indicesArr, jfloatArray valuesArr) {
    try {
        nsparse::Index* index = to_index(indexPtr);

        jint* indptr = env->GetIntArrayElements(indptrArr, nullptr);
        jshort* indices = env->GetShortArrayElements(indicesArr, nullptr);
        jfloat* values = env->GetFloatArrayElements(valuesArr, nullptr);

        static_assert(sizeof(nsparse::idx_t) == sizeof(jint));
        static_assert(sizeof(nsparse::term_t) == sizeof(jshort));
        static_assert(sizeof(float) == sizeof(jfloat));

        index->add(
            static_cast<nsparse::idx_t>(n),
            reinterpret_cast<const nsparse::idx_t*>(indptr),
            reinterpret_cast<const nsparse::term_t*>(indices),
            reinterpret_cast<const float*>(values)
        );

        env->ReleaseIntArrayElements(indptrArr, indptr, JNI_ABORT);
        env->ReleaseShortArrayElements(indicesArr, indices, JNI_ABORT);
        env->ReleaseFloatArrayElements(valuesArr, values, JNI_ABORT);
    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_addVectorsWithIds(
    JNIEnv* env, jclass, jlong indexPtr, jint n, jintArray indptrArr,
    jshortArray indicesArr, jfloatArray valuesArr, jintArray idsArr) {
    try {
        nsparse::Index* index = to_index(indexPtr);

        jint* indptr = env->GetIntArrayElements(indptrArr, nullptr);
        jshort* indices = env->GetShortArrayElements(indicesArr, nullptr);
        jfloat* values = env->GetFloatArrayElements(valuesArr, nullptr);
        jint* ids = env->GetIntArrayElements(idsArr, nullptr);

        index->add_with_ids(
            static_cast<nsparse::idx_t>(n),
            reinterpret_cast<const nsparse::idx_t*>(indptr),
            reinterpret_cast<const nsparse::term_t*>(indices),
            reinterpret_cast<const float*>(values),
            reinterpret_cast<const nsparse::idx_t*>(ids)
        );

        env->ReleaseIntArrayElements(indptrArr, indptr, JNI_ABORT);
        env->ReleaseShortArrayElements(indicesArr, indices, JNI_ABORT);
        env->ReleaseFloatArrayElements(valuesArr, values, JNI_ABORT);
        env->ReleaseIntArrayElements(idsArr, ids, JNI_ABORT);
    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_reserveIndex(
    JNIEnv* env, jclass, jlong indexPtr, jlong numVectors, jlong totalNnz) {
    try {
        to_index(indexPtr)->reserve(static_cast<size_t>(numVectors),
                                    static_cast<size_t>(totalNnz));
    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_buildIndex(
    JNIEnv* env, jclass, jlong indexPtr) {
    try {
#ifdef __linux__
        // Reduce mmap threshold so allocations >= 128KB use mmap instead of sbrk.
        // mmap'd regions are returned to the OS immediately on free, preventing
        // heap fragmentation that would otherwise retain tens of GB after build.
        mallopt(M_MMAP_THRESHOLD, 128 * 1024);
        mallopt(M_TRIM_THRESHOLD, 128 * 1024);
#endif
        to_index(indexPtr)->build();
    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
    }
}

JNIEXPORT jobject JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_search(
    JNIEnv* env, jclass, jlong indexPtr, jint nQueries, jintArray indptrArr,
    jshortArray indicesArr, jfloatArray valuesArr, jint k,
    jfloat heapFactor, jint cut) {
    try {
        nsparse::Index* index = to_index(indexPtr);

        jint* indptr = env->GetIntArrayElements(indptrArr, nullptr);
        jshort* indices = env->GetShortArrayElements(indicesArr, nullptr);
        jfloat* values = env->GetFloatArrayElements(valuesArr, nullptr);

        int total = nQueries * k;
        std::vector<float> distances(total);
        std::vector<nsparse::idx_t> labels(total);

        nsparse::SeismicSearchParameters params(static_cast<int>(cut), static_cast<float>(heapFactor));

        index->search(
            static_cast<nsparse::idx_t>(nQueries),
            reinterpret_cast<const nsparse::idx_t*>(indptr),
            reinterpret_cast<const nsparse::term_t*>(indices),
            reinterpret_cast<const float*>(values),
            static_cast<int>(k),
            distances.data(),
            labels.data(),
            &params
        );

        env->ReleaseIntArrayElements(indptrArr, indptr, JNI_ABORT);
        env->ReleaseShortArrayElements(indicesArr, indices, JNI_ABORT);
        env->ReleaseFloatArrayElements(valuesArr, values, JNI_ABORT);

        jfloatArray distArr = env->NewFloatArray(total);
        env->SetFloatArrayRegion(distArr, 0, total, distances.data());

        jintArray labelArr = env->NewIntArray(total);
        static_assert(sizeof(nsparse::idx_t) == sizeof(jint));
        env->SetIntArrayRegion(labelArr, 0, total, reinterpret_cast<jint*>(labels.data()));

        jclass resultClass = env->FindClass("org/opensearch/neuralsearch/sparse/jni/SearchResult");
        jmethodID ctor = env->GetMethodID(resultClass, "<init>", "([F[II)V");
        return env->NewObject(resultClass, ctor, distArr, labelArr, static_cast<jint>(k));
    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
        return nullptr;
    }
}

JNIEXPORT jobject JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_searchSQ(
    JNIEnv* env, jclass, jlong indexPtr, jint nQueries, jintArray indptrArr,
    jshortArray indicesArr, jfloatArray valuesArr, jint k,
    jfloat heapFactor, jint cut, jfloat vmin, jfloat vmax) {
    try {
        nsparse::Index* index = to_index(indexPtr);

        jint* indptr = env->GetIntArrayElements(indptrArr, nullptr);
        jshort* indices = env->GetShortArrayElements(indicesArr, nullptr);
        jfloat* values = env->GetFloatArrayElements(valuesArr, nullptr);

        int total = nQueries * k;
        std::vector<float> distances(total);
        std::vector<nsparse::idx_t> labels(total);

        nsparse::SeismicSQSearchParameters params(
            static_cast<float>(vmin), static_cast<float>(vmax),
            static_cast<int>(cut), static_cast<float>(heapFactor));

        index->search(
            static_cast<nsparse::idx_t>(nQueries),
            reinterpret_cast<const nsparse::idx_t*>(indptr),
            reinterpret_cast<const nsparse::term_t*>(indices),
            reinterpret_cast<const float*>(values),
            static_cast<int>(k),
            distances.data(),
            labels.data(),
            &params
        );

        env->ReleaseIntArrayElements(indptrArr, indptr, JNI_ABORT);
        env->ReleaseShortArrayElements(indicesArr, indices, JNI_ABORT);
        env->ReleaseFloatArrayElements(valuesArr, values, JNI_ABORT);

        jfloatArray distArr = env->NewFloatArray(total);
        env->SetFloatArrayRegion(distArr, 0, total, distances.data());

        jintArray labelArr = env->NewIntArray(total);
        env->SetIntArrayRegion(labelArr, 0, total, reinterpret_cast<jint*>(labels.data()));

        jclass resultClass = env->FindClass("org/opensearch/neuralsearch/sparse/jni/SearchResult");
        jmethodID ctor = env->GetMethodID(resultClass, "<init>", "([F[II)V");
        return env->NewObject(resultClass, ctor, distArr, labelArr, static_cast<jint>(k));
    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
        return nullptr;
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_buildAndSaveIndex(
    JNIEnv* env, jclass, jlong indexPtr, jstring path) {
    try {
        const char* pathStr = env->GetStringUTFChars(path, nullptr);
        std::string pathCopy(pathStr);
        env->ReleaseStringUTFChars(path, pathStr);

        nsparse::Index* index = to_index(indexPtr);
#ifdef __linux__
        mallopt(M_MMAP_THRESHOLD, 128 * 1024);
        mallopt(M_TRIM_THRESHOLD, 128 * 1024);
#endif
        index->build_and_save(pathCopy.c_str());
        index->release_build_memory();
        delete index;
#ifdef __linux__
        malloc_trim(0);
#endif
    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_saveIndex(
    JNIEnv* env, jclass, jlong indexPtr, jstring path) {
    try {
        const char* pathStr = env->GetStringUTFChars(path, nullptr);
        std::string pathCopy(pathStr);
        env->ReleaseStringUTFChars(path, pathStr);

        nsparse::Index* index = to_index(indexPtr);
        nsparse::write_index(index, pathCopy.data());
        // Release vectors_ and clustered_inverted_lists immediately after save.
        // The index will be loaded fresh from disk for search — keeping build
        // data in memory wastes ~80GB and causes OOM when loadIndex runs.
        index->release_build_memory();
#ifdef __linux__
        malloc_trim(0);
#endif
    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
    }
}

JNIEXPORT jlong JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_loadIndex(
    JNIEnv* env, jclass, jstring path) {
    try {
        const char* pathStr = env->GetStringUTFChars(path, nullptr);
        std::string pathCopy(pathStr);
        env->ReleaseStringUTFChars(path, pathStr);

        nsparse::Index* index = nsparse::read_index(pathCopy.data());
        return reinterpret_cast<jlong>(index);
    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
        return 0;
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_deleteIndex(
    JNIEnv* env, jclass, jlong indexPtr) {
    try {
        delete to_index(indexPtr);
#ifdef __linux__
        // Restore default mmap threshold for subsequent allocations (search path).
        mallopt(M_MMAP_THRESHOLD, 128 * 1024);
        mallopt(M_TRIM_THRESHOLD, 128 * 1024);
        malloc_trim(0);
#endif
    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
    }
}

JNIEXPORT jint JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_numVectors(
    JNIEnv* env, jclass, jlong indexPtr) {
    try {
        return static_cast<jint>(to_index(indexPtr)->num_vectors());
    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
        return 0;
    }
}

JNIEXPORT jint JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_getDimension(
    JNIEnv* env, jclass, jlong indexPtr) {
    try {
        return static_cast<jint>(to_index(indexPtr)->get_dimension());
    } catch (const std::exception& e) {
        throw_java_exception(env, e.what());
        return 0;
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_setNumThreads(
    JNIEnv*, jclass, jint numThreads) {
    omp_set_num_threads(static_cast<int>(numThreads));
}

JNIEXPORT void JNICALL Java_org_opensearch_neuralsearch_sparse_jni_NsparseJni_evictPageCache(
    JNIEnv* env, jclass, jstring dirPath, jstring suffix) {
#ifdef __linux__
    const char* dir = env->GetStringUTFChars(dirPath, nullptr);
    const char* sfx = env->GetStringUTFChars(suffix, nullptr);
    size_t sfx_len = strlen(sfx);
    size_t dir_len = strlen(dir);

    FILE* maps = fopen("/proc/self/maps", "r");
    if (maps) {
        char* line = nullptr;
        size_t line_cap = 0;
        ssize_t line_len;
        while ((line_len = getline(&line, &line_cap, maps)) != -1) {
            char* p = line;
            int fields = 0;
            while (*p && fields < 5) {
                while (*p && *p != ' ') p++;
                while (*p == ' ') p++;
                fields++;
            }
            if (*p != '/') continue;
            char* pathname = p;
            size_t plen = strlen(pathname);
            if (plen > 0 && pathname[plen - 1] == '\n') {
                pathname[plen - 1] = '\0';
                plen--;
            }
            if (plen > dir_len + sfx_len &&
                strncmp(pathname, dir, dir_len) == 0 &&
                pathname[dir_len] == '/' &&
                strcmp(pathname + plen - sfx_len, sfx) == 0) {
                unsigned long start = 0, end = 0;
                if (sscanf(line, "%lx-%lx", &start, &end) == 2 && end > start) {
                    if (madvise(reinterpret_cast<void*>(start), end - start, MADV_DONTNEED) != 0) {
                        fprintf(stderr, "madvise MADV_DONTNEED failed: %s (addr=%lx, len=%lu)\n",
                                strerror(errno), start, end - start);
                    }
                }
            }
        }
        free(line);
        fclose(maps);
    }

    env->ReleaseStringUTFChars(dirPath, dir);
    env->ReleaseStringUTFChars(suffix, sfx);
#endif
}

}  // extern "C"
