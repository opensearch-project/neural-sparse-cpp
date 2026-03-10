/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

%module swignsparse

%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "nsparse/types.h"
#include "nsparse/io/io.h"
#include "nsparse/id_selector.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/invlists/inverted_lists.h"
#include "nsparse/index.h"
#include "nsparse/brutal_index.h"
#include "nsparse/seismic_common.h"
#include "nsparse/seismic_index.h"
#include "nsparse/seismic_scalar_quantized_index.h"
#include "nsparse/id_map_index.h"
#include "nsparse/utils/scalar_quantizer.h"
#include "nsparse/cluster/random_kmeans.h"
#include "nsparse/inverted_index.h"
#include "nsparse/index_factory.h"
#include "nsparse/io/index_io.h"
%}

%init %{
import_array();
%}

%include "nsparse_typemaps.i"
%include "factory.i"
%include "nsparse/io/io.h"
%include "nsparse/types.h"
%include "nsparse/sparse_vectors.h"
%include "nsparse/invlists/inverted_lists.h"
%include "nsparse/utils/scalar_quantizer.h"

// IDSelector support - expose the class hierarchy for Python
// Multi-argument typemap: Python passes a single int32 array, n is inferred
%typemap(in) (size_t n, const nsparse::idx_t* indices) (Py_buffer view) {
    if (PyObject_GetBuffer($input, &view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) == -1) {
        SWIG_fail;
    }
    if (strcmp(view.format, "i") != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError, "Expected int32 array for indices");
        SWIG_fail;
    }
    $1 = (size_t)(view.shape[0]);
    $2 = (nsparse::idx_t*)view.buf;
}
%typemap(freearg) (size_t n, const nsparse::idx_t* indices) {
    PyBuffer_Release(&view$argnum);
}
%include "nsparse/id_selector.h"

// IO classes needed for index serialization (must come before index.h)
// Ignore internal IO classes - not needed in Python API
%ignore nsparse::SeismicIndex::write_index;
%ignore nsparse::SeismicIndex::read_index;
%ignore nsparse::SeismicScalarQuantizedIndex::write_index;
%ignore nsparse::SeismicScalarQuantizedIndex::read_index;
%ignore nsparse::IDMapIndex::write_index;
%ignore nsparse::IDMapIndex::read_index;
%ignore nsparse::SparseVectors::serialize;
%ignore nsparse::SparseVectors::deserialize;
%ignore nsparse::InvertedListClusters::serialize;
%ignore nsparse::InvertedListClusters::deserialize;
%ignore nsparse::detail; 

// Ignore the protected std::vector-based methods
%ignore nsparse::Index::add(idx_t, std::vector<idx_t>&, std::vector<term_t>&, std::vector<float>&);
%ignore nsparse::Index::search(idx_t, std::vector<idx_t>&, std::vector<term_t>&, std::vector<float>&, int);

// Ignore SeismicIndex's vector-based search methods, expose only pointer-based
%ignore nsparse::SeismicIndex::search(idx_t, std::vector<idx_t>&, std::vector<term_t>&, std::vector<float>&, int);
%ignore nsparse::SeismicIndex::search(idx_t, std::vector<idx_t>&, std::vector<term_t>&, std::vector<float>&, int, int, float);

// Ignore SeismicScalarQuantizedIndex's vector-based search methods
%ignore nsparse::SeismicScalarQuantizedIndex::search(idx_t, std::vector<idx_t>&, std::vector<term_t>&, std::vector<float>&, int);

// Ignore InvertedIndex's protected search method (returns pair_of_score_id_vectors_t)
%ignore nsparse::InvertedIndex::search;

// Ignore the original search methods - we'll provide our own via %extend
%ignore nsparse::Index::search;

// Extend Index to provide search methods
// Use separate names because both overloads have the same number of visible Python args
// (labels and distances have numinputs=0, so both would appear as 5-arg functions to Python)
%extend nsparse::Index {
    // Version without SearchParameters
    void search_c(idx_t n, const idx_t* indptr, const term_t* indices,
                  const float* values, int k, float* distances, idx_t* labels) {
        $self->search(n, indptr, indices, values, k, distances, labels, nullptr);
    }
    
    // Version with SearchParameters - different name to avoid overload ambiguity
    void search_with_params(idx_t n, const idx_t* indptr, const term_t* indices,
                            const float* values, int k, float* distances,
                            idx_t* labels,
                            nsparse::SearchParameters* search_parameters) {
        $self->search(n, indptr, indices, values, k, distances, labels, search_parameters);
    }
}

%include "nsparse/index.h"
%include "nsparse/brutal_index.h"
%include "nsparse/seismic_common.h"
%include "nsparse/seismic_index.h"
%include "nsparse/seismic_scalar_quantized_index.h"
%include "nsparse/inverted_index.h"
%include "nsparse/id_map_index.h"
%include "nsparse/cluster/random_kmeans.h"

// index_factory returns a new object - Python takes ownership
// Use %factory to enable proper downcasting based on runtime type
%newobject nsparse::index_factory;
%factory(nsparse::Index* nsparse::index_factory, nsparse::BrutalIndex, nsparse::SeismicIndex, nsparse::SeismicScalarQuantizedIndex, nsparse::IDMapIndex, nsparse::InvertedIndex);
%include "nsparse/index_factory.h"

// read_index returns a new object - Python takes ownership
%newobject nsparse::read_index;
%factory(nsparse::Index* nsparse::read_index, nsparse::BrutalIndex, nsparse::SeismicIndex, nsparse::SeismicScalarQuantizedIndex, nsparse::IDMapIndex, nsparse::InvertedIndex);
%include "nsparse/io/index_io.h"
