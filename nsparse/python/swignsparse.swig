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
#include <new>
#include <stdexcept>
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
#include "nsparse/disk_seismic_index.h"
#include "nsparse/disk_seismic_scalar_quantized_index.h"
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

%include <std_string.i>
%include <exception.i>

// SWIG generates no try/catch of its own, so without this every exception that
// escapes the library reaches std::terminate and kills the interpreter -- an
// unknown index spec or a non-positive k would abort the process instead of
// raising. Declared before any wrapped decl so it applies to all of them.
%exception {
    try {
        $action
    } catch (const std::invalid_argument& e) {
        SWIG_exception(SWIG_ValueError, e.what());
    } catch (const std::out_of_range& e) {
        SWIG_exception(SWIG_IndexError, e.what());
    } catch (const std::bad_alloc& e) {
        SWIG_exception(SWIG_MemoryError, e.what());
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}

%include "nsparse_typemaps.i"
%include "factory.i"

// Everything in nsparse::detail is internal. This has to precede every header
// %include below: %ignore only applies to declarations parsed after it, so
// placing it lower silently missed the detail symbols in the headers already
// pulled in -- INVALID_IDX among them, which then surfaced as a useless
// int32_t* pointer object.
%ignore nsparse::detail;
%include "nsparse/io/io.h"

%include "nsparse/types.h"

// read_csr(path, Residency::kMmap) only accepts the native layout, so without a
// way to produce one the mapped residency is unreachable from Python. convert()
// is also where an out-of-range term is rejected; the mapped reader trusts its
// input and does not rescan.
%include "nsparse/utils/csr_layout.h"
// The mmap plumbing stays internal, and %ignore only takes effect ahead of the
// %include that pulls the declarations in. mmap_deserialize takes an MmapCursor,
// a type this interface does not wrap, so leaving it in makes SWIG emit the
// parameter unqualified and the generated wrapper stops compiling. map_vectors
// returns an object borrowing from a mapping the caller must outlive -- a
// contract Python's refcounting cannot express. Mapping is reached from Python
// through read_index(file, kUseMmap), which owns the mapping it hands back.
%ignore nsparse::SparseVectors::mmap_deserialize;
%ignore nsparse::SparseVectors::map_vectors;
%include "nsparse/sparse_vectors.h"
// Same reasons, for the posting lists: mmap_deserialize and map take an
// MmapCursor, and read() hands back lists Python has no use for on their own.
// Serialization is reached through write_index/read_index, never a list at a
// time. get_doc_ids/get_codes hand out a Buf, which is both a borrow Python
// cannot keep alive and a template this interface does not wrap -- left in,
// SWIG emits it unqualified and the generated wrapper stops compiling.
%ignore nsparse::InvertedList::get_doc_ids;
%ignore nsparse::InvertedList::get_codes;
%ignore nsparse::InvertedList::mmap_deserialize;
%ignore nsparse::InvertedList::serialize;
%ignore nsparse::InvertedList::deserialize;
%ignore nsparse::ArrayInvertedLists::map;
%ignore nsparse::ArrayInvertedLists::read;
%ignore nsparse::ArrayInvertedLists::serialize;
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

// Hands back an index borrowing from a mapping it does not own; see the
// SparseVectors mmap ignores above.
%ignore nsparse::SeismicIndex::mmap_index;
%ignore nsparse::SeismicScalarQuantizedIndex::mmap_index;
%ignore nsparse::InvertedIndex::mmap_index;
%ignore nsparse::DiskSeismicIndex::mmap_index;
%ignore nsparse::DiskSeismicScalarQuantizedIndex::mmap_index;
%ignore nsparse::InvertedListClusters::serialize;
%ignore nsparse::InvertedListClusters::deserialize;

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

// The seismic indexes and InvertedIndex derive from MmapIndex, so SWIG has to be
// told that class exists: without it the base is dropped and the Python classes
// silently lose everything Index declares -- search, num_vectors, read_csr.
%include "nsparse/mmap_index.h"

%include "nsparse/brutal_index.h"
%include "nsparse/seismic_common.h"
%include "nsparse/seismic_index.h"
%include "nsparse/seismic_scalar_quantized_index.h"
%include "nsparse/disk_seismic_index.h"
%include "nsparse/disk_seismic_scalar_quantized_index.h"
%include "nsparse/inverted_index.h"
%include "nsparse/id_map_index.h"
%include "nsparse/cluster/random_kmeans.h"

// index_factory returns a new object - Python takes ownership
// Use %factory to enable proper downcasting based on runtime type
%newobject nsparse::index_factory;
%factory(nsparse::Index* nsparse::index_factory, nsparse::BrutalIndex, nsparse::SeismicIndex, nsparse::SeismicScalarQuantizedIndex, nsparse::DiskSeismicScalarQuantizedIndex, nsparse::IDMapIndex, nsparse::InvertedIndex);
%include "nsparse/index_factory.h"

// read_index returns a new object - Python takes ownership
%newobject nsparse::read_index;
%factory(nsparse::Index* nsparse::read_index, nsparse::BrutalIndex, nsparse::SeismicIndex, nsparse::SeismicScalarQuantizedIndex, nsparse::DiskSeismicScalarQuantizedIndex, nsparse::IDMapIndex, nsparse::InvertedIndex);
%include "nsparse/io/index_io.h"
