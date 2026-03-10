/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/index_factory.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "nsparse/brutal_index.h"
#include "nsparse/id_map_index.h"
#include "nsparse/index.h"
#include "nsparse/inverted_index.h"
#include "nsparse/seismic_common.h"
#include "nsparse/seismic_index.h"
#include "nsparse/seismic_scalar_quantized_index.h"

namespace nsparse {

namespace {

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = str.find(delimiter);
    while (end != std::string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + 1;
        end = str.find(delimiter, start);
    }
    tokens.push_back(str.substr(start));
    return tokens;
}

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return "";
    }
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

}  // namespace

// description is segmented by ,
// first segment is the index type: "brutal", "seismic", "seismic_sq"
// second segment is the parameters to construct index, parameters are separated
// by '|'
//
// Examples:
//   "brutal" - creates BrutalIndex with given dimension
//   "seismic,lambda=10|beta=5|alpha=0.5" - creates SeismicIndex
//   "seismic_sq,quantizer=8bit|vmin=0.0|vmax=1.0|lambda=10|beta=5|alpha=0.5"
Index* index_factory(int dimension, const char* description) {
    if (description == nullptr || std::strlen(description) == 0) {
        throw std::invalid_argument("Description cannot be null or empty");
    }

    std::string desc(description);
    auto segments = split(desc, ',');
    std::string index_type = trim(segments[0]);

    // Parse parameters into a map-like structure
    std::vector<std::pair<std::string, std::string>> params;
    if (segments.size() > 1) {
        auto param_tokens = split(segments[1], '|');
        for (const auto& token : param_tokens) {
            auto key_value = split(token, '=');
            if (key_value.size() == 2) {
                params.emplace_back(trim(key_value[0]), trim(key_value[1]));
            }
        }
    }

    // Helper to find parameter value
    auto get_param = [&params](const std::string& key,
                               const std::string& default_val) -> std::string {
        for (const auto& [param_key, param_val] : params) {
            if (param_key == key) {
                return param_val;
            }
        }
        return default_val;
    };

    if (index_type == "brutal") {
        return new BrutalIndex(dimension);
    }

    if (index_type == "inverted") {
        return new InvertedIndex(dimension);
    }

    if (index_type == "seismic") {
        int lambda = std::stoi(get_param("lambda", "10"));
        int beta = std::stoi(get_param("beta", "5"));
        float alpha = std::stof(get_param("alpha", "0.5"));
        return new SeismicIndex(dimension, {.lambda = lambda = lambda,
                                            .beta = beta = beta,
                                            .alpha = alpha = alpha});
    }

    if (index_type == "seismic_sq") {
        std::string quantizer_str = get_param("quantizer", "8bit");
        QuantizerType quantizer_type = QuantizerType::QT_8bit;
        if (quantizer_str == "16bit") {
            quantizer_type = QuantizerType::QT_16bit;
        }
        float vmin = std::stof(get_param("vmin", "0.0"));
        float vmax = std::stof(get_param("vmax", "1.0"));
        int lambda = std::stoi(get_param("lambda", "10"));
        int beta = std::stoi(get_param("beta", "5"));
        float alpha = std::stof(get_param("alpha", "0.5"));
        return new SeismicScalarQuantizedIndex(quantizer_type, vmin, vmax,
                                               {.lambda = lambda = lambda,
                                                .beta = beta = beta,
                                                .alpha = alpha = alpha},
                                               dimension);
    }

    if (index_type == "idmap") {
        // For idmap, the delegate index description follows after the first
        // comma Example: "idmap,seismic_sq,quantizer=8bit|lambda=10"
        if (segments.size() < 2) {
            throw std::invalid_argument("idmap requires a delegate index type");
        }
        // Reconstruct the delegate description from remaining segments
        std::string delegate_desc;
        for (size_t i = 1; i < segments.size(); ++i) {
            if (i > 1) {
                delegate_desc += ",";
            }
            delegate_desc += segments[i];
        }
        Index* delegate_index = index_factory(dimension, delegate_desc.c_str());
        return new IDMapIndex(delegate_index);
    }

    throw std::invalid_argument("Unknown index type: " + index_type);
}

}  // namespace nsparse