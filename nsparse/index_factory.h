/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef INDEX_FACTORY_H
#define INDEX_FACTORY_H

#include "nsparse/index.h"
namespace nsparse {

Index* index_factory(int dimension, const char* description);
}

#endif  // INDEX_FACTORY_H