<img src="https://opensearch.org/assets/img/opensearch-logo-themed.svg" height="64px">

- [Introduction](#neural-sparse-cpp)
- [Project Resources](#project-resources)
- [Project Style Guidelines](#project-style-guidelines)
- [Code of Conduct](#code-of-conduct)
- [License](#license)
- [Copyright](#copyright)

## neural-sparse-cpp

**neural-sparse-cpp** is a C++ library for high-performance sparse vector similarity search, developed as part of the [OpenSearch Project](https://opensearch.org/). It provides multiple index types for nearest neighbor search over sparse vectors.

Key features include:

- Multiple index types: inverted index, SEISMIC, and SEISMIC with scalar quantization
- Scalar quantization support for reduced memory usage
- SIMD-optimized distance computations (AVX2, AVX512, NEON, SVE)
- ID mapping and ID selector filtering
- Index serialization (save/load)
- Python bindings via SWIG

The library is licensed under the [Apache v2.0 License](LICENSE.txt) and is part of the [OpenSearch Software Foundation](https://foundation.opensearch.org/), a project of [The Linux Foundation](https://www.linuxfoundation.org/).

## Project Resources

* [Project Website](https://opensearch.org/)
* [OpenSearch Software Foundation](https://foundation.opensearch.org/)
* [Downloads](https://opensearch.org/downloads.html)
* [Documentation](https://opensearch.org/docs/latest/)
* Need help? Try [Forums](https://forum.opensearch.org/)
* Talk to other developers? [Slack](https://opensearch.org/slack.html)
* [Communications](https://github.com/opensearch-project/community/blob/main/COMMUNICATIONS.md)
* [Project Principles](https://opensearch.org/about.html#principles-for-development)
* [Contributing to OpenSearch](CONTRIBUTING.md)
* [Proposing Features](FEATURES.md)
* [Onboarding Guide](ONBOARDING.md)
* [Maintainer Responsibilities](RESPONSIBILITIES.md)
* [Release Management](RELEASING.md)
* [Organization Admins](ADMINS.md)
* [Repo Maintainers](MAINTAINERS.md)
* [Issue Triage](TRIAGING.md)
* [Security](SECURITY.md)

## Project Style Guidelines

The [OpenSearch Project style guidelines](https://github.com/opensearch-project/documentation-website/blob/main/STYLE_GUIDE.md) and [OpenSearch terms](https://github.com/opensearch-project/documentation-website/blob/main/TERMS.md) documents provide style standards and terminology to be observed when creating OpenSearch Project content.

## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](CODE_OF_CONDUCT.md). For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq), or contact [conduct@opensearch.foundation](mailto:conduct@opensearch.foundation) with any additional questions or comments.

## License

This project is licensed under the [Apache v2.0 License](LICENSE.txt).

## Copyright

Copyright OpenSearch Contributors. See [NOTICE](NOTICE.txt) for details.
