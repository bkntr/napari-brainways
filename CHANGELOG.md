# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.10]
### Added
 - First use setup
 - Download registration model from huggingface
 - Annotation Viewer controller: add nice text overlay showing the structure under mouse cursor
 - Add sample data

### Fixed
 - Analysis widget: ANOVA shows F value instead of incorrect p value
 - Analysis widget: Show ANOVA button now works
 - Cell Viewer controller: fix cell colors

## [0.1.9]
### Added
 - Network graph analysis

### Changed
 - Use napari's built-in progress bar
 - Contrast analysis/PLS analysis run as async to keep GUI alive
 - Use napari's new text overlay in analysis step

### Fixed
 - Fix tensorflow version for StarDist
 - Fix project unable to save after elastix

## [0.1.8.3]

### Fixed
 - Fix cell detection bug in brainways

## [0.1.8.2]

### Fixed
 - Fix new project crashes on cell detector

## [0.1.8]

### Added
 - PLS analysis
 - Cell detection: added layer to display both high-res crop preview and normalized crop preview

### Fixed
 - Fix installation error by adding napari[all] to project requirements.

## [0.1.7]

### TBD

<!-- [unreleased]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.1.1...HEAD -->
[0.1.10]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.1.8.3...v0.1.9
[0.1.8.3]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.1.8.2...v0.1.8.3
[0.1.8.2]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.1.8...v0.1.8.2
[0.1.8]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.1.6...v0.1.7
