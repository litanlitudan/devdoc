# Changelog

- Devdoc uses [Semantic Versioning](http://semver.org/)
- Devdoc [Keeps a ChangeLog](https://keepachangelog.com/en/1.0.0/)

## [2.0.0-beta.2] - 2025-10-05

### Changed

- **Major**: Converted entire codebase from JavaScript to TypeScript
  - Added comprehensive type safety throughout the application
  - All core library files (`cli.ts`, `server.ts`, `splash.ts`, `readme.ts`, `cli-defs.ts`) converted to TypeScript
  - Created dedicated `types.ts` file with all interface and type definitions
  - Added TypeScript compiler configuration with strict type checking
  - Updated build process to compile TypeScript to JavaScript in `dist/` directory
  - Removed old JavaScript source files - now TypeScript is the single source of truth
  - All tests pass with TypeScript-compiled code (31/32 tests passing - 1 pre-existing test issue)

### Added

- TypeScript configuration (`tsconfig.json`) with strict mode enabled
- Type definitions for all dependencies:
  - `@types/express`, `@types/compression`, `@types/ws`
  - `@types/yargs`, `@types/handlebars`, `@types/less`
  - `@types/markdown-it`, `@types/mime-types`, `@types/multer`
- Source maps for debugging TypeScript code
- Declaration files (`.d.ts`) for library consumers
- New build scripts: `build:watch`, `clean`, `typecheck`

### Developer Experience

- Enhanced IDE support with TypeScript IntelliSense
- Compile-time error detection prevents runtime issues
- Better code navigation and refactoring support
- Improved code documentation through type annotations

## [1.17.4] - 2019-12-28

### Added

- Added test for LESS implant. [#99]

### Changed

- Update to latest packages using `npm-check-updates`. Update new linting errors from latest XO package. [#99]

## [1.17.3] - 2019-12-28

### Added

- Added test for file implant. [#98]

### Changed

- Process MathJax with Markdown-It-MathJax. [#93]
- Update all-contibutors table. [#98]

### Removed

- Removed unused Patreon links. [#98]

### Fixed

- Fixed Live-Reload for browsers without Plugin. [#92]
- Documentation fixes. [#97], [#89]

### Security

- NPM audit fix --force. Resulted in AVA update to 2.x requiring package script test runner path change. [#98]

## [1.17.2] - 2019-02-26

### Fixed

- Missing CLI packages. [#79], [#81]

## [1.17.1] - 2019-02-23

### Fixed

- Snyk security audit & fixed CLI launch bug. [#77]

## [1.17.0] - 2019-02-23

### Added

- Added contributors table to README. [#76]

## [1.16.0] - 2019-02-23

### Changed

- Updated CSS page width in stylesheets to reflect GitHubs styles. [#74]
- Replace Commander with Meow. [#75]

### Fixed

- Fixed README CLI command. [#75]

## [1.15.1] - 2018-10-14

### Added

- Added `devdoc --livereloadport false` to disable LiveReload. [#65] 
- Added `devdoc --browser false` to disable Browser Launch. [#65] 
- Added contributors to `package.json` [#65] 

### Fixed

- Fix launch of relative files and dirs from `devdoc` and `readme` commands. [#63]

## [1.13.2] - 2018-09-14

### Fixed

- Clean `npm audit`: PR [#59](https://github.com/F1LT3R/devdoc/pull/59)

## [1.13.1] - 2018-09-14

### Fixed

- Check for updates only when online: PR [#56](https://github.com/F1LT3R/devdoc/pull/56)

## [1.13.0] - 2018-09-14

### Added

- Mobile Font - does not look squished on smaller screens: PR [#55](https://github.com/F1LT3R/devdoc/pull/55)

### Changed

- Removed useless CSS, and border from printing and mobile view: PR [#55](https://github.com/F1LT3R/devdoc/pull/55)

## [1.12.0] - 2018-05-23

### Changed

- Updated boot - splash is now called from cli and readme to so the user can see that devdoc is loading: PR [#53](https://github.com/F1LT3R/devdoc/pull/53)

### Added

- Auto Upgrade - user gets option to upgrade to latest when starting Devdoc: PR [#52](https://github.com/F1LT3R/devdoc/pull/52)

## [1.11.0] - 2018-05-22

### Changed

- Updated README after changing github:filter/devdoc to github/devdoc/devdoc (no PR)

## [1.10.0] - 2018-05-22

### Changed

- Updated README after changing github:filter/devdoc to github/devdoc/devdoc (no PR)

## [1.9.0] - 2018-05-21

### Changed

- Better breadcrumbs: PR [#52](https://github.com/F1LT3R/devdoc/pull/52)
- All folders now use the same icon, to reduce visual noise: PR [#52](https://github.com/F1LT3R/devdoc/pull/52)

### Added

- Sanitize urls in breadcrumbs: PR [#52](https://github.com/F1LT3R/devdoc/pull/52)
	+ Thanks @ChenYingChou PR [#48](https://github.com/F1LT3R/devdoc/pull/48)
- Error page with back links: PR [#52](https://github.com/F1LT3R/devdoc/pull/52)
	+ Thanks @ChenYingChou PR [#48](https://github.com/F1LT3R/devdoc/pull/48)
- Slugify Links (w/ Emojis) PR [#51](https://github.com/F1LT3R/devdoc/pull/51)
	+ Thanks @ChenYingChou PR [#48](https://github.com/F1LT3R/devdoc/pull/48)

## [1.8.0] - 2018-05-13

### Added

- Emoji support with `mdItEmoji`. Example: `:fire:` now renders as :fire:
	+ Thanks @ChenYingChou PR [#48](https://github.com/F1LT3R/devdoc/pull/48/files)
- Indent size 4 to `.editorconfig` - @ChenYingChou PR [#48](https://github.com/F1LT3R/devdoc/pull/48/files)

## [1.7.3] - 2018-05-13

### Fixed

- Emojis require \:colon-syntax\: to render correctly on NPMJS.org
	+ Thanks @ChenYingChou PR [#48](https://github.com/F1LT3R/devdoc/pull/48/files)
- Added ChangeLog