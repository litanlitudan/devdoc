# Model Explorer Toolbar Visibility Logic

**Date**: 2025-10-14
**Source**: `third_party/model-explorer/src/ui/src/components/visualizer/`
**Primary Files Analyzed**:
- `renderer_wrapper.ts` (lines 76-234)
- `renderer_wrapper.ng.html` (lines 30-201)
- `common/visualizer_config.ts` (lines 29-199)
- `common/types.ts` (ToolbarConfig interface)

---

## Overview

The Model Explorer UI implements a hierarchical visibility control system for its toolbar. The toolbar can be:
1. Completely hidden via global configuration
2. Displayed with selective item visibility based on context
3. Items can be disabled (visible but not interactive) based on application state

---

## Main Toolbar Visibility Control

### Primary Toggle

**Location**: `renderer_wrapper.ts:227-229`

```typescript
get showToolBar(): boolean {
  return !this.appService.config()?.hideToolBar;
}
```

**Logic**:
- ✅ **Toolbar Shown**: When `hideToolBar` is `false`, `undefined`, or not configured (default)
- ❌ **Toolbar Hidden**: When `hideToolBar` is explicitly set to `true`

**Template Usage**: `renderer_wrapper.ng.html:30`
```html
@if (showToolBar) {
  <div class="toolbar" ...>
    <!-- Toolbar content -->
  </div>
}
```

---

## Individual Toolbar Item Visibility

Each toolbar item has independent visibility logic. All items share a common pattern: they're hidden in popup mode, but have additional configuration-based controls.

### 1. Search Bar

**Location**: `renderer_wrapper.ts:181-183`

```typescript
get showSearchBar(): boolean {
  return !this.inPopup;
}
```

**Visibility Rules**:
- ✅ Shown in main window
- ❌ Hidden in popup windows

**Configuration**: No additional config options

**Template**: `renderer_wrapper.ng.html:34-40`

---

### 2. View on Node

**Location**: Always visible when toolbar is shown (no conditional logic)

**Template**: `renderer_wrapper.ng.html:43-47`

**Note**: This component has its own internal configuration via `viewOnNodeConfig`

---

### 3. Zoom to Fit

**Location**: Always visible when toolbar is shown (no conditional logic)

**Template**: `renderer_wrapper.ng.html:51-67`

**Functionality**: Triggers zoom-to-fit with SPACE keyboard shortcut

---

### 4. Expand/Collapse All Layers

**Location**: `renderer_wrapper.ts:185-191`

```typescript
get showExpandCollapseAllLayers(): boolean {
  return (
    !this.inPopup &&
    this.appService.config()?.toolbarConfig?.hideExpandCollapseAllLayers !== true
  );
}
```

**Visibility Rules**:
- ✅ Shown when NOT in popup AND config allows it
- ❌ Hidden when:
  - In popup mode, OR
  - `toolbarConfig.hideExpandCollapseAllLayers` is `true`

**Disabled State**: `renderer_wrapper.ts:219-221`

```typescript
get disableExpandCollapseAllButton(): boolean {
  return this.appService.getFlattenLayers(this.paneId);
}
```

The buttons are **disabled** (grayed out, not clickable) when "Flatten All Layers" mode is active.

**Configuration**:
```typescript
toolbarConfig: {
  hideExpandCollapseAllLayers?: boolean;
}
```

**Template**: `renderer_wrapper.ng.html:70-98`

---

### 5. Flatten All Layers

**Location**: `renderer_wrapper.ts:193-198`

```typescript
get showFlattenLayers(): boolean {
  return (
    !this.inPopup &&
    this.appService.config()?.toolbarConfig?.hideFlattenAllLayers !== true
  );
}
```

**Visibility Rules**:
- ✅ Shown when NOT in popup AND config allows it
- ❌ Hidden when:
  - In popup mode, OR
  - `toolbarConfig.hideFlattenAllLayers` is `true`

**Active State**: Uses `flattenAllLayers()` computed signal to show enabled/disabled visual state

**Configuration**:
```typescript
toolbarConfig: {
  hideFlattenAllLayers?: boolean;
}
```

**Template**: `renderer_wrapper.ng.html:101-119`

---

### 6. Trace I/O

**Location**: Always visible when toolbar is shown (no conditional logic)

**Active State**: `renderer_wrapper.ts:223-225`

```typescript
get tracing(): boolean {
  return this.webglRenderer?.tracing === true;
}
```

Toggles visual highlight for input/output tracing on selected nodes.

**Template**: `renderer_wrapper.ng.html:122-143`

---

### 7. Edge Overlays Dropdown

**Location**: `renderer_wrapper.ts:212-217`

```typescript
get showEdgeOverlaysDropdown(): boolean {
  return (
    !this.inPopup &&
    this.appService.config()?.toolbarConfig?.hideCustomEdgeOverlays !== true
  );
}
```

**Visibility Rules**:
- ✅ Shown when NOT in popup AND config allows it
- ❌ Hidden when:
  - In popup mode, OR
  - `toolbarConfig.hideCustomEdgeOverlays` is `true`

**Configuration**:
```typescript
toolbarConfig: {
  hideCustomEdgeOverlays?: boolean;
}
```

**Template**: `renderer_wrapper.ng.html:146-151`

---

### 8. Download as PNG

**Location**: `renderer_wrapper.ts:200-202`

```typescript
get showDownloadPng(): boolean {
  return !this.inPopup;
}
```

**Visibility Rules**:
- ✅ Shown in main window
- ❌ Hidden in popup windows

**Configuration**: No additional config options

**Template**: `renderer_wrapper.ng.html:154-191`

**Features**:
- Downloads current viewport or full graph
- Optional transparent background toggle

---

### 9. Snapshot Manager

**Location**: `renderer_wrapper.ts:204-206`

```typescript
get showSnapshotManager(): boolean {
  return !this.inPopup;
}
```

**Visibility Rules**:
- ✅ Shown in main window
- ❌ Hidden in popup windows

**Configuration**: No additional config options

**Template**: `renderer_wrapper.ng.html:194-200`

---

## Configuration Schema

### VisualizerConfig Structure

**Location**: `common/visualizer_config.ts:30-199`

```typescript
interface VisualizerConfig {
  /**
   * Whether to hide the tool bar.
   */
  hideToolBar?: boolean;

  /**
   * Config for the toolbar.
   */
  toolbarConfig?: ToolbarConfig;

  // ... other config options
}
```

### ToolbarConfig Structure

**Location**: `common/types.ts`

```typescript
interface ToolbarConfig {
  /** Whether to hide the "Expand/collapse all layers" button. */
  hideExpandCollapseAllLayers?: boolean;

  /** Whether to hide the "Flatten all layers" button. */
  hideFlattenAllLayers?: boolean;

  /** Whether to hide the "Custom edge overlays" button. */
  hideCustomEdgeOverlays?: boolean;
}
```

---

## Popup Mode Behavior

The `inPopup` input property controls special behavior for popup windows:

**Input Property**: `renderer_wrapper.ts:82`
```typescript
@Input() inPopup = false;
```

**Effects on Toolbar Items**:
| Item | Hidden in Popup? |
|------|------------------|
| Search Bar | ✅ Yes |
| View on Node | ❌ No |
| Zoom to Fit | ❌ No |
| Expand/Collapse All | ✅ Yes |
| Flatten Layers | ✅ Yes |
| Trace I/O | ❌ No |
| Edge Overlays | ✅ Yes |
| Download PNG | ✅ Yes |
| Snapshot Manager | ✅ Yes |

**Rationale**: Popup windows show minimal toolbar features focused on navigation and viewing, while hiding features that affect the main application state.

---

## Visual States Summary

### Three-Level State System

1. **Hidden** - Element not rendered in DOM
   - Controlled by visibility getter methods
   - Cannot be interacted with

2. **Disabled** - Element rendered but non-interactive
   - Example: Expand/Collapse buttons when flatten mode is active
   - Visual indication via CSS classes (`.disable`)

3. **Active/Enabled** - Element rendered and interactive
   - Example: Flatten layers button showing enabled state
   - Visual indication via CSS classes (`.enabled`)

---

## Implementation Pattern

All visibility controls follow this pattern:

```typescript
// 1. Getter method computes visibility
get showFeature(): boolean {
  return !this.inPopup &&
    this.appService.config()?.toolbarConfig?.hideFeature !== true;
}

// 2. Template uses @if directive
@if (showFeature) {
  <div class="feature">...</div>
}

// 3. Optional disabled state
get disableFeature(): boolean {
  return this.appService.getSomeState();
}

// 4. CSS class binding
<div [class.disable]="disableFeature">
```

---

## Integration Points

### AppService

The toolbar queries configuration through `appService.config()`:

```typescript
private readonly appService: AppService
```

**Key Methods Used**:
- `config()` - Returns current VisualizerConfig
- `getFlattenLayers(paneId)` - Returns flatten layers state
- Various event subjects for toolbar actions

### WebGL Renderer

Toolbar actions often delegate to the WebGL renderer:

```typescript
@ViewChild('webglRenderer') webglRenderer?: WebglRenderer;
```

**Example**: `renderer_wrapper.ts:164-166`
```typescript
handleClickTrace() {
  this.webglRenderer?.toggleIoTrace();
}
```

---

## Usage Examples

### Example 1: Hide Entire Toolbar

```typescript
const config: VisualizerConfig = {
  hideToolBar: true
};
```

### Example 2: Show Toolbar, Hide Specific Items

```typescript
const config: VisualizerConfig = {
  hideToolBar: false,  // or omit (defaults to false)
  toolbarConfig: {
    hideExpandCollapseAllLayers: true,
    hideFlattenAllLayers: true,
    hideCustomEdgeOverlays: false
  }
};
```

### Example 3: Minimal Toolbar (Popup-like Behavior)

Set `inPopup = true` as component input:

```html
<renderer-wrapper
  [inPopup]="true"
  ...>
</renderer-wrapper>
```

This hides: Search, Expand/Collapse, Flatten, Edge Overlays, Download, Snapshot Manager

---

## Decision Tree

```
Is hideToolBar = true?
├─ YES → Hide entire toolbar
└─ NO → Show toolbar container
    │
    ├─ Is inPopup = true?
    │  ├─ YES → Hide most items, show only: View, Zoom, Trace
    │  └─ NO → Continue to item-specific checks
    │
    └─ For each item:
        │
        ├─ Is toolbarConfig.hideItem = true?
        │  ├─ YES → Hide item
        │  └─ NO → Show item
        │
        └─ Check disabled state (if applicable)
           ├─ Flatten mode active? → Disable Expand/Collapse
           └─ Other state checks...
```

---

## Key Findings

1. **Two-Level Configuration**: Global toolbar visibility + individual item controls
2. **Popup Mode Override**: Single flag that hides most advanced features
3. **State-Based Disabling**: Some items can be disabled without being hidden
4. **Default Behavior**: Everything shown unless explicitly hidden
5. **No Dynamic Dependencies**: Item visibility is independent (no cascading hide effects)
6. **Change Detection**: Uses `OnPush` strategy with manual `markForCheck()` calls

---

## Related Files

- **Component Logic**: `src/ui/src/components/visualizer/renderer_wrapper.ts`
- **Template**: `src/ui/src/components/visualizer/renderer_wrapper.ng.html`
- **Styles**: `src/ui/src/components/visualizer/renderer_wrapper.scss`
- **Configuration Types**: `src/ui/src/components/visualizer/common/visualizer_config.ts`
- **Type Definitions**: `src/ui/src/components/visualizer/common/types.ts`
- **App Service**: `src/ui/src/components/visualizer/app_service.ts`

---

## Notes for Integration

When integrating Model Explorer into Markserv:

1. **Default Configuration**: If no config is provided, toolbar will show with all items visible
2. **Minimal UI Mode**: Set `hideToolBar: true` for document-focused view
3. **Selective Hiding**: Use `toolbarConfig` to customize which features are available
4. **Popup Behavior**: Consider using `inPopup=true` for embedded/iframe scenarios
5. **State Management**: Monitor `appService` for toolbar-related state changes

---

## Appendix: Complete Visibility Matrix

| Toolbar Item | Default | inPopup=true | hideToolBar=true | Config Override |
|--------------|---------|--------------|------------------|-----------------|
| Entire Toolbar | Visible | Visible | Hidden | `hideToolBar` |
| Search Bar | Visible | Hidden | N/A | None |
| View on Node | Visible | Visible | N/A | `viewOnNodeConfig` |
| Zoom to Fit | Visible | Visible | N/A | None |
| Expand/Collapse All | Visible | Hidden | N/A | `hideExpandCollapseAllLayers` |
| Flatten Layers | Visible | Hidden | N/A | `hideFlattenAllLayers` |
| Trace I/O | Visible | Visible | N/A | None |
| Edge Overlays | Visible | Hidden | N/A | `hideCustomEdgeOverlays` |
| Download PNG | Visible | Hidden | N/A | None |
| Snapshot Manager | Visible | Hidden | N/A | None |

---

**Document Version**: 1.0
**Last Updated**: 2025-10-14
