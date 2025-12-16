# Frontend plan

## Structure
- `index.html`: bootstraps the SPA root at `#root` and loads `/src/main.tsx`.
- `vite.config.ts`: Vite config with the React plugin.
- `src/main.tsx`: React entry point that renders `<App />`.
- `src/App.tsx`: Excalidraw canvas shell plus surrounding React UI (menus, metadata, persistence hooks).
- `src/index.css`: Tailwind base + global tokens for the non-canvas UI.
- `src/vite-env.d.ts`: Vite type helpers for TS.

## Core tools
- Vite for fast dev server/build.
- React 18 + TypeScript for the SPA shell.
- Excalidraw (embedded component, not forked) as the canvas engine.
- Tailwind CSS for non-canvas UI (menus, panels, metadata overlays).

## Canvas source of truth
- Excalidraw owns all canvas interactions (selection, linking, pan/zoom, tool modes). The React app observes changes via `onChange` and mirrored state to drive UI labels, menus, and persistence.
- Imperative hooks (via `ExcalidrawImperativeAPI` ref) trigger canvas-native actions such as zoom-to-fit, resetting the scene, or reading the current selection—avoiding duplicate interaction logic outside Excalidraw.
- Persistence flows should read/write Excalidraw data structures (`elements`, `appState`, `files`) directly: serialize them for storage, and restore through the `initialData` prop.
- Surrounding React components should treat Excalidraw as the single source of truth for canvas state, only augmenting it with metadata (e.g., titles, tags, graph notes) and controls that live outside the drawing surface.
