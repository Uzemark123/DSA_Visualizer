# Frontend (React + Excalidraw)

The frontend is a Vite/React application that embeds Excalidraw as the canvas engine and layers project-specific UI around it (roadmaps, progress state, help/legend overlays, minimap, etc.).

## Canvas-based UI architecture
- `frontend/src/App.tsx` hosts the Excalidraw component and orchestrates all UI overlays.
- The canvas scene is loaded from `frontend/public/roadmap.excalidraw`.
- Excalidraw remains the source of truth for canvas interactions; the app listens to `onChange` to derive selection, progress, and minimap state.
- Screen-space UI (top bar, minimap, panels) is rendered outside the canvas subtree to avoid inheriting zoom/pan transforms.

## Node metadata and interactions
- Pattern nodes are identified via a custom link scheme: `dsa://node/{patternId}[?dis=...]`.
- `frontend/src/roadmap/link_parsing.ts` parses these links and populates `customData.patternId` and optional `disambiguation`.
- Progress state is stored on pattern nodes in `customData.progressState` and mirrored into localStorage for persistence.
- The label shown in selectors is derived from bound text elements and optional disambiguation.

## Mapping backend output into frontend state
- `frontend/src/roadmap/collectPatterns.ts` gathers selected/in-progress patterns and builds the API payload.
- API responses (roadmap nodes with chunks + revision problems) are flattened into a list of problems for the “My Roadmaps” panel in `App.tsx`.
- Problem URLs from the backend are displayed directly in the roadmap list.

## Generic vs project-specific code
Generic (Excalidraw integration):
- Canvas host, `onChange` listener, and minimap rendering.
- Selection, pan, zoom, and base interaction handling.

Project-specific:
- Parsing `dsa://node/*` links into `customData` metadata.
- Progress state rules and roadmap generation UI.
- Roadmap response shaping and “My Roadmaps” display.

## Run the frontend
- `cd frontend`
- `npm install`
- `npm run dev`

The Vite dev server proxies `/api/*` to the backend (see `frontend/vite.config.ts`).
