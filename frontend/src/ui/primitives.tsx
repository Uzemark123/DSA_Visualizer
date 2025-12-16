import React from 'react'

/* =========================
   UI PRIMITIVES
   ========================= */

export const Panel: React.FC<
  React.PropsWithChildren<{
    className?: string
  }>
> = ({ children, className = '' }) => (
  <div className={`bg-panel/90 backdrop-blur border border-white/10 shadow-float ${className}`}>
    {children}
  </div>
)

export const Button: React.FC<
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: 'default' | 'destructive'
  }
> = ({ className = '', variant = 'default', ...props }) => {
  const base = 'px-3 py-1 rounded-full text-xs transition border shadow-sm'
  const variants = {
    default: 'bg-card/80 text-slate-200 border-white/10 hover:bg-card',
    destructive: 'bg-red-500/10 text-red-100 border-red-400/30 hover:bg-red-500/20',
  }
  return <button {...props} className={`${base} ${variants[variant]} ${className}`} />
}

export const IconButton: React.FC<React.ButtonHTMLAttributes<HTMLButtonElement>> = ({
  className = '',
  ...props
}) => (
  <button
    {...props}
    className={`w-10 h-10 rounded-lg bg-card/80 hover:bg-card transition text-xs text-slate-200 border border-white/5 ${className}`}
  />
)

/* =========================
   LAYOUT COMPONENTS
   ========================= */

export const AppShell: React.FC<React.PropsWithChildren> = ({ children }) => (
  <div className="relative min-h-screen bg-ink text-slate-100 overflow-hidden">{children}</div>
)

export const TopBar: React.FC = () => (
  <div className="fixed z-30 top-6 left-1/2 -translate-x-1/2">
    <Panel className="flex items-center gap-3 px-4 py-2 rounded-full">
      <div className="flex items-center gap-2 text-sm text-slate-200">
        <span className="w-2 h-2 rounded-full bg-accent-teal" />
        <span className="font-semibold">DSA Visual Canvas</span>
      </div>
      <div className="flex items-center gap-2 text-slate-300 text-sm">
        <Button>Home</Button>
        <Button>Library</Button>
        <Button>Share</Button>
      </div>
    </Panel>
  </div>
)

export const RightToolbar: React.FC = () => (
  <div className="fixed z-30 right-6 top-28">
    <Panel className="rounded-xl2 px-2 py-3 flex flex-col gap-2">
      {['Pan', 'Select', 'Add', 'Color', 'Note', 'Connect'].map((label) => (
        <IconButton key={label} title={label}>
          {label[0]}
        </IconButton>
      ))}
    </Panel>
  </div>
)

/* =========================
   DRAWER
   ========================= */

export const DrawerToggleButton: React.FC<{
  open: boolean
  onToggle: () => void
}> = ({ open, onToggle }) => (
  <Button onClick={onToggle} className="px-3 py-2 text-sm rounded-xl2">
    {open ? 'Close library' : 'Open library'}
  </Button>
)

export const LeftDrawer: React.FC<
  React.PropsWithChildren<{
    open: boolean
    onToggle: () => void
    title?: string
    subtitle?: string
  }>
> = ({ open, onToggle, title, subtitle, children }) => (
  <div className="fixed z-30 top-24 left-6">
    <DrawerToggleButton open={open} onToggle={onToggle} />

    <Panel
      className={`mt-3 w-72 rounded-2xl transition-all duration-300 origin-left
        ${
          open
            ? 'opacity-100 translate-x-0 scale-100'
            : 'opacity-0 -translate-x-4 scale-95 pointer-events-none'
        }`}
    >
      {(title || subtitle) && (
        <div className="px-4 py-3 border-b border-white/5">
          {title && <p className="text-sm font-semibold text-slate-100">{title}</p>}
          {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
        </div>
      )}
      <div className="px-4 py-3">{children}</div>
    </Panel>
  </div>
)

/* =========================
   CONTEXT PALETTE
   ========================= */

export const PaletteColorSwatch: React.FC<{
  color: string
  onClick?: () => void
}> = ({ color, onClick }) => (
  <button
    onClick={onClick}
    className="w-8 h-8 rounded-full border border-white/15 hover:scale-105 transition"
    style={{ backgroundColor: color }}
  />
)

export const ContextPalette: React.FC<{
  x: number
  y: number
  children: React.ReactNode
}> = ({ x, y, children }) => (
  <div
    className="fixed z-40"
    style={{
      left: x,
      top: y,
      transform: 'translate(-50%, -100%)',
    }}
  >
    <Panel className="rounded-xl2 px-3 py-2 w-64 flex flex-col gap-2">{children}</Panel>
  </div>
)

/* =========================
   EXPORTS
   ========================= */

export default {
  AppShell,
  Panel,
  Button,
  IconButton,
  TopBar,
  RightToolbar,
  LeftDrawer,
  DrawerToggleButton,
  ContextPalette,
  PaletteColorSwatch,
}
