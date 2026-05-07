// ─── Global Space Theme ────────────────────────────────────────────────────
export const Colors = {
  background: '#02030a',
  surface: '#080d1a',
  surfaceLight: '#0d1530',
  border: '#1a2540',
  text: '#ccdff5',
  textMuted: '#6a8aaa',
  accent: '#00ffff',
  accentDim: '#005f5f',
  tabActive: '#00ffff',
  tabInactive: '#3a5a7a',
  tabBar: '#060c18',
  white: '#ffffff',
  danger: '#ff4d6d',
  success: '#00ffaa',
} as const;

export const FontFamily = {
  regular: 'SpaceMono_400Regular',
  bold: 'SpaceMono_700Bold',
} as const;

export const Spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
} as const;

export const BorderRadius = {
  sm: 6,
  md: 12,
  lg: 20,
  full: 9999,
} as const;
