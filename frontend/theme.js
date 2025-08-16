/**
 * MUI Theme Configuration with Semantic Tokens
 * Brand Colors: Primary #1E3A8A, Secondary #06B6D4
 * Implements light/dark modes with WCAG AA compliance
 */

// Brand color definitions
const brandColors = {
  primary: '#1E3A8A',      // Deep blue
  secondary: '#06B6D4',    // Cyan
  // Calculated shades for variations
  primaryLight: '#2E4EAE',
  primaryDark: '#152C66',
  secondaryLight: '#22D3EE',
  secondaryDark: '#0891B2',
};

// Neutral colors for UI elements
const neutralColors = {
  white: '#FFFFFF',
  black: '#000000',
  gray: {
    50: '#F9FAFB',
    100: '#F3F4F6',
    200: '#E5E7EB',
    300: '#D1D5DB',
    400: '#9CA3AF',
    500: '#6B7280',
    600: '#4B5563',
    700: '#374151',
    800: '#1F2937',
    900: '#111827',
  }
};

// Light theme palette
const lightPalette = {
  mode: 'light',
  // Primary colors (10% usage max for solid colors)
  primary: {
    main: brandColors.primary,
    light: brandColors.primaryLight,
    dark: brandColors.primaryDark,
    contrastText: neutralColors.white,
  },
  secondary: {
    main: brandColors.secondary,
    light: brandColors.secondaryLight,
    dark: brandColors.secondaryDark,
    contrastText: neutralColors.white,
  },
  // Background colors (90% of layout)
  background: {
    default: neutralColors.white,
    paper: neutralColors.gray[50],
    elevated: neutralColors.white,
    surface: neutralColors.gray[100],
  },
  // Text colors with WCAG compliance
  text: {
    primary: neutralColors.gray[900],      // Contrast ratio > 7:1 on white
    secondary: neutralColors.gray[700],    // Contrast ratio > 4.5:1 on white
    disabled: neutralColors.gray[400],
    hint: neutralColors.gray[500],
  },
  // UI element colors
  action: {
    active: brandColors.primary,
    hover: `${brandColors.primary}14`,     // 8% opacity
    hoverOpacity: 0.08,
    selected: `${brandColors.primary}1F`,  // 12% opacity
    selectedOpacity: 0.12,
    disabled: neutralColors.gray[300],
    disabledBackground: neutralColors.gray[100],
    disabledOpacity: 0.38,
    focus: `${brandColors.primary}29`,     // 16% opacity
    focusOpacity: 0.16,
  },
  // Divider colors
  divider: neutralColors.gray[200],
  // Status colors
  error: {
    main: '#DC2626',
    light: '#EF4444',
    dark: '#B91C1C',
    contrastText: neutralColors.white,
  },
  warning: {
    main: '#F59E0B',
    light: '#FBBf24',
    dark: '#D97706',
    contrastText: neutralColors.gray[900],
  },
  info: {
    main: brandColors.secondary,
    light: brandColors.secondaryLight,
    dark: brandColors.secondaryDark,
    contrastText: neutralColors.white,
  },
  success: {
    main: '#10B981',
    light: '#34D399',
    dark: '#059669',
    contrastText: neutralColors.white,
  },
};

// Dark theme palette
const darkPalette = {
  mode: 'dark',
  // Primary colors adjusted for dark mode
  primary: {
    main: brandColors.secondaryLight,      // Brighter for dark backgrounds
    light: '#67E8F9',
    dark: brandColors.secondary,
    contrastText: neutralColors.gray[900],
  },
  secondary: {
    main: brandColors.primaryLight,        // Lighter blue for dark mode
    light: '#4361C2',
    dark: brandColors.primary,
    contrastText: neutralColors.white,
  },
  // Dark backgrounds
  background: {
    default: neutralColors.gray[900],
    paper: neutralColors.gray[800],
    elevated: '#0F172A',
    surface: neutralColors.gray[800],
  },
  // Text colors for dark mode with WCAG compliance
  text: {
    primary: neutralColors.gray[50],       // Contrast ratio > 7:1 on gray-900
    secondary: neutralColors.gray[300],    // Contrast ratio > 4.5:1 on gray-900
    disabled: neutralColors.gray[600],
    hint: neutralColors.gray[500],
  },
  // UI element colors for dark mode
  action: {
    active: brandColors.secondaryLight,
    hover: `${brandColors.secondary}1F`,   // 12% opacity
    hoverOpacity: 0.12,
    selected: `${brandColors.secondary}29`, // 16% opacity
    selectedOpacity: 0.16,
    disabled: neutralColors.gray[700],
    disabledBackground: neutralColors.gray[800],
    disabledOpacity: 0.38,
    focus: `${brandColors.secondary}33`,    // 20% opacity
    focusOpacity: 0.20,
  },
  // Divider colors for dark mode
  divider: neutralColors.gray[700],
  // Status colors adjusted for dark mode
  error: {
    main: '#EF4444',
    light: '#F87171',
    dark: '#DC2626',
    contrastText: neutralColors.gray[900],
  },
  warning: {
    main: '#FBBf24',
    light: '#FCD34D',
    dark: '#F59E0B',
    contrastText: neutralColors.gray[900],
  },
  info: {
    main: brandColors.secondaryLight,
    light: '#67E8F9',
    dark: brandColors.secondary,
    contrastText: neutralColors.gray[900],
  },
  success: {
    main: '#34D399',
    light: '#6EE7B7',
    dark: '#10B981',
    contrastText: neutralColors.gray[900],
  },
};

// Typography configuration
const typography = {
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  h1: {
    fontSize: '3.5rem',
    fontWeight: 700,
    lineHeight: 1.2,
    letterSpacing: '-0.02em',
  },
  h2: {
    fontSize: '2.5rem',
    fontWeight: 700,
    lineHeight: 1.3,
    letterSpacing: '-0.01em',
  },
  h3: {
    fontSize: '2rem',
    fontWeight: 600,
    lineHeight: 1.4,
  },
  h4: {
    fontSize: '1.5rem',
    fontWeight: 600,
    lineHeight: 1.5,
  },
  h5: {
    fontSize: '1.25rem',
    fontWeight: 600,
    lineHeight: 1.6,
  },
  h6: {
    fontSize: '1.125rem',
    fontWeight: 600,
    lineHeight: 1.6,
  },
  subtitle1: {
    fontSize: '1.125rem',
    lineHeight: 1.7,
  },
  subtitle2: {
    fontSize: '1rem',
    lineHeight: 1.6,
    fontWeight: 500,
  },
  body1: {
    fontSize: '1rem',
    lineHeight: 1.7,
  },
  body2: {
    fontSize: '0.875rem',
    lineHeight: 1.6,
  },
  button: {
    fontSize: '1rem',
    fontWeight: 600,
    letterSpacing: '0.02em',
    textTransform: 'none',
  },
  caption: {
    fontSize: '0.75rem',
    lineHeight: 1.5,
  },
  overline: {
    fontSize: '0.75rem',
    fontWeight: 600,
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
  },
};

// Component overrides
const components = {
  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: 8,
        padding: '12px 24px',
        fontSize: '1rem',
        fontWeight: 600,
        textTransform: 'none',
        transition: 'all 0.3s ease',
      },
      containedPrimary: {
        boxShadow: 'none',
        '&:hover': {
          boxShadow: '0 4px 12px rgba(30, 58, 138, 0.2)',
          transform: 'translateY(-2px)',
        },
      },
      containedSecondary: {
        boxShadow: 'none',
        '&:hover': {
          boxShadow: '0 4px 12px rgba(6, 182, 212, 0.2)',
          transform: 'translateY(-2px)',
        },
      },
      outlined: {
        borderWidth: 2,
        '&:hover': {
          borderWidth: 2,
        },
      },
    },
  },
  MuiCard: {
    styleOverrides: {
      root: {
        borderRadius: 16,
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
        transition: 'all 0.3s ease',
        '&:hover': {
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.12)',
          transform: 'translateY(-4px)',
        },
      },
    },
  },
  MuiPaper: {
    styleOverrides: {
      root: {
        borderRadius: 12,
      },
      elevation1: {
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.06)',
      },
      elevation2: {
        boxShadow: '0 4px 8px rgba(0, 0, 0, 0.08)',
      },
      elevation3: {
        boxShadow: '0 8px 16px rgba(0, 0, 0, 0.1)',
      },
    },
  },
};

// Shape configuration
const shape = {
  borderRadius: 8,
};

// Spacing configuration (8px base)
const spacing = 8;

// Breakpoints
const breakpoints = {
  values: {
    xs: 0,
    sm: 600,
    md: 900,
    lg: 1200,
    xl: 1536,
  },
};

// Create theme function
function createBlyanTheme(mode = 'light') {
  return {
    palette: mode === 'dark' ? darkPalette : lightPalette,
    typography,
    components,
    shape,
    spacing,
    breakpoints,
  };
}

// Export themes
const lightTheme = createBlyanTheme('light');
const darkTheme = createBlyanTheme('dark');

// CSS variables for non-MUI components
const cssVariables = {
  light: {
    '--color-primary': lightPalette.primary.main,
    '--color-primary-light': lightPalette.primary.light,
    '--color-primary-dark': lightPalette.primary.dark,
    '--color-secondary': lightPalette.secondary.main,
    '--color-secondary-light': lightPalette.secondary.light,
    '--color-secondary-dark': lightPalette.secondary.dark,
    '--color-background': lightPalette.background.default,
    '--color-surface': lightPalette.background.paper,
    '--color-text-primary': lightPalette.text.primary,
    '--color-text-secondary': lightPalette.text.secondary,
    '--color-divider': lightPalette.divider,
    '--color-action-hover': lightPalette.action.hover,
    '--color-action-selected': lightPalette.action.selected,
    '--shadow-sm': '0 2px 4px rgba(0, 0, 0, 0.06)',
    '--shadow-md': '0 4px 8px rgba(0, 0, 0, 0.08)',
    '--shadow-lg': '0 8px 16px rgba(0, 0, 0, 0.1)',
    '--shadow-xl': '0 16px 32px rgba(0, 0, 0, 0.12)',
  },
  dark: {
    '--color-primary': darkPalette.primary.main,
    '--color-primary-light': darkPalette.primary.light,
    '--color-primary-dark': darkPalette.primary.dark,
    '--color-secondary': darkPalette.secondary.main,
    '--color-secondary-light': darkPalette.secondary.light,
    '--color-secondary-dark': darkPalette.secondary.dark,
    '--color-background': darkPalette.background.default,
    '--color-surface': darkPalette.background.paper,
    '--color-text-primary': darkPalette.text.primary,
    '--color-text-secondary': darkPalette.text.secondary,
    '--color-divider': darkPalette.divider,
    '--color-action-hover': darkPalette.action.hover,
    '--color-action-selected': darkPalette.action.selected,
    '--shadow-sm': '0 2px 4px rgba(0, 0, 0, 0.2)',
    '--shadow-md': '0 4px 8px rgba(0, 0, 0, 0.3)',
    '--shadow-lg': '0 8px 16px rgba(0, 0, 0, 0.4)',
    '--shadow-xl': '0 16px 32px rgba(0, 0, 0, 0.5)',
  },
};

// Export everything
export {
  lightTheme,
  darkTheme,
  cssVariables,
  createBlyanTheme,
  brandColors,
  neutralColors,
};