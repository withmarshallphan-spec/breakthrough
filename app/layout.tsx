import type { Metadata } from 'next';
import { Geist, Instrument_Serif } from 'next/font/google';
import './globals.css';

const publicAssetBasePath = process.env.GITHUB_PAGES === 'true' ? '/breakthrough' : '';

// All three are SIL Open Font License 1.1, and next/font self-hosts the files
// at build time -- nothing is requested from Google at runtime, and there is no
// layout shift while they load.
//
// Instrument Serif carries the reading copy, and the symbols and figures with
// it -- they are body text too. Geist Sans is kept for the small caption tier
// only. The display face, Alte Haas Grotesk Bold, is self-hosted from
// public/fonts and declared in globals.css. A monospace was dropped: with the
// readouts drawn as meters there is no column of changing digits left for it
// to align.
const instrumentSerif = Instrument_Serif({
  variable: '--font-instrument-serif',
  subsets: ['latin'],
  weight: '400',
});

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'Too Expensive to Collapse',
  description:
    'A hand-tracked interactive on the quantum stability of matter: the kinetic-energy cost of localisation, and why bulk matter needs fermionic antisymmetry.',
  icons: { icon: `${publicAssetBasePath}/favicon.svg` },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // The font variables live on the root element, not on body. `--font-sans`
    // in globals.css is declared on :root and substitutes `--font-geist-sans`
    // there, so the variable has to be defined on the same element or the
    // substitution resolves to nothing and the stack silently falls through to
    // the system font.
    <html lang="en" className={`${instrumentSerif.variable} ${geistSans.variable}`}>
      <body className="antialiased">
        <style>{`
          @font-face {
            font-family: 'Alte Haas Grotesk';
            src: url('${publicAssetBasePath}/fonts/alte-haas-grotesk/AlteHaasGroteskRegular.woff2') format('woff2');
            font-weight: 400;
            font-style: normal;
            font-display: swap;
          }
          @font-face {
            font-family: 'Alte Haas Grotesk';
            src: url('${publicAssetBasePath}/fonts/alte-haas-grotesk/AlteHaasGroteskBold.woff2') format('woff2');
            font-weight: 700;
            font-style: normal;
            font-display: swap;
          }
        `}</style>
        {/* The display face is self-hosted and small enough to fetch eagerly;
            the titles are the first thing on screen. Its licence text ships
            beside it, which is what that licence requires. */}
        <link
          rel="preload"
          as="font"
          type="font/woff2"
          href={`${publicAssetBasePath}/fonts/alte-haas-grotesk/AlteHaasGroteskBold.woff2`}
          crossOrigin="anonymous"
        />
        {children}
      </body>
    </html>
  );
}
