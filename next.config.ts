import type { NextConfig } from 'next';

// GitHub Pages serves project sites beneath the repository name. Vinext's
// exporter renders the root route at /, so use an asset prefix rather than a
// basePath to keep the static entry point portable.
const isGitHubPages = process.env.GITHUB_PAGES === 'true';

const nextConfig: NextConfig = {
  output: 'export',
  assetPrefix: isGitHubPages ? '/breakthrough' : '',
  trailingSlash: true,
};

export default nextConfig;
