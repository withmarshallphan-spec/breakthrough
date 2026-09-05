/**
 * Node's ESM resolver requires a file extension. The library modules import
 * each other the way the bundler expects -- './palm-geometry' -- so under
 * `--experimental-strip-types` those specifiers need the extension added back.
 * Only relative specifiers are retried, and only after the normal resolution
 * has already failed.
 */
export async function resolve(specifier, context, nextResolve) {
  try {
    return await nextResolve(specifier, context);
  } catch (error) {
    if (specifier.startsWith('.') && !/\.[cm]?[jt]s$/.test(specifier)) {
      return nextResolve(`${specifier}.ts`, context);
    }
    throw error;
  }
}
