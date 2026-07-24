/**
 * Joins Astro's configured base path with an internal route.
 * Exported separately from `href` so it can be unit-tested without
 * `import.meta.env`, which only exists inside the Astro build.
 */
export function joinBase(base: string, path: string): string {
  const cleanBase = base.endsWith('/') ? base.slice(0, -1) : base;
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  if (cleanPath === '/') return `${cleanBase}/`;
  return `${cleanBase}${cleanPath}`;
}

/** Builds an internal link that is correct under the deployed base path. */
export function href(path: string): string {
  return joinBase(import.meta.env.BASE_URL, path);
}
