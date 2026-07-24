/**
 * Node ESM loader hook used only when scripts are run directly via `tsx`
 * (outside Astro/Vite).
 *
 * `astro:content` is a Vite virtual module supplied by Astro's dev/build
 * pipeline. `tests/vitest.config.ts` already substitutes it with a trivial
 * pass-through stub for the same reason: Vitest doesn't run Astro's Vite
 * plugin either. `scripts/lib/content-io.ts` imports `stripOrderPrefix` and
 * `lessonSchema` from `src/content.config.ts`, which also imports
 * `defineCollection` from `astro:content` at module scope — so loading it
 * under plain `tsx` needs the same substitution. Neither
 * `stripOrderPrefix` nor `lessonSchema` ever exercises `defineCollection`'s
 * real behaviour, so the existing stub is safe to reuse here.
 */
const STUB_URL = new URL('../tests/stubs/astro-content.ts', import.meta.url).href;

export async function resolve(specifier, context, nextResolve) {
  if (specifier === 'astro:content') {
    return { url: STUB_URL, shortCircuit: true };
  }
  return nextResolve(specifier, context);
}
