export const STORAGE_KEY = 'ai-roadmap:progress';

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface ProgressStore {
  completed(): string[];
  isComplete(slug: string): boolean;
  toggle(slug: string): string[];
  markComplete(slug: string): string[];
  clear(): void;
}

const noopStorage: StorageLike = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
};

function resolveStorage(storage?: StorageLike): StorageLike {
  if (storage) return storage;
  try {
    return globalThis.localStorage ?? noopStorage;
  } catch {
    return noopStorage; // blocked by browser privacy settings
  }
}

/**
 * Completed-lesson store backed by localStorage.
 *
 * State is held in memory as well as written through, so the UI stays correct
 * even when the browser refuses to persist (private mode, quota exceeded).
 */
export function createProgressStore(storage?: StorageLike): ProgressStore {
  const backing = resolveStorage(storage);

  function load(): string[] {
    try {
      const raw = backing.getItem(STORAGE_KEY);
      if (!raw) return [];
      const parsed: unknown = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter((item): item is string => typeof item === 'string');
    } catch {
      return []; // corrupt payload — start clean rather than crash the page
    }
  }

  let state = load();

  function persist(): string[] {
    try {
      backing.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      /* keep the in-memory value; this session still behaves correctly */
    }
    return [...state];
  }

  return {
    completed: () => [...state],
    isComplete: (slug) => state.includes(slug),
    markComplete(slug) {
      if (!state.includes(slug)) state = [...state, slug];
      return persist();
    },
    toggle(slug) {
      state = state.includes(slug) ? state.filter((s) => s !== slug) : [...state, slug];
      return persist();
    },
    clear() {
      state = [];
      try {
        backing.removeItem(STORAGE_KEY);
      } catch {
        /* ignore */
      }
    },
  };
}
