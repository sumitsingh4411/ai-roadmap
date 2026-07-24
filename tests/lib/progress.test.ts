import { describe, it, expect, beforeEach } from 'vitest';
import { createProgressStore, STORAGE_KEY, type StorageLike } from '../../src/lib/progress';

function memoryStorage(seed: Record<string, string> = {}): StorageLike {
  const map = new Map(Object.entries(seed));
  return {
    getItem: (k) => map.get(k) ?? null,
    setItem: (k, v) => void map.set(k, v),
    removeItem: (k) => void map.delete(k),
  };
}

describe('createProgressStore', () => {
  let storage: StorageLike;

  beforeEach(() => {
    storage = memoryStorage();
  });

  it('starts empty', () => {
    expect(createProgressStore(storage).completed()).toEqual([]);
  });

  it('marks a lesson complete', () => {
    const store = createProgressStore(storage);
    store.markComplete('numpy');
    expect(store.isComplete('numpy')).toBe(true);
  });

  it('does not duplicate a lesson marked twice', () => {
    const store = createProgressStore(storage);
    store.markComplete('numpy');
    store.markComplete('numpy');
    expect(store.completed()).toEqual(['numpy']);
  });

  it('toggles a lesson off again', () => {
    const store = createProgressStore(storage);
    store.toggle('numpy');
    expect(store.isComplete('numpy')).toBe(true);
    store.toggle('numpy');
    expect(store.isComplete('numpy')).toBe(false);
  });

  it('persists across store instances', () => {
    createProgressStore(storage).markComplete('pandas');
    expect(createProgressStore(storage).completed()).toEqual(['pandas']);
  });

  it('clears everything', () => {
    const store = createProgressStore(storage);
    store.markComplete('a');
    store.clear();
    expect(store.completed()).toEqual([]);
  });

  it('recovers from corrupt stored JSON', () => {
    const store = createProgressStore(memoryStorage({ [STORAGE_KEY]: 'not json{{' }));
    expect(store.completed()).toEqual([]);
    store.markComplete('a');
    expect(store.completed()).toEqual(['a']);
  });

  it('discards stored data that is not an array of strings', () => {
    const store = createProgressStore(memoryStorage({ [STORAGE_KEY]: '{"a":1}' }));
    expect(store.completed()).toEqual([]);
  });

  it('filters non-string members out of a stored array', () => {
    const store = createProgressStore(memoryStorage({ [STORAGE_KEY]: '["a",3,null,"b"]' }));
    expect(store.completed()).toEqual(['a', 'b']);
  });

  it('survives a storage that throws on write', () => {
    const throwing: StorageLike = {
      getItem: () => null,
      setItem: () => {
        throw new Error('QuotaExceeded');
      },
      removeItem: () => {},
    };
    const store = createProgressStore(throwing);
    expect(() => store.markComplete('a')).not.toThrow();
    expect(store.isComplete('a')).toBe(true);
  });
});
