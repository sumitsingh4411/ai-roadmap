import { describe, it, expect } from 'vitest';
import { searchDocs } from '../../src/lib/search';
import type { SearchDoc } from '../../scripts/build-search-index';

const docs: SearchDoc[] = [
  { slug: 'numpy', title: 'NumPy', stage: 1, summary: 'Arrays and shapes.', headings: ['Why this matters'], text: 'vectorised arrays broadcasting' },
  { slug: 'pandas', title: 'Pandas', stage: 1, summary: 'DataFrames.', headings: ['In code'], text: 'tables columns groupby numpy interop' },
  { slug: 'rag', title: 'RAG', stage: 5, summary: 'Retrieval augmented generation.', headings: [], text: 'retrieval chunks embeddings' },
];

describe('searchDocs', () => {
  it('returns nothing for an empty query', () => {
    expect(searchDocs(docs, '')).toEqual([]);
  });

  it('matches on title', () => {
    expect(searchDocs(docs, 'numpy')[0].slug).toBe('numpy');
  });

  it('is case-insensitive', () => {
    expect(searchDocs(docs, 'NUMPY')[0].slug).toBe('numpy');
  });

  it('ranks a title match above a body match', () => {
    const results = searchDocs(docs, 'numpy');
    expect(results.map((d) => d.slug)).toEqual(['numpy', 'pandas']);
  });

  it('matches on summary', () => {
    expect(searchDocs(docs, 'dataframes')[0].slug).toBe('pandas');
  });

  it('matches on body text', () => {
    expect(searchDocs(docs, 'broadcasting')[0].slug).toBe('numpy');
  });

  it('matches on headings', () => {
    expect(searchDocs(docs, 'why this matters')[0].slug).toBe('numpy');
  });

  it('requires every term to match', () => {
    expect(searchDocs(docs, 'numpy zzz')).toEqual([]);
  });

  it('respects the result limit', () => {
    expect(searchDocs(docs, 'a', 1).length).toBeLessThanOrEqual(1);
  });

  it('ranks a word-boundary match above a mid-word match in an equal-weight field', () => {
    // Slugs are deliberately reverse-alphabetical to the expected ranking, so
    // this only passes because of the boundary score bonus — not because of
    // the slug tiebreaker in searchDocs' sort.
    const boundaryDocs: SearchDoc[] = [
      { slug: 'zzz-boundary', title: 'x', stage: 1, summary: 's', headings: [], text: 'gen ai basics' },
      { slug: 'aaa-midword', title: 'x', stage: 1, summary: 's', headings: [], text: 'imagen ai basics' },
    ];
    const results = searchDocs(boundaryDocs, 'gen');
    expect(results.map((d) => d.slug)).toEqual(['zzz-boundary', 'aaa-midword']);
  });
});
