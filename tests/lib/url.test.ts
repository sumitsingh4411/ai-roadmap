import { describe, it, expect } from 'vitest';
import { joinBase } from '../../src/lib/url';

describe('joinBase', () => {
  it('joins a base path and a route without doubling slashes', () => {
    expect(joinBase('/ai-roadmap/', '/lessons/numpy')).toBe('/ai-roadmap/lessons/numpy');
  });

  it('handles a base without a trailing slash', () => {
    expect(joinBase('/ai-roadmap', '/lessons/numpy')).toBe('/ai-roadmap/lessons/numpy');
  });

  it('handles a route without a leading slash', () => {
    expect(joinBase('/ai-roadmap/', 'lessons/numpy')).toBe('/ai-roadmap/lessons/numpy');
  });

  it('returns the base itself for the root route', () => {
    expect(joinBase('/ai-roadmap/', '/')).toBe('/ai-roadmap/');
  });

  it('works when deployed at the domain root', () => {
    expect(joinBase('/', '/lessons/numpy')).toBe('/lessons/numpy');
  });
});
