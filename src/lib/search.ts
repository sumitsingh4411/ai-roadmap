import type { SearchDoc } from '../../scripts/build-search-index';

const FIELD_WEIGHTS = { title: 10, summary: 5, headings: 3, text: 1 } as const;

function scoreField(haystack: string, term: string, weight: number): number {
  const index = haystack.indexOf(term);
  if (index === -1) return 0;
  // A match at a word boundary is a stronger signal than one mid-word.
  const boundary = index === 0 || /\W/.test(haystack[index - 1]);
  return weight * (boundary ? 2 : 1);
}

/**
 * Ranks documents by weighted substring matches.
 * Every term must match somewhere, so extra words narrow the result set.
 */
export function searchDocs(docs: SearchDoc[], query: string, limit = 12): SearchDoc[] {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return [];

  const ranked: Array<{ doc: SearchDoc; score: number }> = [];

  for (const doc of docs) {
    const fields = {
      title: doc.title.toLowerCase(),
      summary: doc.summary.toLowerCase(),
      headings: doc.headings.join(' ').toLowerCase(),
      text: doc.text.toLowerCase(),
    };

    let total = 0;
    let allTermsMatched = true;

    for (const term of terms) {
      let termScore = 0;
      for (const [field, weight] of Object.entries(FIELD_WEIGHTS)) {
        termScore += scoreField(fields[field as keyof typeof fields], term, weight);
      }
      if (termScore === 0) {
        allTermsMatched = false;
        break;
      }
      total += termScore;
    }

    if (allTermsMatched) ranked.push({ doc, score: total });
  }

  return ranked
    .sort((a, b) => b.score - a.score || a.doc.slug.localeCompare(b.doc.slug))
    .slice(0, limit)
    .map((r) => r.doc);
}
