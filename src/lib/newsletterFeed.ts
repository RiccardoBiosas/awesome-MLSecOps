export type NewsletterIssue = {
  id: string;
  title: string;
  date: string;
  summary: string;
  link: string;
};

const ENTITIES: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", "#39": "'" };

function decodeEntities(value: string): string {
  return value.replace(/&(#\d+|#x[0-9a-f]+|[a-z]+);/gi, (match, entity: string) => {
    if (entity[0] === "#") {
      const code = entity[1]?.toLowerCase() === "x" ? Number.parseInt(entity.slice(2), 16) : Number.parseInt(entity.slice(1), 10);
      return Number.isNaN(code) ? match : String.fromCodePoint(code);
    }
    return ENTITIES[entity.toLowerCase()] ?? match;
  });
}

function textOf(xml: string, tag: string): string | undefined {
  const match = xml.match(new RegExp(`<${tag}(?:\\s[^>]*)?>([\\s\\S]*?)</${tag}>`, "i"));
  if (!match) return undefined;
  const raw = match[1].replace(/^\s*<!\[CDATA\[([\s\S]*?)\]\]>\s*$/, "$1");
  return decodeEntities(raw.replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ").trim();
}

export function issueIdFromLink(link: string): string {
  const path = new URL(link).pathname.replace(/\/+$/, "");
  return path.split("/").filter(Boolean).pop() ?? path;
}

/** Parse an RSS 2.0 feed (Substack's format) into dated newsletter issues, newest first. */
export function parseNewsletterFeed(xml: string): NewsletterIssue[] {
  const items = [...xml.matchAll(/<item(?:\s[^>]*)?>([\s\S]*?)<\/item>/gi)].map((match) => match[1]);
  const issues: NewsletterIssue[] = [];
  for (const item of items) {
    const title = textOf(item, "title");
    const link = textOf(item, "link");
    const pubDate = textOf(item, "pubDate");
    const summary = textOf(item, "description");
    if (!title || !link || !pubDate || !summary) continue;
    const date = new Date(pubDate);
    if (Number.isNaN(date.getTime())) continue;
    issues.push({ id: issueIdFromLink(link), title, date: date.toISOString().slice(0, 10), summary, link });
  }
  return issues.sort((left, right) => right.date.localeCompare(left.date));
}

/**
 * Merge freshly fetched issues into an existing archive, keyed by canonical link.
 * Title, date, and id follow the feed; a hand-written summary in the archive is kept,
 * because Substack's feed description is only the post subtitle.
 */
export function mergeNewsletterIssues(existing: NewsletterIssue[], fetched: NewsletterIssue[]): NewsletterIssue[] {
  const byLink = new Map(existing.map((issue) => [issue.link, issue]));
  for (const issue of fetched) {
    const current = byLink.get(issue.link);
    byLink.set(issue.link, current?.summary ? { ...issue, summary: current.summary } : issue);
  }
  return [...byLink.values()].sort((left, right) => right.date.localeCompare(left.date));
}

export function sortIssues<T extends { data: { date: Date } }>(issues: T[]): T[] {
  return [...issues].sort((left, right) => right.data.date.getTime() - left.data.date.getTime());
}
