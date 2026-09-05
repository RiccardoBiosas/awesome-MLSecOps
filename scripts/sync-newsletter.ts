/**
 * Refresh the on-site newsletter archive and the README newsletter block from the
 * Substack RSS feed. Fails soft: if the feed cannot be fetched, the committed data stays.
 *
 *   npm run sync:newsletter            # refresh src/data/newsletter.json and README.md
 *   NEWSLETTER_SYNC=off npm run sync   # skip the network call
 */
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { NEWSLETTER_FEED_URL } from "../src/config/site";
import { mergeNewsletterIssues, parseNewsletterFeed, type NewsletterIssue } from "../src/lib/newsletterFeed";

const DATA_PATH = resolve("src/data/newsletter.json");
const README_PATH = resolve(process.env.README_SOURCE ?? "README.md");
const README_START = "<!-- newsletter-issues:start -->";
const README_END = "<!-- newsletter-issues:end -->";
const README_ISSUE_LIMIT = 5;

function readmeUtm(link: string): string {
  const url = new URL(link);
  url.searchParams.set("utm_source", "github");
  url.searchParams.set("utm_medium", "readme");
  url.searchParams.set("utm_campaign", "newsletter-section");
  return url.toString();
}

function formatReadmeDate(date: string): string {
  return new Intl.DateTimeFormat("en", { dateStyle: "long", timeZone: "UTC" }).format(new Date(`${date}T00:00:00Z`));
}

export function renderReadmeIssues(issues: NewsletterIssue[]): string {
  const lines = issues.slice(0, README_ISSUE_LIMIT).map((issue) => `- **${formatReadmeDate(issue.date)}** — [${issue.title}](${readmeUtm(issue.link)}) — ${issue.summary}`);
  return `${README_START}\n${lines.join("\n")}\n${README_END}`;
}

async function fetchFeed(): Promise<string | undefined> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10_000);
  try {
    const response = await fetch(NEWSLETTER_FEED_URL, {
      signal: controller.signal,
      headers: { accept: "application/rss+xml, application/xml, text/xml", "user-agent": "awesome-mlsecops-site newsletter sync" },
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.text();
  } catch (error) {
    console.warn(`Newsletter sync skipped: could not fetch ${NEWSLETTER_FEED_URL} (${error instanceof Error ? error.message : String(error)}).`);
    return undefined;
  } finally {
    clearTimeout(timer);
  }
}

async function main(): Promise<void> {
  const existing = JSON.parse(await readFile(DATA_PATH, "utf8")) as NewsletterIssue[];
  let issues = existing;

  if (process.env.NEWSLETTER_SYNC !== "off") {
    const xml = await fetchFeed();
    const fetched = xml ? parseNewsletterFeed(xml) : [];
    if (fetched.length > 0) {
      issues = mergeNewsletterIssues(existing, fetched);
      const serialized = `${JSON.stringify(issues, null, 2)}\n`;
      if (serialized !== (await readFile(DATA_PATH, "utf8"))) {
        await writeFile(DATA_PATH, serialized);
        console.log(`Newsletter archive updated: ${issues.length} issues, latest "${issues[0].title}" (${issues[0].date}).`);
      } else {
        console.log(`Newsletter archive unchanged: ${issues.length} issues.`);
      }
    } else if (xml) {
      console.warn("Newsletter sync skipped: feed parsed to zero issues.");
    }
  }

  const readme = await readFile(README_PATH, "utf8");
  const start = readme.indexOf(README_START);
  const end = readme.indexOf(README_END);
  if (start === -1 || end === -1 || end < start) {
    console.warn(`README newsletter block not found (${README_START} … ${README_END}); README left unchanged.`);
    return;
  }
  const updated = readme.slice(0, start) + renderReadmeIssues(issues) + readme.slice(end + README_END.length);
  if (updated !== readme) {
    await writeFile(README_PATH, updated);
    console.log("README newsletter block updated.");
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
