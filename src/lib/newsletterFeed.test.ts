import { describe, expect, it } from "vitest";
import { issueIdFromLink, mergeNewsletterIssues, parseNewsletterFeed, truncateSummary } from "./newsletterFeed";

const feed = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>The MLSecOps Hacker Newsletter</title>
<item>
  <title><![CDATA[Safetensors Won. Model Serialization Attacks Didn't Stop.]]></title>
  <description><![CDATA[<p>Attackers moved from model files to <em>pipelines</em> &amp; converters.</p>]]></description>
  <link>https://themlsecopshacker.com/p/safetensors-won-model-serialization</link>
  <pubDate>Fri, 28 Aug 2026 12:00:00 GMT</pubDate>
</item>
<item>
  <title>What is MLSecOps?</title>
  <description>An introduction to securing ML systems.</description>
  <link>https://themlsecopshacker.com/p/what-is-mlsecops/</link>
  <pubDate>Tue, 08 Oct 2024 09:30:00 GMT</pubDate>
</item>
<item><title>Broken</title><link>https://themlsecopshacker.com/p/broken</link><pubDate>not a date</pubDate><description>x</description></item>
</channel></rss>`;

describe("parseNewsletterFeed", () => {
  it("extracts dated issues, strips markup, decodes entities, and sorts newest first", () => {
    const issues = parseNewsletterFeed(feed);
    expect(issues).toEqual([
      {
        id: "safetensors-won-model-serialization",
        title: "Safetensors Won. Model Serialization Attacks Didn't Stop.",
        date: "2026-08-28",
        summary: "Attackers moved from model files to pipelines & converters.",
        link: "https://themlsecopshacker.com/p/safetensors-won-model-serialization",
      },
      {
        id: "what-is-mlsecops",
        title: "What is MLSecOps?",
        date: "2024-10-08",
        summary: "An introduction to securing ML systems.",
        link: "https://themlsecopshacker.com/p/what-is-mlsecops/",
      },
    ]);
  });

  it("derives stable ids from the post slug", () => {
    expect(issueIdFromLink("https://themlsecopshacker.com/p/what-is-mlsecops/")).toBe("what-is-mlsecops");
  });
});

describe("truncateSummary", () => {
  it("leaves short summaries alone", () => {
    expect(truncateSummary("Short.")).toBe("Short.");
  });

  it("cuts long summaries at a word boundary with an ellipsis", () => {
    const long = "word ".repeat(60).trim();
    const result = truncateSummary(long);
    expect(result.length).toBeLessThanOrEqual(160);
    expect(result.endsWith("…")).toBe(true);
    expect(result).not.toMatch(/wor…$/);
  });
});

describe("mergeNewsletterIssues", () => {
  it("takes feed metadata for known links, keeps curated summaries, and keeps issues the feed no longer lists", () => {
    const existing = [
      { id: "old", title: "Old", date: "2023-01-01", summary: "Kept.", link: "https://themlsecopshacker.com/p/old" },
      { id: "what-is-mlsecops", title: "Stale title", date: "2024-10-08", summary: "Stale.", link: "https://themlsecopshacker.com/p/what-is-mlsecops" },
    ];
    const fetched = [{ id: "what-is-mlsecops", title: "What is MLSecOps?", date: "2024-10-08", summary: "Fresh.", link: "https://themlsecopshacker.com/p/what-is-mlsecops" }];
    expect(mergeNewsletterIssues(existing, fetched).map((issue) => [issue.id, issue.title, issue.summary])).toEqual([
      ["what-is-mlsecops", "What is MLSecOps?", "Stale."],
      ["old", "Old", "Kept."],
    ]);
  });
});
