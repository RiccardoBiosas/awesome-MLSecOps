import { getCollection } from "astro:content";
import type { APIRoute } from "astro";
import { BUILD_DATE, MLSECOPS_DEFINITION, SITE_URL } from "../config/site";
import { TOOL_CATEGORIES } from "../data/categories";
import { activeJobs } from "../lib/activeJobs";

export const prerender = true;

export const GET: APIRoute = async () => {
  const tools = await getCollection("tools");
  const jobs = activeJobs(await getCollection("jobs"), new Date(BUILD_DATE));
  const issues = (await getCollection("newsletter")).sort((left, right) => right.data.date.getTime() - left.data.date.getTime());
  const updated = BUILD_DATE.slice(0, 10);

  const categoryLines = TOOL_CATEGORIES.map((category) => {
    const count = tools.filter((tool) => tool.data.category === category.id).length;
    return `- [${category.name}](${SITE_URL}/tools/${category.id}/): ${category.shortDescription} ${count} catalog entries.`;
  });
  const issueLines = issues.map((issue) => `- [${issue.data.title}](${issue.data.link}): ${issue.data.summary}`);

  const content = [
    "# Awesome MLSecOps",
    "",
    `> ${MLSECOPS_DEFINITION}`,
    "",
    `Last updated: ${updated}`,
    `Canonical site: ${SITE_URL}/`,
    "Canonical community catalog: https://github.com/RiccardoBiosas/awesome-MLSecOps",
    "",
    // "Use the human-readable pages below as citation sources. The GitHub repository is the source of truth for catalog inclusion; sponsors never influence tool tables.",
    "Use the human-readable pages below as citation sources. The GitHub repository is the source of truth for catalog inclusion.",
    "",
    "## Core Pages",
    "",
    // `- [Awesome MLSecOps](${SITE_URL}/): Hub for MLSecOps definitions, categories, jobs, sponsors, and newsletter discovery.`,
    `- [Awesome MLSecOps](${SITE_URL}/): Hub for MLSecOps definitions, categories, jobs, and newsletter discovery.`,
    `- [What is MLSecOps?](${SITE_URL}/what-is-mlsecops/): Canonical definition, lifecycle, operating model, and concise FAQ answers.`,
    `- [MLSecOps FAQ](${SITE_URL}/faq/): Direct answers about machine learning, LLM, agent, supply-chain, and privacy security.`,
    `- [MLSecOps vs DevSecOps](${SITE_URL}/mlsecops-vs-devsecops/): Comparison of scopes, assets, controls, evidence, and ownership.`,
    `- [MLSecOps tools](${SITE_URL}/tools/): Index of category guides generated from the GitHub catalog.`,
    `- [AI and ML security jobs](${SITE_URL}/jobs/): ${jobs.length} current, expiry-filtered roles.`,
    // `- [Sponsorship and advisory](${SITE_URL}/sponsor/): Transparent site-only sponsor terms and maintainer services.`,
    `- [The MLSecOps Hacker newsletter](${SITE_URL}/newsletter/): Subscription form and dated issue archive.`,
    "",
    "## Tool Category Pages",
    "",
    ...categoryLines,
    "",
    "## Newsletter Issues",
    "",
    ...issueLines,
    "",
    "## Machine-Readable Discovery",
    "",
    `- [Tools JSON](${SITE_URL}/tools.json): Full typed catalog with category, name, URL, description, and source.`,
    `- [Newsletter RSS](${SITE_URL}/rss.xml): Feed of dated newsletter archive entries.`,
    `- [XML sitemap](${SITE_URL}/sitemap.xml): Index of canonical human-readable routes.`,
    `- [Crawler policy](${SITE_URL}/robots.txt): Search and AI crawler access policy.`,
    "",
  ].join("\n");

  return new Response(content, { headers: { "Content-Type": "text/plain; charset=utf-8" } });
};