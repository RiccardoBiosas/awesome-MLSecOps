export const SITE_NAME = "Awesome MLSecOps";
export const SITE_URL = "https://awesomemlsecops.com";
export const REPOSITORY_URL = "https://github.com/RiccardoBiosas/awesome-MLSecOps";
export const CONTRIBUTE_URL = `${REPOSITORY_URL}/blob/main/CONTRIBUTING.md`;
export const GITHUB_SPONSORS_URL = "https://github.com/sponsors/RiccardoBiosas";
export const NEWSLETTER_URL = "https://themlsecopshacker.com";
export const NEWSLETTER_EMBED_URL = "https://themlsecopshacker.com/embed";
export const NEWSLETTER_FEED_URL = "https://themlsecopshacker.com/feed";
export const NEWSLETTER_NAME = "The MLSecOps Hacker";
/** The publication's own description, including the cadence stated in its Substack settings. */
export const NEWSLETTER_TAGLINE =
  "Mapping the AI security landscape: weekly insights on AI governance, GenAI threat modeling, and deep dives on MLSecOps best practices, tooling, and attack vectors for CISOs, security professionals, and AI engineers.";
export const NEWSLETTER_AUTHOR = { name: "Riccardo Biosas", url: "https://github.com/RiccardoBiosas" };
export const X_URL = "https://x.com/MLSecOpsHacker";
export const INSTAGRAM_URL = "https://www.instagram.com/aisecurity.hacker/";
export const SAME_AS_URLS = [NEWSLETTER_URL, X_URL, INSTAGRAM_URL, REPOSITORY_URL];

/**
 * Subscribe link tagged so Substack's traffic report attributes the signup to this site and
 * to the placement that produced it. Article links stay canonical and carry no parameters.
 */
export function subscribeUrl(placement: string): string {
  const url = new URL("/subscribe", NEWSLETTER_URL);
  url.searchParams.set("utm_source", "awesomemlsecops.com");
  url.searchParams.set("utm_medium", "referral");
  url.searchParams.set("utm_campaign", placement);
  return url.toString();
}
export const CONTACT_EMAIL = "riccardobiosas@gmail.com";
export const BUILD_DATE = process.env.BUILD_DATE || new Date().toISOString();
export const CATALOG_REVIEW_DATE = "2026-07-23";

export const ENTITY_DESCRIPTION =
  "A curated list of awesome open-source tools, resources, and tutorials for MLSecOps (Machine Learning Security Operations).";

export const MLSECOPS_DEFINITION_LEAD =
  "MLSecOps (Machine Learning Security Operations) integrates security engineering, threat modeling, testing, supply-chain controls, monitoring, and incident response across the machine-learning lifecycle.";

export const MLSECOPS_DEFINITION_SUPPORT =
  "It protects data, models, pipelines, infrastructure, LLM applications, and AI agents against poisoning, adversarial manipulation, unsafe artifacts, privacy leakage, model extraction, prompt injection, and excessive agency.";

export const MLSECOPS_DEFINITION = `${MLSECOPS_DEFINITION_LEAD} ${MLSECOPS_DEFINITION_SUPPORT}`;

export function absoluteUrl(path: string): string {
  return new URL(path, SITE_URL).toString();
}

export function toolEntryPath(id: string): string {
  return `/tools/entries/${id}/`;
}

export function toolEntryUrl(id: string): string {
  return absoluteUrl(toolEntryPath(id));
}

export function formattedBuildDate(): string {
  return new Intl.DateTimeFormat("en", {
    dateStyle: "long",
    timeZone: "UTC",
  }).format(new Date(BUILD_DATE));
}

export function formattedCatalogReviewDate(): string {
  return new Intl.DateTimeFormat("en", {
    dateStyle: "long",
    timeZone: "UTC",
  }).format(new Date(`${CATALOG_REVIEW_DATE}T00:00:00Z`));
}