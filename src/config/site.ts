export const SITE_NAME = "Awesome MLSecOps";
export const SITE_URL = "https://awesomemlsecops.com";
export const REPOSITORY_URL = "https://github.com/RiccardoBiosas/awesome-MLSecOps";
export const CONTRIBUTE_URL = `${REPOSITORY_URL}/blob/main/CONTRIBUTING.md`;
export const GITHUB_SPONSORS_URL = "https://github.com/sponsors/RiccardoBiosas";
export const NEWSLETTER_URL = "https://themlsecopshacker.com";
export const NEWSLETTER_EMBED_URL = "https://themlsecopshacker.com/embed";
export const CONTACT_EMAIL = "riccardobiosas@gmail.com";
export const BUILD_DATE = process.env.BUILD_DATE || new Date().toISOString();

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