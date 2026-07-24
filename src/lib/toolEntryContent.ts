import type { ToolCategory } from "../data/categories";

type ToolRecord = {
  name: string;
  url: string;
  description: string;
};

export type ToolEntryContent = {
  answer: string;
  metaDescription: string;
  sourceStatement: string;
  title: string;
};

function withTerminalPunctuation(value: string): string {
  return /[.!?]$/.test(value) ? value : `${value}.`;
}

function sourceStatement(value: string): string {
  const url = new URL(value);
  const host = url.hostname.replace(/^www\./, "");

  if (host === "github.com") {
    const [owner, repository] = url.pathname.split("/").filter(Boolean);
    if (owner && repository) {
      return `The linked first-party source is the ${owner}/${repository.replace(/\.git$/, "")} repository on GitHub.`;
    }
  }

  return `The linked first-party source is published at ${host}.`;
}

function truncateAtWord(value: string, maxLength: number): string {
  if (value.length <= maxLength) return value;
  const shortened = value.slice(0, maxLength - 1);
  const lastSpace = shortened.lastIndexOf(" ");
  return `${shortened.slice(0, lastSpace > maxLength * 0.7 ? lastSpace : maxLength - 1).trimEnd()}…`;
}

function formatList(values: string[]): string {
  if (values.length < 2) return values[0] ?? "documented evidence";
  return `${values.slice(0, -1).join(", ")}, and ${values.at(-1)}`;
}

export function createToolEntryContent(tool: ToolRecord, category: ToolCategory): ToolEntryContent {
  const source = sourceStatement(tool.url);
  const criteria = formatList(category.evaluation);
  const answer = [
    `${tool.name} is included in the Awesome MLSecOps ${category.name} directory.`,
    `The community-maintained README describes it as: “${withTerminalPunctuation(tool.description)}”`,
    category.entryScope,
    source,
    `A technical review should test the project's documented evidence across four criteria: ${criteria}. Compare that evidence with the intended architecture and threat model.`,
    `Catalog inclusion establishes relevance to this security category; it is not a certification, comparative ranking, or endorsement. Confirm current capabilities, maintenance, licensing, limitations, and deployment assumptions in the first-party documentation before adoption.`,
  ].join(" ");
  const metaDescription = truncateAtWord(
    `${tool.name}: ${withTerminalPunctuation(tool.description)} MLSecOps context, evaluation criteria, and first-party source from Awesome MLSecOps.`,
    158,
  );
  const title = truncateAtWord(`${tool.name}: ${category.shortName} context | Awesome MLSecOps`, 60);

  return { answer, metaDescription, sourceStatement: source, title };
}
