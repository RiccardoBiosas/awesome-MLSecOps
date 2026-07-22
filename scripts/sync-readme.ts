import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { toString } from "mdast-util-to-string";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import { unified } from "unified";
import type { ToolCategoryId } from "../src/lib/toolCategories";

type AstNode = {
  type: string;
  depth?: number;
  url?: string;
  children?: AstNode[];
};

type ParsedTool = {
  id: string;
  category: ToolCategoryId;
  name: string;
  url: string;
  description: string;
};

const README_SOURCE = process.env.README_SOURCE ?? "README.md";
const OUTPUT_PATH = resolve("src/data/tools.json");

const TOOL_SECTIONS = new Set([
  "open source security tools",
  "commercial tools",
  "commercial mlsecops and ai security tools",
  "data",
  "data privacy and anonymization tools",
  "ml code security",
  "ml code security and model scanning",
]);

const CATEGORY_RULES: Array<[ToolCategoryId, RegExp]> = [
  ["agent-security", /\b(agent|agentic|mcp|memory guard|skill-audit|crewai|langgraph|autogen)\b/i],
  ["supply-chain", /\b(sbom|bomlens|supply.?chain|artifact|signing|serialization|safetensor|pickle|modelscan|ml-bom|provenance|mlops|datalake)\b/i],
  ["privacy", /\b(privacy|anonym|differential|encrypted|encryption|pii|data masking|opendp|membership inference)\b/i],
  ["llm-security", /\b(llms?|large language models?|prompt|jailbreak|garak|giskard|guardrail|trustgate|pyrit|gandalf|genai|generative ai)\b/i],
  ["adversarial-ml", /\b(adversarial|robust\w*|fool|attack|inversion|steal|copycat|evasion|poison|pentest\w*|payloads?|textattack|openattack|cleverhans|augly)\b/i],
  ["model-scanning", /\b(scan|scanner|lint\w*|validation|evaluation|benchmark|bias testing|watchtower|model analysis|model protection|vulnerabilit\w*|secure jupyter|model as code|security best practice|detection and response)\b/i],
];

function normalizeHeading(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function findLink(node: AstNode): { name: string; url: string } | undefined {
  if (node.type === "link" && node.url) {
    return { name: toString(node as never).trim(), url: node.url };
  }

  for (const child of node.children ?? []) {
    const link = findLink(child);
    if (link) return link;
  }

  return undefined;
}

function canonicalizeUrl(value: string): string {
  const url = new URL(value);
  for (const key of [...url.searchParams.keys()]) {
    if (key.toLowerCase().startsWith("utm_") || ["referral", "wt.mc_id"].includes(key.toLowerCase())) {
      url.searchParams.delete(key);
    }
  }
  return url.toString().replace(/\?$/, "");
}

function classify(name: string, description: string, section: string): ToolCategoryId {
  if (section.includes("data privacy") || section === "data") return "privacy";
  const text = `${name} ${description}`;
  return CATEGORY_RULES.find(([, pattern]) => pattern.test(text))?.[0] ?? "model-scanning";
}

function makeId(name: string, url: string, seen: Set<string>): string {
  const base = name
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "") || "tool";
  if (!seen.has(base)) {
    seen.add(base);
    return base;
  }
  const suffix = createHash("sha1").update(url).digest("hex").slice(0, 7);
  const id = `${base}-${suffix}`;
  seen.add(id);
  return id;
}

async function loadReadme(): Promise<string> {
  if (/^https?:\/\//.test(README_SOURCE)) {
    const response = await fetch(README_SOURCE, { headers: { "User-Agent": "awesome-mlsecops-site-sync" } });
    if (!response.ok) throw new Error(`README fetch failed: ${response.status} ${response.statusText}`);
    return response.text();
  }
  return readFile(resolve(README_SOURCE), "utf8");
}

function parseTools(markdown: string): ParsedTool[] {
  const root = unified().use(remarkParse).use(remarkGfm).parse(markdown) as AstNode;
  const pending: Array<Omit<ParsedTool, "id">> = [];
  let section = "";

  for (const child of root.children ?? []) {
    if (child.type === "heading" && child.depth === 2) {
      section = normalizeHeading(toString(child as never));
      continue;
    }
    if (!TOOL_SECTIONS.has(section)) continue;

    if (child.type === "table") {
      for (const row of (child.children ?? []).slice(1)) {
        const [nameCell, descriptionCell] = row.children ?? [];
        if (!nameCell || !descriptionCell) continue;
        const link = findLink(nameCell);
        const description = toString(descriptionCell as never).trim().replace(/\s+/g, " ");
        if (!link || !description) continue;
        pending.push({
          category: classify(link.name, description, section),
          name: link.name,
          url: canonicalizeUrl(link.url),
          description,
        });
      }
    }

    if (section.startsWith("ml code security") && child.type === "list") {
      for (const item of child.children ?? []) {
        const link = findLink(item);
        if (!link) continue;
        const fullText = toString(item as never).trim().replace(/\s+/g, " ");
        const description = fullText.replace(link.name, "").replace(/^\s*-\s*/, "").trim();
        if (!description) continue;
        pending.push({
          category: classify(link.name, description, section),
          name: link.name,
          url: canonicalizeUrl(link.url),
          description,
        });
      }
    }
  }

  const seenIds = new Set<string>();
  const seenEntries = new Set<string>();
  return pending
    .filter((tool) => {
      const key = `${tool.name}\u0000${tool.url}`;
      if (seenEntries.has(key)) return false;
      seenEntries.add(key);
      return true;
    })
    .map((tool) => ({ id: makeId(tool.name, tool.url, seenIds), ...tool }));
}

const tools = parseTools(await loadReadme());
if (tools.length < 60) {
  throw new Error(`README sync found only ${tools.length} tools; refusing to overwrite the collection.`);
}

await mkdir(resolve("src/data"), { recursive: true });
await writeFile(OUTPUT_PATH, `${JSON.stringify(tools, null, 2)}\n`, "utf8");
console.log(`Synced ${tools.length} tools from ${README_SOURCE} to ${OUTPUT_PATH}`);