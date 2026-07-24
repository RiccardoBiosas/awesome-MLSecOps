import { getCollection } from "astro:content";
import type { APIRoute } from "astro";
import { absoluteUrl, toolEntryUrl } from "../config/site";
import { CATEGORY_BY_ID } from "../data/categories";
import { createToolEntryContent } from "../lib/toolEntryContent";

export const prerender = true;

export const GET: APIRoute = async () => {
  const tools = (await getCollection("tools"))
    .map((tool) => {
      const category = CATEGORY_BY_ID.get(tool.data.category);
      if (!category) throw new Error(`Unknown tool category: ${tool.data.category}`);
      const entryContent = createToolEntryContent(tool.data, category);

      return {
        id: tool.id,
        directoryUrl: toolEntryUrl(tool.id),
        categoryUrl: absoluteUrl(`/tools/${tool.data.category}/`),
        category: tool.data.category,
        name: tool.data.name,
        sourceUrl: tool.data.url,
        description: tool.data.description,
        directoryDescription: entryContent.answer,
        source: "github.com/RiccardoBiosas/awesome-MLSecOps" as const,
      };
    })
    .sort((left, right) => left.category.localeCompare(right.category) || left.name.localeCompare(right.name));

  return new Response(JSON.stringify(tools, null, 2), {
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
};