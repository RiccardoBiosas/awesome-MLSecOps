import { getCollection } from "astro:content";
import type { APIRoute } from "astro";

export const prerender = true;

export const GET: APIRoute = async () => {
  const tools = (await getCollection("tools"))
    .map((tool) => ({
      category: tool.data.category,
      name: tool.data.name,
      url: tool.data.url,
      description: tool.data.description,
      source: "github.com/RiccardoBiosas/awesome-MLSecOps" as const,
    }))
    .sort((left, right) => left.category.localeCompare(right.category) || left.name.localeCompare(right.name));

  return new Response(JSON.stringify(tools, null, 2), {
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
};