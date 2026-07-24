import type { APIRoute } from "astro";
import { SITE_URL } from "../config/site";

export const prerender = true;

// @astrojs/sitemap only emits /sitemap-index.xml; this route serves the same index
// at /sitemap.xml, the canonical URL used by robots.txt and Search Console.
export const GET: APIRoute = () => {
  const body = `<?xml version="1.0" encoding="UTF-8"?>\n<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n  <sitemap>\n    <loc>${SITE_URL}/sitemap-0.xml</loc>\n  </sitemap>\n</sitemapindex>\n`;
  return new Response(body, { headers: { "Content-Type": "application/xml; charset=utf-8" } });
};