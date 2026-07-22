import type { APIRoute } from "astro";
import { BUILD_DATE, SITE_URL } from "../config/site";

export const prerender = true;

export const GET: APIRoute = () => {
  const body = `<?xml version="1.0" encoding="UTF-8"?>\n<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n  <sitemap>\n    <loc>${SITE_URL}/sitemap-0.xml</loc>\n    <lastmod>${BUILD_DATE}</lastmod>\n  </sitemap>\n</sitemapindex>\n`;
  return new Response(body, { headers: { "Content-Type": "application/xml; charset=utf-8" } });
};