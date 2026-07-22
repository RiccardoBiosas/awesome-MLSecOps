import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import type { APIRoute } from "astro";
import { SITE_URL } from "../config/site";

export const prerender = true;

export const GET: APIRoute = async () => {
  const issues = (await getCollection("newsletter")).sort((left, right) => right.data.date.getTime() - left.data.date.getTime());
  return rss({
    title: "The MLSecOps Hacker Newsletter",
    description: "Machine learning security, model supply-chain, LLM attack, and MLOps security research from The MLSecOps Hacker.",
    site: SITE_URL,
    items: issues.map((issue) => ({
      title: issue.data.title,
      description: issue.data.summary,
      pubDate: issue.data.date,
      link: issue.data.link,
    })),
    customData: "<language>en-us</language>",
  });
};