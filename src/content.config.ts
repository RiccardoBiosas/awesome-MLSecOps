import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { defineCollection } from "astro:content";
import { file } from "astro/loaders";
import { z } from "zod";
import sponsorData from "./data/sponsors.json";
import { TOOL_CATEGORY_IDS } from "./lib/toolCategories";

const TOOL_CONTENT_DIR = resolve("./src/content/tools");

const toolDataSchema = z.object({
  category: z.enum(TOOL_CATEGORY_IDS),
  name: z.string().min(1),
  url: z.url(),
  description: z.string().min(1),
});

const toolFileSchema = toolDataSchema.extend({
  id: z.string().min(1),
});

const tools = defineCollection({
  loader: async () => {
    try {
      const fileNames = (await readdir(TOOL_CONTENT_DIR)).filter((fileName) => fileName.endsWith(".json")).sort();
      return await Promise.all(
        fileNames.map(async (fileName) => {
          const content = await readFile(resolve(TOOL_CONTENT_DIR, fileName), "utf8");
          return toolFileSchema.parse(JSON.parse(content));
        }),
      );
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw error;
    }
  },
  schema: toolDataSchema,
});

const jobs = defineCollection({
  loader: file("./src/data/jobs.json"),
  schema: z.object({
    title: z.string().min(1),
    company: z.string().min(1),
    company_url: z.url(),
    url: z.url(),
    location: z.string().min(1),
    country: z.string().length(2),
    description: z.string().min(1),
    employment_type: z.string().default("FULL_TIME"),
    posted_date: z.coerce.date(),
    expires_date: z.coerce.date(),
    featured: z.boolean().default(false),
  }),
});

const sponsors = defineCollection({
  loader: async () => sponsorData,
  schema: z.object({
    name: z.string().min(1),
    url: z.url(),
    logo: z.string().min(1),
    tagline: z.string().min(1),
  }),
});

const newsletter = defineCollection({
  loader: file("./src/data/newsletter.json"),
  schema: z.object({
    title: z.string().min(1),
    date: z.coerce.date(),
    summary: z.string().min(1),
    link: z.url(),
  }),
});

export const collections = { jobs, newsletter, sponsors, tools };