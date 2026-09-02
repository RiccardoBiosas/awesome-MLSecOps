import sitemap from "@astrojs/sitemap";
import { defineConfig } from "astro/config";
import jobs from "./src/data/jobs.json";
import { BUILD_DATE } from "./src/config/site";
import { activeJobs } from "./src/lib/activeJobs";

const hasActiveJobs = activeJobs(
  jobs.map((job) => ({ data: { expires_date: new Date(`${job.expires_date}T00:00:00Z`) } })),
  new Date(BUILD_DATE),
).length > 0;

export default defineConfig({
  site: "https://awesomemlsecops.com",
  output: "static",
  integrations: [
    sitemap({
      filter(page) {
        const pathname = new URL(page).pathname;
        if (pathname === "/jobs/" && !hasActiveJobs) return false;
        if (pathname === "/sponsor/") return false; // Sponsorship is temporarily disabled.
        if (pathname.startsWith("/tools/entries/")) return false;
        return !pathname.startsWith("/og/") && !["/llms.txt", "/rss.xml", "/sitemap.xml", "/tools.json"].includes(pathname);
      },
    }),
  ],
});